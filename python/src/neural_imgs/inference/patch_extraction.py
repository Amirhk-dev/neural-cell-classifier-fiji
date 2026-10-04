"""Turning one image's patches into scored single cells, with the CPU work overlapped.

Profile of the serial loop on a 16-patch OPC CZI (2074 cells, CUDA node):
BaSiC fit 1.5 s | illumination ~16 s (CPU) | CellPose ~82 s (GPU) |
**extraction ~111 s (CPU)**. The bottleneck is the CPU extraction, not CellPose,
so that is what is parallelised here:

* :class:`FastCellExtractor` replaces ``extract_cells_512_nodistort``'s
  ``np.where(masks == lab)`` scan-per-label -- ~130 full passes over a 2000x2000
  array per patch -- with a single ``find_objects`` pass plus ``bincount`` for
  the areas. Bit-identical output, verified cell-by-cell on bounding boxes,
  masks, pixels and ``analyze_image`` results.
* :class:`ParallelPatchDetector` keeps CellPose serial on the calling thread --
  one device, one model, so several workers would only contend -- and hands each
  patch's CPU work to a thread pool, so extraction of patch *i* overlaps CellPose
  on patch *i+1*. Threads, not processes: numpy and skimage release the GIL for
  this workload, while processes would have to pickle gigabytes of cell dicts
  back to the parent.

**Determinism.** CellPose runs in patch order on one thread and the per-patch
results are reassembled in patch order, so ``cells`` -- and therefore every
``cell_id`` -- is identical to the serial loop's, whatever the pool does.

**Memory.** The deployed path keeps no per-cell display crops at all. The RFP
heuristic needs the normalised RFP plane, so it is scored *inside* the worker,
per patch, from views onto the patch that is alive anyway; the views die with
the patch. Retaining the five crops instead (what the notebook does, for its
montages) costs ~1.3 MB per cell -- several GB on a crowded image, which is the
difference between running on a laptop and not.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import ndimage as ndi
from skimage.transform import resize as sk_resize

from neural_imgs.inference.rfp import RFP_CHANNEL, RfpConfig, score_rfp_neighbour
from neural_imgs.positivity import neighbour_mask_from_labels
from neural_imgs.processing.processing import (
    analyze_image,
    correct_illumination,
    detect_cells_on_DAPI,
    percentile_normalization,
    zero_out_image_border,
)

#: Keys the notebook stores its normalised per-channel crops under, in channel order.
DISPLAY_CROP_KEYS = ("dapi_crop", "opc_crop", "rfp_crop", "b3tub_crop", "bf_crop")


@dataclass(frozen=True)
class CellExtractionConfig:
    """Geometry and analysis knobs for turning one patch's labels into cells.

    Parameters
    ----------
    pad / min_area / boundary_margin / target:
        ``extract_cells_512_nodistort``'s arguments, unchanged.
    border / percent:
        ``zero_out_image_border`` and ``analyze_image``, applied to the 512 view
        for the legacy RFP score.
    p_low / p_high / eps_norm:
        ``percentile_normalization``'s arguments -- global, not per crop.
    rfp:
        Operating point of the RFP rules. ``analyze_image``'s ``eps`` comes from
        it, so the legacy gate cannot drift from the score it gates; when
        ``method == "neighbour"`` the neighbour score is computed here too.
    keep_display_crops:
        Retain the five normalised ``*_crop`` arrays per cell (~1.3 MB each).
        Off by default: the per-case montages rebuild them on demand from the
        BaSiC fit, so nothing downstream of a deployed run reads them.
    """

    pad: int = 100
    min_area: int = 100
    boundary_margin: int = 20
    target: int = 512
    border: int = 100
    percent: int = 95
    p_low: float = 0.1
    p_high: float = 99.9
    eps_norm: float = 1e-8
    rfp: RfpConfig = field(default_factory=RfpConfig)
    keep_display_crops: bool = False


@dataclass
class PatchDetectionResult:
    """Every cell of one image, plus where the time went."""

    cells: list[dict]
    elapsed_s: float
    n_workers: int
    per_stage_s: dict[str, float]

    def timing_line(self) -> str:
        stages = "  ".join(f"{k} {v:.1f}s" for k, v in self.per_stage_s.items())
        return (f"{len(self.cells)} cells in {self.elapsed_s:.1f}s "
                f"({self.n_workers} workers)  [{stages}]")


class FastCellExtractor:
    """One patch's labels -> its scored cell dicts. Pure CPU, no shared state.

    Free of shared mutable state by construction, so several :meth:`extract`
    calls can run concurrently on a thread pool. The only per-call object with
    state is the RFP scorer, which is constructed inside the call.
    """

    def __init__(
        self, config: CellExtractionConfig, channel_names: list[str],
        target_channels: list[int],
    ) -> None:
        self.config = config
        self.channel_names = channel_names
        self.target_channels = target_channels

    def _bounding_boxes(self, masks: np.ndarray):
        """``(label, y_min, y_max, x_min, x_max)`` per label, ascending.

        Equivalent to ``np.unique`` + ``np.where(masks == lab)`` per label,
        ``min_area`` filter included, in one pass over the array rather than one
        pass per label.
        """
        objects = ndi.find_objects(masks)
        areas = np.bincount(masks.ravel())
        for lab, sl in enumerate(objects, start=1):
            if sl is None or areas[lab] < self.config.min_area:
                continue
            yield lab, sl[0].start, sl[0].stop - 1, sl[1].start, sl[1].stop - 1

    def extract(
        self, pre: np.ndarray, raw: np.ndarray, masks: np.ndarray, patch_idx: int,
    ) -> list[dict]:
        """``pre`` = normalised patch, ``raw`` = the same patch's raw pixels."""
        cfg = self.config
        C, H, W = pre.shape
        cells: list[dict] = []

        for lab, y_min, y_max, x_min, x_max in self._bounding_boxes(masks):
            if (y_min < cfg.boundary_margin or y_max > H - cfg.boundary_margin or
                    x_min < cfg.boundary_margin or x_max > W - cfg.boundary_margin):
                continue

            y0 = max(y_min - cfg.pad, 0)
            y1 = min(y_max + cfg.pad + 1, H)
            x0 = max(x_min - cfg.pad, 0)
            x1 = min(x_max + cfg.pad + 1, W)

            label_crop = masks[y0:y1, x0:x1]
            labels_in_bb = np.unique(label_crop)
            n_cells_in_region = int(len(labels_in_bb[labels_in_bb != 0]))

            # --- resize to a 512 short side without distorting, then centre-crop ---
            img_crop = pre[:, y0:y1, x0:x1].copy()
            mask_crop = (label_crop == lab).astype(np.uint8)
            h, w = mask_crop.shape
            s = cfg.target / float(min(h, w))
            new_h, new_w = max(1, int(round(h * s))), max(1, int(round(w * s)))
            img_resized = np.stack([
                sk_resize(img_crop[c], (new_h, new_w), order=1,
                          anti_aliasing=True, preserve_range=True)
                for c in range(C)
            ], axis=0).astype(pre.dtype)
            mask_resized = sk_resize(mask_crop, (new_h, new_w), order=0,
                                     anti_aliasing=False,
                                     preserve_range=True).astype(np.uint8)
            y_start, x_start = (new_h - cfg.target) // 2, (new_w - cfg.target) // 2
            img_final = img_resized[:, y_start:y_start + cfg.target,
                                    x_start:x_start + cfg.target]
            mask_final = mask_resized[y_start:y_start + cfg.target,
                                      x_start:x_start + cfg.target]
            img_final[0] *= mask_final

            # --- legacy RFP heuristic: exactly the original 512 pipeline ----------
            img_final = zero_out_image_border(img_final, border=cfg.border)
            result = analyze_image(img_final, mask_final, self.channel_names,
                                   self.target_channels, percent=cfg.percent,
                                   eps=cfg.rfp.analyze_eps)

            # --- classifier input: RAW native crop + native DAPI mask -------------
            raw_img = raw[:, y0:y1, x0:x1].copy()
            rh, rw = raw_img.shape[1], raw_img.shape[2]
            native_mask = (sk_resize(mask_final.astype(float), (rh, rw), order=0,
                                     preserve_range=True) > 0.5).astype(np.uint8)

            cell = dict(
                label=int(lab),
                mask=mask_final,
                bb_position=[int(y0), int(y1), int(x0), int(x1)],
                n_cells_in_region=n_cells_in_region,
                patch_idx=patch_idx,
                result=result,
                raw_img=raw_img,
                native_mask=native_mask,
                neighbour_mask=neighbour_mask_from_labels(label_crop, lab),
            )
            # Views, not copies: `pre` outlives this call, and the RFP score below
            # is the only consumer -- unless display crops were explicitly asked
            # for, in which case they must own their pixels.
            if cfg.keep_display_crops:
                for idx, key in enumerate(DISPLAY_CROP_KEYS):
                    cell[key] = pre[idx, y0:y1, x0:x1].copy()
                cell["native_channels"] = [cell[k] for k in DISPLAY_CROP_KEYS]
            else:
                channels: list[np.ndarray | None] = [None] * C
                channels[RFP_CHANNEL] = pre[RFP_CHANNEL, y0:y1, x0:x1]
                cell["native_channels"] = channels
            cells.append(cell)

        self._score_rfp(cells)
        return cells

    def _score_rfp(self, cells: list[dict]) -> None:
        """Attach the neighbour-rule score, then drop what only it needed.

        Done per patch inside the worker so the normalised planes never have to
        outlive the patch: holding them until a whole-image scoring pass would
        cost gigabytes on a crowded image, and buys nothing -- the rule scores
        each cell from its own arrays alone.
        """
        if not cells:
            return
        scores, positive = score_rfp_neighbour(
            cells, self.config.rfp, tuple(self.channel_names),
        )
        for cell, score, pos in zip(cells, scores, positive):
            cell["rfp_neighbour_score"] = float(score)
            cell["rfp_neighbour_pos"] = bool(pos)
            if not self.config.keep_display_crops:
                del cell["native_channels"]


class ParallelPatchDetector:
    """CellPose serial on the calling thread; per-patch CPU extraction on a pool.

    The GPU/MPS stage stays single-threaded on purpose (one device, one model).
    What overlaps is the CPU extraction of already-detected patches with CellPose
    on the next patch, which is where ~53% of the serial runtime sat.
    """

    @staticmethod
    def available_cpus() -> int:
        """Cores this process may actually use.

        ``os.cpu_count()`` reports the machine's total, which on a shared HPC
        node or in a container can be far larger than this job's allocation --
        128 against an 8-core affinity mask on the node this was written on.
        Sizing the pool from that number oversubscribes every nested BLAS/OpenMP
        pool inside the workers too, which is enough to get the process killed.
        """
        try:
            return len(os.sched_getaffinity(0))   # Linux: taskset / cgroup / SLURM
        except AttributeError:
            return os.cpu_count() or 4            # macOS / Windows: no affinity mask

    def __init__(
        self, cellpose_model, extractor: FastCellExtractor,
        flow_threshold: float, cellprob_threshold: float,
        n_workers: int | None = None, max_inflight: int | None = None,
    ) -> None:
        self.cellpose_model = cellpose_model
        self.extractor = extractor
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.n_workers = n_workers or max(1, self.available_cpus())
        # Backpressure. Each queued patch keeps its own normalised copy alive
        # (5 x 2054 x 2558 float32 ~= 105 MB), so submitting all 16 at once pins
        # ~1.7 GB before one is collected. Capping the in-flight window costs
        # nothing in throughput while the workers stay busy.
        self.max_inflight = max_inflight or self.n_workers

    def run(
        self, patches: np.ndarray, basics: list,
        on_progress: Callable[[str, int | None, int | None], None] | None = None,
    ) -> PatchDetectionResult:
        report = on_progress or (lambda *_: None)
        cfg = self.extractor.config
        t0 = time.perf_counter()
        stage = {"illum": 0.0, "cellpose": 0.0, "gather": 0.0}
        by_patch: dict[int, list[dict]] = {}
        inflight: deque = deque()
        n_patches = len(patches)

        def collect_one() -> None:
            idx, fut = inflight.popleft()
            t = time.perf_counter()
            by_patch[idx] = fut.result()
            stage["gather"] += time.perf_counter() - t

        with open(os.devnull, "w") as devnull, ThreadPoolExecutor(self.n_workers) as pool:
            for patch_idx, patch in enumerate(patches):
                t = time.perf_counter()
                pre = correct_illumination(patch, basics)   # new array; patch untouched
                with contextlib.redirect_stdout(devnull):   # hide the noisy print
                    pre = percentile_normalization(pre, cfg.p_low, cfg.p_high, cfg.eps_norm)
                stage["illum"] += time.perf_counter() - t

                t = time.perf_counter()
                masks, _, _ = detect_cells_on_DAPI(
                    pre[0], self.cellpose_model, 1,
                    self.flow_threshold, self.cellprob_threshold,
                )
                stage["cellpose"] += time.perf_counter() - t

                # Queued, not awaited: this patch's CPU work overlaps the next
                # patch's CellPose call.
                inflight.append((patch_idx, pool.submit(
                    self.extractor.extract, pre, patch, masks, patch_idx)))
                report(f"Detected {int(masks.max())} nuclei in patch "
                       f"{patch_idx + 1}/{n_patches}...", patch_idx + 1, n_patches)
                while len(inflight) > self.max_inflight:
                    collect_one()

            while inflight:
                collect_one()

        # Reassembled in patch order -> cell_id identical to the serial loop's.
        cells = [c for i in range(n_patches) for c in by_patch[i]]
        return PatchDetectionResult(
            cells=cells, elapsed_s=time.perf_counter() - t0,
            n_workers=self.n_workers, per_stage_s=stage,
        )
