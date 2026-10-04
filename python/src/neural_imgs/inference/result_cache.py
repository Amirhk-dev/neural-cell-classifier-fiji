"""On-disk cache of a completed classification run.

Classifying one CZI is a multi-hour job (BaSiC + CellPose + two ResNet-18s over
every detected nucleus), and nothing about it is interactive: given the same
image, the same settings and the same models, the answer never changes. This
module writes that answer next to the run's other outputs so a second look at
the same image -- a new Fiji session, a re-opened ROI set, a regenerated UpSet
plot -- costs seconds instead of hours.

Each run writes two files into the run's **output directory**, both named after
the image:

* ``<image_stem>_classification_cache.json`` -- the key (see :class:`CacheKey`)
  plus every cell's ids, bounding box, probabilities, marker scores
  and calls.
* ``<image_stem>_classification_masks.npz`` -- the per-cell native DAPI masks,
  bit-packed and compressed. They are far too bulky for the JSON but are what
  the per-case single-cell montages outline, so a cached run would otherwise
  render worse-looking figures than a fresh one. A missing or mismatched
  sidecar is not an error: the cells simply come back with ``native_mask=None``
  and callers degrade gracefully.

That directory is chosen by the user, is writable by definition, and keeps
everything about one run together; the CZI's own directory is often a read-only
share, and a hidden per-user cache would be invisible and unclean-able.

A cache entry is only reused when it provably describes the *same computation*
(see :class:`CacheKey`); any mismatch reports ``stale`` and the caller re-runs
rather than silently serving numbers from different settings or different
weights. Moving the CZI to another folder is explicitly not such a mismatch --
its location is not an input to the classification.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

from neural_imgs.inference.fixed_model_pipeline import (
    CellResult,
    ClassifyImageOutput,
    MARKERS,
    prod_exp_name,
)
from neural_imgs.inference.rfp import RfpConfig

# 2: cache keys gained the CellPose flow/cellprob thresholds, which change which
# nuclei are detected at all -- v1 entries cannot be told apart by them and must
# not be reused.
# 3: the RFP call moved from the legacy rule at gate 10 to the neighbour-excluded
# rule at r=60 / gate 16.5 (see neural_imgs.inference.rfp). Every v2 entry holds
# ``rfp_pos`` from the old rule, and the new score cannot be recovered from the
# file -- it needs the normalised RFP pixels, which a cache entry does not store
# -- so those entries are stale by construction, not upgradeable. The key also
# gained ``rfp_fingerprint`` so a later change of gate or radius invalidates on
# its own without another version bump.
CACHE_VERSION = 3

CacheStatus = Literal["hit", "miss", "stale", "unreadable"]


def model_fingerprint(model_dir: str | Path) -> str:
    """Cheap identity of the model bundle in ``model_dir``.

    Combines the decision thresholds (read in full -- they are the values most
    likely to be deliberately re-tuned in place) with each checkpoint's byte
    size. Deliberately does NOT use mtimes: re-downloading an identical bundle
    would change them and force a needless multi-hour re-run, while a genuinely
    different checkpoint is essentially certain to differ in thresholds or size.
    """
    model_dir = Path(model_dir)
    thresholds = json.loads((model_dir / "thresholds.json").read_text())
    parts = [f"thresholds={json.dumps(thresholds, sort_keys=True)}"]
    for marker in MARKERS:
        ckpt = model_dir / prod_exp_name(marker) / "best_model.pt"
        size = ckpt.stat().st_size if ckpt.exists() else -1
        parts.append(f"{marker}={size}")
    return "|".join(parts)


@dataclass(frozen=True)
class CacheKey:
    """Everything that has to match for cached results to still be correct.

    ``image_path`` is recorded but deliberately **not** compared: where a file
    sits is not an input to the classification, so relocating a CZI must not
    cost a multi-hour re-run. What identifies the image instead is its name
    (the cache file is looked up as ``<image_stem>_classification_cache.json``)
    together with ``image_size``, which catches a *different* CZI arriving under
    a name already classified -- including one written over the same path. Two
    genuinely different CZIs sharing both a filename and an exact byte size,
    classified into one output directory, is the residual case this cannot tell
    apart; untick "Load saved results if available" if that is ever a real risk.

    The CellPose thresholds are part of the key because they decide which nuclei
    exist in the first place -- every count in the run is downstream of them.
    """
    image_path: str
    image_size: int
    patch_grid: int
    flow_threshold: float
    cellprob_threshold: float
    models_fingerprint: str
    # Which RFP rule and gate produced ``rfp_pos``. RFP has no trained model, so
    # models_fingerprint says nothing about it -- without this, changing the gate
    # would silently serve calls made at the old one.
    rfp_fingerprint: str = RfpConfig().fingerprint()

    @classmethod
    def build(
        cls, image_path: str | Path, patch_grid: int, model_dir: str | Path,
        flow_threshold: float, cellprob_threshold: float,
        rfp: RfpConfig | None = None,
    ) -> "CacheKey":
        image_path = Path(image_path)
        return cls(
            image_path=str(image_path.resolve()),
            image_size=image_path.stat().st_size,
            patch_grid=int(patch_grid),
            flow_threshold=float(flow_threshold),
            cellprob_threshold=float(cellprob_threshold),
            models_fingerprint=model_fingerprint(model_dir),
            rfp_fingerprint=(rfp or RfpConfig()).fingerprint(),
        )

    def mismatch_reason(self, other: "CacheKey") -> str | None:
        """Human-readable first *disqualifying* difference against ``other``, or
        None if ``other``'s results still describe this computation.

        A differing ``image_path`` is not disqualifying -- see the class
        docstring, and :meth:`moved_from` for reporting it.
        """
        if self.image_size != other.image_size:
            return f"image file changed ({other.image_size} -> {self.image_size} bytes)"
        if self.patch_grid != other.patch_grid:
            return f"patch grid {other.patch_grid} -> {self.patch_grid}"
        if self.flow_threshold != other.flow_threshold:
            return f"flow threshold {other.flow_threshold:g} -> {self.flow_threshold:g}"
        if self.cellprob_threshold != other.cellprob_threshold:
            return (f"cell probability threshold {other.cellprob_threshold:g} -> "
                    f"{self.cellprob_threshold:g}")
        if self.models_fingerprint != other.models_fingerprint:
            return "model bundle changed (different checkpoints or thresholds)"
        if self.rfp_fingerprint != other.rfp_fingerprint:
            return f"RFP rule changed ({other.rfp_fingerprint} -> {self.rfp_fingerprint})"
        return None

    def moved_from(self, other: "CacheKey") -> str | None:
        """``other``'s path if the image has been relocated since, else None.

        Reused results are still correct after a move; this only exists so the
        log can say the results came from an entry written elsewhere, rather
        than leaving a silently surprising hit.
        """
        return other.image_path if self.image_path != other.image_path else None


@dataclass
class CachedClassification:
    """A previously computed :class:`ClassifyImageOutput` plus its provenance.

    ``masks_restored`` says whether the cells carry their ``native_mask``; when
    False the sidecar was absent or did not match, and mask-dependent rendering
    has to do without.
    """
    key: CacheKey
    created_at: str  # ISO-8601, local time
    output: ClassifyImageOutput
    masks_restored: bool = False


@dataclass
class CacheLookup:
    """Outcome of :meth:`ClassificationCache.load`.

    ``status`` is ``hit`` (``result`` is set), ``miss`` (no cache file),
    ``stale`` (a cache file exists but describes a different computation) or
    ``unreadable`` (present but corrupt / written by another version).
    ``detail`` explains the non-hit cases well enough to log verbatim.
    """
    status: CacheStatus
    detail: str
    result: CachedClassification | None = None


def _cell_to_dict(cell: CellResult) -> dict:
    return {
        "cell_id": int(cell.cell_id),
        "patch_idx": int(cell.patch_idx),
        "bb_position": [int(v) for v in cell.bb_position],
        "opc_prob": float(cell.opc_prob),
        "opc_pos": bool(cell.opc_pos),
        "b3tub_prob": float(cell.b3tub_prob),
        "b3tub_pos": bool(cell.b3tub_pos),
        "rfp_pos": bool(cell.rfp_pos),
        "rfp_score": float(cell.rfp_score),
        "opc_channel_score": float(cell.opc_channel_score),
        "b3tub_channel_score": float(cell.b3tub_channel_score),
        "rfp_method": str(cell.rfp_method),
        "rfp_neighbour_score": float(cell.rfp_neighbour_score),
        "rfp_neighbour_pos": bool(cell.rfp_neighbour_pos),
        "rfp_legacy_pos": bool(cell.rfp_legacy_pos),
        "n_cells_in_region": int(cell.n_cells_in_region),
    }


def _cell_from_dict(d: dict) -> CellResult:
    return CellResult(
        cell_id=int(d["cell_id"]),
        patch_idx=int(d["patch_idx"]),
        bb_position=tuple(int(v) for v in d["bb_position"]),
        opc_prob=float(d["opc_prob"]),
        opc_pos=bool(d["opc_pos"]),
        b3tub_prob=float(d["b3tub_prob"]),
        b3tub_pos=bool(d["b3tub_pos"]),
        rfp_pos=bool(d["rfp_pos"]),
        native_mask=None,  # filled in from the sidecar, when there is one
        # Absent from entries written before the continuous marker scores
        # existed. Read with a default rather than behind a CACHE_VERSION bump:
        # the stored calls and probabilities are unchanged, so invalidating
        # those entries would cost a multi-hour re-run per image to recover
        # numbers the file already holds. Cells come back with NaN scores, the
        # same graceful degradation the mask sidecar already uses.
        rfp_score=float(d.get("rfp_score", float("nan"))),
        opc_channel_score=float(d.get("opc_channel_score", float("nan"))),
        b3tub_channel_score=float(d.get("b3tub_channel_score", float("nan"))),
        # Written by every v3 entry; the defaults only matter if a future
        # version ever reads a partial file.
        rfp_method=str(d.get("rfp_method", "neighbour")),
        rfp_neighbour_score=float(d.get("rfp_neighbour_score", float("nan"))),
        rfp_neighbour_pos=bool(d.get("rfp_neighbour_pos", False)),
        rfp_legacy_pos=bool(d.get("rfp_legacy_pos", False)),
        n_cells_in_region=int(d.get("n_cells_in_region", -1)),
    )


class MaskSidecar:
    """The per-cell native DAPI masks of one cache entry, in a single ``.npz``.

    Masks are ragged (one bounding box per cell), so they are bit-packed and
    concatenated into one flat buffer with an offset table rather than stored as
    thousands of separate arrays -- an order of magnitude faster to write and
    read, and a fraction of the size.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, cells: list[CellResult]) -> Path | None:
        """Write the masks of ``cells``; returns None if none of them has one."""
        masks = [c.native_mask for c in cells]
        if not any(m is not None for m in masks):
            return None

        packed: list[np.ndarray] = []
        offsets = [0]
        shapes = []
        for mask in masks:
            if mask is None:
                shapes.append((0, 0))
            else:
                mask = np.ascontiguousarray(mask, dtype=bool)
                packed.append(np.packbits(mask.reshape(-1)))
                shapes.append(mask.shape)
            offsets.append(offsets[-1] + (packed[-1].size if mask is not None else 0))

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".npz.tmp")
        # Written through an open handle, not a path: savez_compressed appends
        # ".npz" to any path that does not already end in it, which would put
        # the file somewhere other than where we then move it from.
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                packed=np.concatenate(packed) if packed else np.zeros(0, dtype=np.uint8),
                offsets=np.asarray(offsets, dtype=np.int64),
                shapes=np.asarray(shapes, dtype=np.int64),
                cell_ids=np.asarray([int(c.cell_id) for c in cells], dtype=np.int64),
            )
        tmp.replace(self.path)
        return self.path

    def restore_into(self, cells: list[CellResult]) -> bool:
        """Attach saved masks to ``cells`` in place; False if unavailable/mismatched."""
        if not self.path.exists():
            return False
        try:
            with np.load(self.path) as data:
                offsets, shapes = data["offsets"], data["shapes"]
                cell_ids, packed = data["cell_ids"], data["packed"]
                if len(cell_ids) != len(cells):
                    return False
                if any(int(a) != int(b.cell_id) for a, b in zip(cell_ids, cells)):
                    return False
                for i, cell in enumerate(cells):
                    h, w = int(shapes[i][0]), int(shapes[i][1])
                    if h == 0 or w == 0:
                        continue
                    bits = packed[int(offsets[i]):int(offsets[i + 1])]
                    cell.native_mask = (
                        np.unpackbits(bits, count=h * w).reshape(h, w).astype(np.uint8)
                    )
        except Exception:  # noqa: BLE001 -- a bad sidecar only costs the contours
            return False
        return True


class ClassificationCache:
    """Reads/writes one cache entry per image inside a single output directory."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)

    def path_for(self, image_path: str | Path) -> Path:
        return self.out_dir / f"{Path(image_path).stem}_classification_cache.json"

    def masks_path_for(self, image_path: str | Path) -> Path:
        return self.out_dir / f"{Path(image_path).stem}_classification_masks.npz"

    def save(self, key: CacheKey, output: ClassifyImageOutput) -> Path:
        """Write ``output`` as the cache entry for ``key``; returns the JSON path.

        Both files are written to a temporary name and moved into place, so an
        interrupted write can never leave a half-file that a later run would
        treat as a hit. The masks go first: the JSON is what ``load`` looks for,
        so it must not exist before the sidecar it advertises.
        """
        path = self.path_for(key.image_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        MaskSidecar(self.masks_path_for(key.image_path)).save(output.cells)

        payload = {
            "cache_version": CACHE_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "key": asdict(key),
            "image_shape": [int(v) for v in output.image_shape],
            "cells": [_cell_to_dict(c) for c in output.cells],
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)
        return path

    def load(self, key: CacheKey) -> CacheLookup:
        """Look up the entry for ``key``, never raising on a bad cache file."""
        path = self.path_for(key.image_path)
        if not path.exists():
            return CacheLookup("miss", f"no saved results at {path}")
        try:
            payload = json.loads(path.read_text())
            if payload.get("cache_version") != CACHE_VERSION:
                return CacheLookup(
                    "unreadable",
                    f"saved results use cache version {payload.get('cache_version')}, "
                    f"this build expects {CACHE_VERSION}",
                )
            stored_key = CacheKey(**payload["key"])
            output = ClassifyImageOutput(
                image_shape=tuple(payload["image_shape"]),
                cells=[_cell_from_dict(d) for d in payload["cells"]],
            )
            created_at = str(payload["created_at"])
        except Exception as exc:  # noqa: BLE001 -- a bad cache must never break a run
            return CacheLookup("unreadable", f"could not read {path}: {exc}")

        reason = key.mismatch_reason(stored_key)
        if reason is not None:
            return CacheLookup("stale", f"saved results do not match: {reason}")

        masks_restored = MaskSidecar(
            self.masks_path_for(key.image_path)
        ).restore_into(output.cells)

        # The entry keeps the path it was written with; it is not rewritten here,
        # so a load stays a pure read.
        moved_from = key.moved_from(stored_key)
        detail = f"{len(output.cells)} cells, saved {created_at}"
        if moved_from is not None:
            detail += f", image moved from {moved_from}"
        if not masks_restored:
            detail += ", without cell masks"

        return CacheLookup(
            "hit", detail,
            CachedClassification(
                key=stored_key, created_at=created_at, output=output,
                masks_restored=masks_restored,
            ),
        )
