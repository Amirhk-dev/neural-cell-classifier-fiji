"""Classify a whole folder of images, with the CPU stages overlapped across them.

One image's run has three costs: reading the CZI and fitting BaSiC (CPU, minutes),
CellPose plus the two ResNet-18s (GPU), and per-cell extraction (CPU, the largest
share -- see :mod:`neural_imgs.inference.patch_extraction`). Run back-to-back,
the GPU idles through every image's read-and-fit and the CPU idles through every
CellPose call.

So the batch is a **pipeline, not a pool of independent runs**::

    image A: [read+BaSiC]--[GPU + extract]
    image B:      [read+BaSiC]-------------[GPU + extract]
    image C:            [read+BaSiC]--------------------[GPU + extract]
                  ^ prepare lane (threads)   ^ one GPU lane, never contended

Preparation of the next images runs on a background thread pool while the
current image holds the GPU. The GPU stage stays strictly single-threaded: there
is one device and one copy of each model, so a second concurrent CellPose call
would contend rather than add throughput -- and a second worker *process* would
need its own copy of CellPose and both ResNets in GPU memory.

``prefetch`` bounds how many prepared images may wait. Each one holds its whole
patch array (~840 MB for a 16-patch OPC CZI), so this is the batch's memory knob;
the default of 1 is enough to keep the GPU lane fed.

Cache hits skip preparation entirely -- they never read the CZI -- so a re-run
over a folder where most images are already done costs seconds, and only the
genuinely new images are prepared.

**One image's failure is not the batch's.** Each image is isolated: a bad file,
an out-of-memory patch or a montage that cannot render is recorded as a failed
:class:`ImageOutcome` and the remaining images still run. The caller gets one
outcome per image, in input order, whatever happened.
"""

from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from neural_imgs.inference import reporting
from neural_imgs.inference.case_montage import save_case_montages
from neural_imgs.inference.fixed_model_pipeline import (
    ClassifyImageInput,
    ClassifyImageOutput,
    LoadedModels,
    PatchedImage,
    PatchPreprocessor,
    classify_image,
    load_patches,
)
from neural_imgs.inference.result_cache import CacheKey, ClassificationCache
from neural_imgs.inference.rfp import RfpConfig

#: Extensions considered images when scanning an input folder.
IMAGE_SUFFIXES = (".czi",)

ProgressFn = Callable[[str, "int | None", "int | None"], None]
ImageStatus = Literal["classified", "cached", "failed"]


def discover_images(
    input_dir: str | Path, suffixes: tuple[str, ...] = IMAGE_SUFFIXES,
) -> list[Path]:
    """Images directly inside ``input_dir``, sorted by name.

    Not recursive, and hidden files are skipped: a folder of images is the unit
    the user picked, and silently walking into subfolders would turn "this
    plate" into "everything under it". Sorted so a batch's cell ids and its log
    are reproducible run to run.
    """
    input_dir = Path(input_dir)
    lowered = tuple(s.lower() for s in suffixes)
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and not p.name.startswith(".")
        and p.suffix.lower() in lowered
    )


@dataclass
class BatchInput:
    """What to classify, where to put it, and how the run is configured.

    ``out_root`` gets one subdirectory per image, named after the image stem:
    every artefact of that image -- cache entry, mask sidecar, per-cell CSV,
    quantification, UpSet plot, case montages -- lands inside it. Keeping them
    separated is what makes a folder run re-runnable per image: the cache is
    looked up by image stem, so two images with the same stem in one flat
    directory would otherwise overwrite each other's results.
    """

    image_paths: list[str]
    out_root: str
    model_dir: str
    patch_grid: int = 4
    flow_threshold: float = 0.2
    cellprob_threshold: float = 0.1
    rfp: RfpConfig = field(default_factory=RfpConfig)
    reuse_cache: bool = True
    generate_upset: bool = True
    show_case_montage: bool = False
    n_workers: int | None = None
    #: Prepared-but-not-yet-classified images allowed to wait. Memory knob.
    prefetch: int = 1

    def out_dir_for(self, image_path: str | Path) -> Path:
        return Path(self.out_root) / Path(image_path).stem


@dataclass
class ImageOutcome:
    """What happened to one image. Always produced, success or failure."""

    image_path: str
    out_dir: str
    status: ImageStatus
    detail: str = ""
    n_cells: int = 0
    n_opc_pos: int = 0
    n_b3tub_pos: int = 0
    n_rfp_pos: int = 0
    elapsed_s: float = 0.0
    csv_path: str = ""
    cache_path: str = ""
    quantification: list[str] = field(default_factory=list)
    quantification_path: str = ""
    upset_plot_path: str = ""
    case_montage_paths: list[str] = field(default_factory=list)
    #: Non-fatal problems: a montage or UpSet plot that failed while the
    #: classification itself succeeded. Never empties ``cells``.
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status != "failed"

    def summary_line(self) -> str:
        name = Path(self.image_path).name
        if self.status == "failed":
            return f"{name}: FAILED -- {self.detail}"
        return (f"{name}: {self.n_cells} cells, OPC+ {self.n_opc_pos}, "
                f"B3Tub+ {self.n_b3tub_pos}, RFP+ {self.n_rfp_pos} "
                f"({self.status}, {self.elapsed_s:.1f}s)")


@dataclass
class BatchOutput:
    """One outcome per input image, in input order, plus the cells themselves.

    ``cells_by_image`` is keyed by image path so a caller can render a combined
    table without re-reading anything; a failed image is absent from it but
    still present in ``outcomes``.
    """

    outcomes: list[ImageOutcome]
    cells_by_image: dict[str, ClassifyImageOutput]
    elapsed_s: float

    @property
    def n_failed(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "failed")

    def summary_lines(self) -> list[str]:
        lines = [o.summary_line() for o in self.outcomes]
        done = len(self.outcomes) - self.n_failed
        lines.append(f"Batch: {done}/{len(self.outcomes)} images succeeded "
                     f"in {self.elapsed_s:.1f}s")
        return lines


@dataclass
class _Prepared:
    """One image read and BaSiC-fitted, waiting for the GPU lane."""
    patched: PatchedImage
    preprocessor: PatchPreprocessor


class BatchClassifier:
    """Runs :class:`BatchInput` against already-loaded models.

    Models are loaded once by the caller and shared across every image -- that
    is the whole reason this is one process rather than one per image.
    """

    def __init__(
        self, models: LoadedModels,
        on_progress: ProgressFn | None = None,
        on_image_done: Callable[[ImageOutcome], None] | None = None,
    ) -> None:
        self.models = models
        self._report = on_progress or (lambda *_: None)
        self._on_image_done = on_image_done or (lambda _: None)

    # -- preparation lane ---------------------------------------------------

    def _prepare(self, image_path: str, patch_grid: int, gate: threading.Semaphore) -> _Prepared:
        """Read one CZI and fit BaSiC. Runs on the prefetch pool.

        ``gate`` is acquired *here*, not at submission: a task that has not
        started yet costs nothing, while a finished one pins its whole patch
        array. Released by the consumer once the image has been classified.
        """
        gate.acquire()
        try:
            patched = load_patches(image_path, patch_grid)
            preprocessor = PatchPreprocessor.fit(patched)
            return _Prepared(patched=patched, preprocessor=preprocessor)
        except BaseException:
            gate.release()   # nothing will be consumed, so nothing will release it
            raise

    # A permit is held from "started preparing" to "handed to the GPU lane", NOT
    # to the end of that image's classification -- releasing it late is the
    # difference between a pipeline and a serial loop, because the next image
    # cannot begin reading until a permit is free. So `prefetch` bounds the
    # images *waiting*, and one more (the one being classified) is alive on top
    # of that: peak is prefetch + 1 patch arrays, ~1.7 GB at the default.

    # -- per-image outputs --------------------------------------------------

    def _write_outputs(
        self, image_path: str, out_dir: Path, result: ClassifyImageOutput,
        input: BatchInput, outcome: ImageOutcome,
        prepared: "_Prepared | None" = None,
    ) -> None:
        """CSV + quantification always; UpSet and montages only if asked.

        Everything here sits on top of work that costs minutes to hours, so a
        failure in the optional parts is recorded as a warning and never
        propagates -- the classification is already saved by the time this runs.
        """
        stem, label = Path(image_path).stem, Path(image_path).name

        outcome.csv_path = str(reporting.save_cells_csv(
            result.cells, out_dir / f"{stem}_fixed_model_classifications.csv"))

        summary = reporting.quantify(result.cells)
        outcome.quantification = summary.lines()
        outcome.quantification_path = str(reporting.save_quantification(
            summary, out_dir / f"{stem}_quantification.txt", label))

        if input.generate_upset:
            try:
                path = out_dir / f"{stem}_upset_plot.png"
                reporting.save_upset_plot(result.cells, path, image_label=label)
                outcome.upset_plot_path = str(path)
            except Exception as exc:  # noqa: BLE001
                outcome.warnings.append(f"UpSet plot failed: {type(exc).__name__}: {exc}")

        if input.show_case_montage:
            try:
                # The montages show the illumination-corrected pixels the cells
                # were detected on, so they need that image's BaSiC fit. A
                # freshly classified image still has it (``prepared``); a cached
                # one never read the CZI at all and has to now -- minutes,
                # against the hours a re-classification would cost.
                fitted = prepared or self._prepared_for(image_path, input.patch_grid)
                montage = save_case_montages(
                    result.cells, fitted.preprocessor, out_dir, stem, image_label=label)
                outcome.case_montage_paths = [str(p) for p in montage.paths.values()]
            except Exception as exc:  # noqa: BLE001
                outcome.warnings.append(
                    f"case montages failed: {type(exc).__name__}: {exc}")

    def _prepared_for(self, image_path: str, patch_grid: int) -> _Prepared:
        """Read + fit on demand, for a cached image whose pixels are needed."""
        self._report(f"Reading {Path(image_path).name} for the case montages...", None, None)
        patched = load_patches(image_path, patch_grid)
        return _Prepared(patched=patched, preprocessor=PatchPreprocessor.fit(patched))

    # -- the run ------------------------------------------------------------

    def run(self, input: BatchInput) -> BatchOutput:
        t_batch = time.perf_counter()
        n = len(input.image_paths)
        outcomes: list[ImageOutcome] = []
        cells_by_image: dict[str, ClassifyImageOutput] = {}

        # Cache lookups first, for every image: a hit must not cause a CZI read,
        # so the prefetch lane below is only given the images that will actually
        # be classified.
        lookups: dict[str, object] = {}
        for image_path in input.image_paths:
            out_dir = input.out_dir_for(image_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            if not input.reuse_cache:
                lookups[image_path] = None
                continue
            try:
                key = CacheKey.build(
                    image_path, input.patch_grid, input.model_dir,
                    input.flow_threshold, input.cellprob_threshold, input.rfp)
                lookups[image_path] = ClassificationCache(out_dir).load(key)
            except Exception as exc:  # noqa: BLE001 -- a bad cache never blocks a run
                self._report(f"Cache check failed for {Path(image_path).name}: {exc}",
                             None, None)
                lookups[image_path] = None

        def is_hit(path: str) -> bool:
            lookup = lookups.get(path)
            return lookup is not None and getattr(lookup, "status", None) == "hit"

        to_prepare = [p for p in input.image_paths if not is_hit(p)]
        gate = threading.Semaphore(max(1, input.prefetch))
        futures: dict[str, Future] = {}

        with ThreadPoolExecutor(max(1, input.prefetch), thread_name_prefix="prepare") as pool:
            for image_path in to_prepare:
                futures[image_path] = pool.submit(
                    self._prepare, image_path, input.patch_grid, gate)

            for index, image_path in enumerate(input.image_paths, start=1):
                label = Path(image_path).name
                out_dir = input.out_dir_for(image_path)
                self._report(f"[{index}/{n}] {label}", index - 1, n)
                outcome = ImageOutcome(
                    image_path=image_path, out_dir=str(out_dir), status="classified")
                t_image = time.perf_counter()
                prepared: _Prepared | None = None
                try:
                    lookup = lookups.get(image_path)
                    if is_hit(image_path):
                        result = lookup.result.output  # type: ignore[union-attr]
                        outcome.status = "cached"
                        outcome.detail = lookup.detail  # type: ignore[union-attr]
                        outcome.cache_path = str(
                            ClassificationCache(out_dir).path_for(image_path))
                        self._report(f"[{index}/{n}] {label}: loaded saved results", index, n)
                    else:
                        if lookup is not None:
                            outcome.detail = f"re-classified: {lookup.detail}"  # type: ignore[union-attr]
                        prepared = futures[image_path].result()
                        futures.pop(image_path, None)
                        # Released the moment it is in hand: the next image's
                        # read + BaSiC fit now overlaps this image's GPU work,
                        # which is the entire point of the prepare lane.
                        gate.release()
                        result = classify_image(
                            ClassifyImageInput(
                                image_path=image_path, patch_grid=input.patch_grid,
                                flow_threshold=input.flow_threshold,
                                cellprob_threshold=input.cellprob_threshold,
                                rfp=input.rfp, n_workers=input.n_workers,
                            ),
                            self.models, patched=prepared.patched,
                            preprocessor=prepared.preprocessor,
                            on_progress=lambda m, c, mx, _l=f"[{index}/{n}] {label}: ":
                                self._report(_l + m, c, mx),
                        )
                        # Saved before anything optional runs: these numbers cost
                        # hours and a montage failure must not lose them.
                        key = CacheKey.build(
                            image_path, input.patch_grid, input.model_dir,
                            input.flow_threshold, input.cellprob_threshold, input.rfp)
                        outcome.cache_path = str(
                            ClassificationCache(out_dir).save(key, result))

                    cells_by_image[image_path] = result
                    outcome.n_cells = len(result.cells)
                    outcome.n_opc_pos = sum(1 for c in result.cells if c.opc_pos)
                    outcome.n_b3tub_pos = sum(1 for c in result.cells if c.b3tub_pos)
                    outcome.n_rfp_pos = sum(1 for c in result.cells if c.rfp_pos)
                    self._write_outputs(image_path, out_dir, result, input, outcome,
                                        prepared=prepared)
                except Exception as exc:  # noqa: BLE001 -- one bad image, not a dead batch
                    outcome.status = "failed"
                    outcome.detail = f"{type(exc).__name__}: {exc}"
                    self._report(f"[{index}/{n}] {label}: FAILED -- {outcome.detail}", index, n)
                    traceback.print_exc()
                finally:
                    # The patch array is the batch's largest single allocation;
                    # dropping the reference here rather than letting it live to
                    # the next iteration keeps two images' worth from overlapping.
                    prepared = None
                    outcome.elapsed_s = time.perf_counter() - t_image
                    outcomes.append(outcome)
                    self._on_image_done(outcome)

            # A batch that failed early leaves prepare tasks queued; cancelling
            # them keeps the pool from reading CZIs nobody will classify.
            for future in futures.values():
                future.cancel()

        return BatchOutput(outcomes=outcomes, cells_by_image=cells_by_image,
                           elapsed_s=time.perf_counter() - t_batch)
