"""Fixed production single-cell classification pipeline.

Load a CZI -> patchify -> BaSiC illumination correction -> percentile
normalisation -> CellPose nucleus detection -> per-marker ResNet-18
classification (OPC, B3-Tub) -> RFP heuristic. This is the deployed inference
path: it mirrors ``notebooks/single_cell_classification_fixed_model.ipynb``
(cells 2-20) minus plotting/GradCAM/threshold-sweep code, so it can be called
from a plain script, a Fiji plugin's Python worker, or anywhere else that
needs the exact same numbers the notebook produces.

``PROD_CONFIG`` / ``build_raw_cfg`` / ``prod_exp_name`` are the single source
of truth for which checkpoint (crop size, soft-mask sigma) backs each marker;
``scripts/select_classifier_thresholds.py`` imports them rather than keeping
its own copy, so the frozen ``thresholds.json`` this module loads always
matches the models it actually loads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from neural_imgs.channels import (
    CLASSIFIER_MARKERS,
    DEFAULT_LAYOUT,
    DEFAULT_ORDER,
    MARKER_ROLES,
    ChannelLayout,
    MarkerAxes,
)
from neural_imgs.inference.patch_extraction import (
    CellExtractionConfig,
    FastCellExtractor,
    ParallelPatchDetector,
)
from neural_imgs.inference.rfp import RfpConfig
from neural_imgs.io.read_image import read_czi
from neural_imgs.processing.processing import (
    squeeze_img,
    patch_image,
    fit_illumination_profiles,
    correct_illumination,
    percentile_normalization,
)
from neural_imgs.utils.utils import best_device, get_cellpose_model, make_deterministic
from neural_imgs.training.model import CellClassifier
from neural_imgs.training.raw_dataset import RawCropConfig, raw_cell_to_tensor

# The historical fixed acquisition. Still the default for every entry point, so
# a caller that declares nothing behaves exactly as before -- but no longer the
# only possibility: a run carries a neural_imgs.channels.ChannelLayout saying
# which of its channels is which, and markers whose channel that layout does
# not name are reported as *not measured* rather than negative.
CHANNEL_NAMES = list(DEFAULT_ORDER)
TARGET_CHANNELS = list(DEFAULT_LAYOUT.target_channels)  # OPC, RFP, B3Tub
P_LOW, P_HIGH, EPS = 0.1, 99.9, 1e-8
MAX_ITERATIONS = 100  # BaSiC iterations
# CellPose nucleus-detection defaults. Overridable per run via
# ClassifyImageInput -- these two are the knobs that decide which nuclei exist
# at all, so every downstream count depends on them (and so does the cache key,
# see neural_imgs.inference.result_cache).
FLOW_THRESHOLD = 0.2      # mask accepted only if its flows match CellPose's; lower = stricter
CELLPROB_THRESHOLD = 0.1  # "is this pixel part of a cell?"; lower = more cells detected
SEED = 42

# The RFP rule and its gate now live in neural_imgs.inference.rfp, which the
# notebook's constants are mirrored into. The old module-level
# CONFIDENCE_THRESHOLD = 10 is gone on purpose: it gated the *legacy* score and
# was never updated when the notebook moved to the neighbour-excluded rule at
# 16.5, so a plugin run and a notebook run on the same image disagreed about
# which cells were transduced. Override per run via ClassifyImageInput.rfp.

# Fixed production configuration PER MARKER. MUST match
# scripts/train_raw_classifier_final.py and the checkpoint directory names
# produced by prod_exp_name().
PROD_CONFIG = {
    "OPC":   {"channel": "OPC",   "crop_size": 200, "mask_sigma": 32.0},
    "B3Tub": {"channel": "B3Tub", "crop_size": 200, "mask_sigma": 16.0},
}
MARKERS = tuple(PROD_CONFIG)


def build_raw_cfg(
    marker: str, layout: ChannelLayout = DEFAULT_LAYOUT
) -> RawCropConfig:
    """Inference preprocessing for a marker (single soft-masked channel; augment=False).

    ``channels`` indexes the *run's* crop, so it has to come from ``layout``:
    the classifier always sees one channel, but which plane of the image that
    is depends on the acquisition. Everything else -- crop size, soft-mask sigma
    -- is a property of the trained checkpoint and never moves.
    """
    c = PROD_CONFIG[marker]
    channel = layout.index_of(c["channel"])
    if channel is None:
        raise ValueError(
            f"{marker} cannot be classified: the run's channel layout "
            f"({layout.describe_order()}) has no {c['channel']} channel")
    return RawCropConfig(
        crop_size=c["crop_size"], channels=[channel],
        include_mask=False, apply_soft_mask=True, mask_sigma=c["mask_sigma"],
        augment=False,
    )


def prod_exp_name(marker: str) -> str:
    """Checkpoint dir name; MUST match scripts/train_raw_classifier_final.py.

    The ``_ch<N>`` part is the marker's index in the channel order the models
    were *trained* under (DEFAULT_ORDER), not in the layout of whatever image is
    being classified: it names a directory inside a published model bundle, so
    it has to be the same string on every run or the checkpoint is not found.
    """
    c = PROD_CONFIG[marker]
    idx = DEFAULT_ORDER.index(c["channel"])
    return (f"raw_{marker}_resnet18_crop{c['crop_size']}"
            f"_soft{c['mask_sigma']:g}_ch{idx}_prod")


@dataclass
class PatchedImage:
    """A CZI, squeezed and split into an NxN patch grid -- the shared starting
    point for both full-image classification and on-demand single-cell crop
    lookup (see :func:`crop_cell`), so both read the exact same pixels.

    ``layout`` is this image's channel layout, already reconciled with the
    file's actual channel count (:meth:`ChannelLayout.for_image`). It travels
    with the pixels rather than being re-derived downstream: the montages, the
    single-cell view and the classifier all have to agree about which plane is
    which, and an image is the thing that settles it.
    """
    patches: np.ndarray  # (N, C, H_patch, W_patch), pre-illumination-correction
    image_shape: tuple[int, int, int]  # (C, H, W) of the full squeezed image
    layout: ChannelLayout = DEFAULT_LAYOUT


def load_patches(
    image_path: str | Path, patch_grid: int,
    layout: ChannelLayout = DEFAULT_LAYOUT,
) -> PatchedImage:
    """Read one image and tile it, checking it against the declared layout.

    This is where a wrong channel declaration is caught -- before BaSiC, before
    CellPose, before anything that costs minutes -- and reported as the
    mismatch it is: how many channels the layout names against how many the
    file actually has. A file with *more* channels than the layout names is
    fine; the extra ones are read, corrected and ignored.
    """
    img = squeeze_img(read_czi(image_path))
    if img.ndim != 3:
        raise ValueError(
            f"expected a (C, H, W) image after squeezing, got shape {img.shape} "
            f"from {Path(image_path).name}")
    layout = layout.for_image(img.shape[0])
    patches = patch_image(img, patch_grid)
    return PatchedImage(
        patches=patches, image_shape=tuple(img.shape), layout=layout)


def crop_cell(
    patched: PatchedImage, patch_idx: int, bb_position: tuple[int, int, int, int]
) -> np.ndarray:
    """Raw (uncorrected) all-channel crop for one cell -- exactly the pixels
    ``classify_image`` fed to the classifier as ``raw_img``, e.g. for display."""
    y0, y1, x0, x1 = bb_position
    return patched.patches[patch_idx][:, y0:y1, x0:x1]


@dataclass
class PatchPreprocessor:
    """The BaSiC illumination fit for one image, plus the per-patch pixels it
    produces: correction + percentile normalisation to [0, 1].

    These are the pixels CellPose is run on and the pixels the notebook displays
    (its ``dapi_crop``/``opc_crop``/... come straight out of them), so keeping
    the fit as an object rather than a local in :func:`classify_image` is what
    lets a later step -- the per-case single-cell montages -- reproduce exactly
    what was classified instead of approximating it from raw pixels.

    The fit is the expensive part (minutes); applying it to one patch is cheap.
    """
    patched: PatchedImage
    basics: list

    @classmethod
    def fit(
        cls, patched: PatchedImage, max_iterations: int = MAX_ITERATIONS,
        on_progress: Callable[[str, int | None, int | None], None] | None = None,
    ) -> "PatchPreprocessor":
        report = on_progress or (lambda *_: None)
        report("Fitting illumination correction profiles...", None, None)
        return cls(patched=patched, basics=fit_illumination_profiles(
            patched.patches, max_iterations,
        ))

    @property
    def layout(self) -> ChannelLayout:
        return self.patched.layout

    def normalized_patch(self, patch_idx: int) -> np.ndarray:
        """One patch as (C, H, W) in [0, 1] -- corrected and percentile-normalised.

        Not cached: a normalised patch is a full float array per channel, so
        holding several would cost more memory than recomputing one costs time.
        Callers wanting several cells should group them by patch.
        """
        pre = correct_illumination(self.patched.patches[patch_idx], self.basics)
        return percentile_normalization(pre, P_LOW, P_HIGH, EPS)


@dataclass
class ClassifyImageInput:
    image_path: str
    patch_grid: int = 4  # NxN tiling -- same semantics as patch_image()'s patch_size arg
    # CellPose detection thresholds; see FLOW_THRESHOLD / CELLPROB_THRESHOLD.
    flow_threshold: float = FLOW_THRESHOLD
    cellprob_threshold: float = CELLPROB_THRESHOLD
    # Which RFP rule makes the call, and at what gate. See neural_imgs.inference.rfp.
    rfp: RfpConfig = field(default_factory=RfpConfig)
    # Which channel of this image is which marker. Markers it does not name are
    # not measured: their probability comes back NaN and their call False, and
    # every report says so rather than counting them as negative. DAPI must be
    # named -- there are no cells without it. See neural_imgs.channels.
    layout: ChannelLayout = DEFAULT_LAYOUT
    # CPU threads for per-patch cell extraction; None = this process's core
    # allocation. CellPose and the classifiers stay on one thread regardless --
    # one device, one model.
    n_workers: int | None = None
    # Retain the five normalised per-channel crops on every cell (~1.3 MB each).
    # Only worth it for a caller that renders them without a PatchPreprocessor.
    keep_display_crops: bool = False


@dataclass
class CellResult:
    """One cell's calls.

    A marker the run did not measure -- no channel for it in the layout -- comes
    back with a NaN probability/score and a False call. The False is a storage
    detail, not a finding: whether a marker was measured at all is a property of
    the run, carried on :class:`ClassifyImageOutput.layout`, and every report
    consults that before reading these flags.
    """
    cell_id: int
    patch_idx: int
    bb_position: tuple[int, int, int, int]
    opc_prob: float
    opc_pos: bool
    b3tub_prob: float
    b3tub_pos: bool
    rfp_pos: bool
    # None for cells restored from a cache entry written without its mask
    # sidecar (see neural_imgs.inference.result_cache).
    native_mask: np.ndarray | None = None

    # Continuous per-channel marker intensity: the cell's masked mean minus the
    # crop's 95th percentile, x100 (analyze_image's "score(%)"). ``rfp_pos`` is
    # this score gated by RfpConfig, so keeping the score is what
    # makes the RFP call re-thresholdable after the fact -- and what lets the
    # RFP and OPC channels be correlated to test for spectral bleed-through.
    # NaN for cells restored from a cache entry written before these existed.
    # Keyword-only so the positional signature above stays unchanged.
    rfp_score: float = field(default=float("nan"), kw_only=True)
    opc_channel_score: float = field(default=float("nan"), kw_only=True)
    b3tub_channel_score: float = field(default=float("nan"), kw_only=True)

    # Both RFP rules are recorded for every cell, whichever one made the call:
    # ``rfp_pos`` is ``rfp_<method>_pos``, and keeping the other lets a saved
    # table be re-read under the other rule without re-classifying the image.
    # ``rfp_score`` above is the legacy score -- the two are NOT on the same
    # scale (the background estimator differs), so a single gate cannot be
    # compared across them.
    rfp_method: str = field(default="neighbour", kw_only=True)
    rfp_neighbour_score: float = field(default=float("nan"), kw_only=True)
    rfp_neighbour_pos: bool = field(default=False, kw_only=True)
    rfp_legacy_pos: bool = field(default=False, kw_only=True)
    # How many nuclei share this cell's crop. The crowding that the neighbour
    # rule exists to correct for, kept so a suspicious call can be checked
    # against it.
    n_cells_in_region: int = field(default=-1, kw_only=True)


@dataclass
class ClassifyImageOutput:
    image_shape: tuple[int, ...]
    cells: list[CellResult]
    #: The channel mapping the pixels were read under, reconciled with the
    #: image's real channel count. Stored with the results (and in the cache
    #: entry) because a table of booleans cannot say what was measured.
    layout: ChannelLayout = DEFAULT_LAYOUT
    #: Marker roles actually called, when that is *narrower* than the layout --
    #: the one case being a model directory that shipped no checkpoint for a
    #: marker whose channel is present. ``None`` means "every marker the layout
    #: names", which is the normal run.
    #:
    #: Kept apart from ``layout`` because the two answer different questions: an
    #: unclassifiable marker's channel is still read, still illumination-
    #: corrected and still scored for intensity, so it must stay in the layout
    #: -- it just cannot carry a positive/negative call.
    measured: "tuple[str, ...] | None" = None

    @property
    def markers(self) -> tuple[str, ...]:
        """Marker roles whose calls on these cells mean something."""
        return self.layout.markers if self.measured is None else self.measured

    @property
    def axes(self) -> MarkerAxes:
        """How to enumerate, plot and quantify this result's cases."""
        return MarkerAxes(markers=self.markers)


@dataclass
class LoadedModels:
    """Everything a run needs that costs minutes to load, loaded once.

    Deliberately **layout-free**: which plane of an image a classifier reads is
    a property of the run, not of the weights, so the same loaded models serve
    images with different channel layouts and nothing has to be reloaded when
    the layout changes. ``classifiers`` holds whichever markers the model
    directory actually provides -- see :func:`load_models`.
    """
    device: str
    cellpose_model: object
    classifiers: dict[str, CellClassifier]
    thresholds: dict[str, float]

    @property
    def available_markers(self) -> tuple[str, ...]:
        """Markers with both a loaded classifier and a decision threshold."""
        return tuple(
            m for m in CLASSIFIER_MARKERS
            if m in self.classifiers and m in self.thresholds
        )


def load_models(
    model_dir: str | Path, device: str | None = None,
    on_progress: Callable[[str, int | None, int | None], None] | None = None,
) -> LoadedModels:
    """Load the frozen production classifiers + decision thresholds.

    ``model_dir`` should contain one subdirectory per marker (named via
    :func:`prod_exp_name`) holding ``best_model.pt``, plus a
    ``thresholds.json`` with ``{"OPC": <float>, "B3Tub": <float>}`` (produced
    by ``scripts/select_classifier_thresholds.py``). Thresholds are loaded
    once here, never recomputed at inference time -- a deployed install has
    no access to the held-out validation CSVs that selection needs.

    A marker whose checkpoint or threshold is absent is **skipped and
    reported**, not fatal. The published bundle always carries both, but a
    user's own retrained directory may hold one, and a run whose image has no
    channel for the other marker anyway should not be blocked by that -- the
    missing marker is then simply not measured, exactly as if the channel were
    absent. A directory providing neither is an error: there would be nothing
    to classify with.
    """
    report = on_progress or (lambda *_: None)
    model_dir = Path(model_dir)
    device = device or best_device()
    device_label = {
        "cuda": "NVIDIA GPU (CUDA)", "mps": "Apple GPU (MPS)", "cpu": "CPU",
    }.get(device, device)
    report(f"Using device: {device_label}", None, None)
    thresholds = json.loads((model_dir / "thresholds.json").read_text())

    make_deterministic(SEED)
    report("Loading CellPose model...", None, None)
    cellpose_model = get_cellpose_model()

    classifiers: dict[str, CellClassifier] = {}
    for marker in MARKERS:
        ckpt_path = model_dir / prod_exp_name(marker) / "best_model.pt"
        if not ckpt_path.exists() or marker not in thresholds:
            report(
                f"No {marker} classifier in {model_dir} -- "
                f"{marker} will not be measured", None, None)
            continue
        report(f"Loading {marker} classifier...", None, None)
        # The classifier always sees exactly one (soft-masked) channel, so the
        # network's input width does not depend on the run's channel layout.
        cfg = build_raw_cfg(marker)
        model = CellClassifier(
            input_channels=cfg.n_channels, num_classes=2, pretrained=False,
            dropout=0.2, freeze_backbone=False, architecture="resnet18",
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        classifiers[marker] = model

    if not classifiers:
        raise FileNotFoundError(
            f"no usable classifier in {model_dir}: expected "
            f"{'/'.join(prod_exp_name(m) for m in MARKERS)}/best_model.pt and "
            f"the matching entries in thresholds.json")

    return LoadedModels(
        device=device, cellpose_model=cellpose_model,
        classifiers=classifiers, thresholds=thresholds,
    )


def _channel_scores(result: dict) -> dict[str, float]:
    """The three continuous marker scores of one ``analyze_image`` result.

    Plain floats, never numpy scalars: these travel over Appose's JSON wire to
    the Fiji plugin, which hands Java an opaque object handle for a numpy scalar
    instead of a number.
    """
    score = result.get("score(%)", {})
    return {
        "rfp_score": float(score.get("RFP", float("nan"))),
        "opc_channel_score": float(score.get("OPC", float("nan"))),
        "b3tub_channel_score": float(score.get("B3Tub", float("nan"))),
    }


def run_markers(layout: ChannelLayout, models: LoadedModels) -> tuple[str, ...]:
    """Classifier-backed markers this run can actually call.

    The intersection of two independent facts: the image has a channel for the
    marker (``layout``) and the model directory provided its checkpoint
    (``models``). Either one missing means the marker is not measured.
    """
    return tuple(
        m for m in layout.classifier_markers if m in models.available_markers)


def measured_markers(layout: ChannelLayout, models: LoadedModels) -> tuple[str, ...]:
    """Every marker this run can call, classifier-backed or not.

    :func:`run_markers` plus RFP when the layout has an RFP channel -- RFP needs
    no checkpoint, since it is an intensity rule. This is the set every report
    is built over: what the UpSet plot has sets for, what the quantification can
    compute a ratio from, and what the plugin renders columns for.
    """
    backed = run_markers(layout, models)
    return tuple(
        m for m in MARKER_ROLES
        if m in backed or (m == "RFP" and layout.has("RFP"))
    )


@torch.no_grad()
def _classify_cells(
    cells: list[dict], models: LoadedModels, layout: ChannelLayout,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Score every cell with each marker's own preprocessing (raw_cell_to_tensor).

    Only the markers :func:`run_markers` allows are scored; the rest are absent
    from the returned dict, which is how the caller tells "not measured" from
    "measured 0.0".
    """
    probs: dict[str, np.ndarray] = {}
    for marker in run_markers(layout, models):
        model = models.classifiers[marker]
        # Built per run, not per model load: `channels` points at the marker's
        # plane in *this* image, which the layout decides.
        cfg = build_raw_cfg(marker, layout)
        model.eval()
        out: list[float] = []
        for i in range(0, len(cells), batch_size):
            batch = cells[i:i + batch_size]
            xb = torch.stack([
                raw_cell_to_tensor(c["raw_img"], c["native_mask"], cfg) for c in batch
            ]).to(models.device)
            out.extend(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy().tolist())
        probs[marker] = np.array(out)
    return probs


def classify_image(
    input: ClassifyImageInput, models: LoadedModels, patched: PatchedImage | None = None,
    on_progress: Callable[[str, int | None, int | None], None] | None = None,
    preprocessor: PatchPreprocessor | None = None,
) -> ClassifyImageOutput:
    """Run the full fixed-model pipeline on one CZI image.

    ``patched`` lets a caller that already loaded the image (e.g. the Fiji
    plugin worker, which also needs it afterwards for on-demand single-cell
    crops via :func:`crop_cell`) pass it in instead of re-reading the CZI.
    ``preprocessor`` does the same for the BaSiC fit -- pass one in (fitted on
    the same ``patched``) to reuse it afterwards for display, instead of paying
    for a second fit.

    ``on_progress``, if given, is called as ``on_progress(message, current,
    maximum)`` at each pipeline stage -- ``current``/``maximum`` are ``None``
    for non-countable stages. Lets a caller (e.g. the Fiji plugin worker)
    surface step-by-step status without this module depending on Appose.

    ``input.layout`` decides which markers are measured. A marker with no
    channel (or no shipped checkpoint) is skipped: its cells come back with a
    NaN probability and a False call, and :attr:`ClassifyImageOutput.layout`
    records that it was never measured so no report counts it as negative. At
    the extreme, a DAPI-only layout runs detection and nothing else -- which is
    a cell count, and a legitimate thing to ask for.
    """
    report = on_progress or (lambda *_: None)
    make_deterministic(SEED)

    if patched is None:
        report("Reading CZI image...", None, None)
        patched = load_patches(input.image_path, input.patch_grid, input.layout)
    # The image's own reconciled layout wins: load_patches padded the declared
    # one out to the file's real channel count, and a caller that passed
    # `patched` in already went through that same check.
    layout = patched.layout

    patches = patched.patches

    if preprocessor is None:
        preprocessor = PatchPreprocessor.fit(patched, MAX_ITERATIONS, on_progress)

    report(layout.describe(), None, None)
    extractor = FastCellExtractor(CellExtractionConfig(
        rfp=input.rfp, layout=layout,
        keep_display_crops=input.keep_display_crops,
    ))
    detection = ParallelPatchDetector(
        models.cellpose_model, extractor,
        flow_threshold=input.flow_threshold,
        cellprob_threshold=input.cellprob_threshold,
        n_workers=input.n_workers,
    ).run(patches, preprocessor.basics, on_progress=report)
    all_cells = detection.cells
    report(f"Detected {detection.timing_line()}", None, None)

    classifiable = run_markers(layout, models)
    if classifiable:
        report(f"Classifying {len(all_cells)} detected cells "
               f"({', '.join(classifiable)})...", None, None)
    else:
        report(f"No classifier channel in this run -- {len(all_cells)} cells "
               f"detected and counted, none classified", None, None)
    probs = _classify_cells(all_cells, models, layout)

    def marker_calls(marker: str) -> tuple[np.ndarray, np.ndarray]:
        """``(probabilities, calls)`` for one marker; NaN/False when unmeasured."""
        if marker not in probs:
            return (np.full(len(all_cells), np.nan),
                    np.zeros(len(all_cells), dtype=bool))
        values = probs[marker]
        return values, values >= models.thresholds[marker]

    opc_probs, opc_preds = marker_calls("OPC")
    b3tub_probs, b3tub_preds = marker_calls("B3Tub")
    rfp_measured = layout.has("RFP")

    results = []
    for i, cell in enumerate(all_cells):
        scores = _channel_scores(cell["result"])
        # Two rules, one call. The legacy rule's own positivity flag is not used
        # here: analyze_image applies eps to the *unrounded* mean while the gate
        # below is on score(%), and deriving both from one comparison is what
        # keeps them from disagreeing at the boundary.
        #
        # With no RFP channel both scores are NaN, so both comparisons are False
        # already; the explicit guard is so a reader does not have to rely on
        # NaN comparison semantics to see that nothing was called.
        legacy_pos = rfp_measured and scores["rfp_score"] > input.rfp.legacy_threshold
        neighbour_pos = rfp_measured and bool(cell.get("rfp_neighbour_pos", False))
        results.append(CellResult(
            cell_id=i,
            patch_idx=int(cell["patch_idx"]),
            # Plain ints, not numpy int64s: these travel over Appose's JSON wire
            # to the Fiji plugin, which has no encoding for a numpy scalar and
            # would hand Java an opaque object handle instead of a number.
            bb_position=tuple(int(v) for v in cell["bb_position"]),
            opc_prob=float(opc_probs[i]), opc_pos=bool(opc_preds[i]),
            b3tub_prob=float(b3tub_probs[i]), b3tub_pos=bool(b3tub_preds[i]),
            rfp_pos=neighbour_pos if input.rfp.method == "neighbour" else legacy_pos,
            native_mask=cell["native_mask"],
            rfp_method=input.rfp.method,
            rfp_neighbour_score=float(cell.get("rfp_neighbour_score", float("nan"))),
            rfp_neighbour_pos=neighbour_pos,
            rfp_legacy_pos=legacy_pos,
            n_cells_in_region=int(cell.get("n_cells_in_region", -1)),
            **scores,
        ))
    return ClassifyImageOutput(
        image_shape=patched.image_shape, cells=results, layout=layout,
        measured=measured_markers(layout, models))
