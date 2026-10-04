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

CHANNEL_NAMES = ["DAPI", "OPC", "RFP", "B3Tub", "BF"]
TARGET_CHANNELS = [1, 2, 3]  # OPC, RFP, B3Tub -- for the RFP heuristic (analyze_image)
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


def build_raw_cfg(marker: str) -> RawCropConfig:
    """Inference preprocessing for a marker (single soft-masked channel; augment=False)."""
    c = PROD_CONFIG[marker]
    return RawCropConfig(
        crop_size=c["crop_size"], channels=[CHANNEL_NAMES.index(c["channel"])],
        include_mask=False, apply_soft_mask=True, mask_sigma=c["mask_sigma"],
        augment=False,
    )


def prod_exp_name(marker: str) -> str:
    """Checkpoint dir name; MUST match scripts/train_raw_classifier_final.py."""
    c = PROD_CONFIG[marker]
    idx = CHANNEL_NAMES.index(c["channel"])
    return (f"raw_{marker}_resnet18_crop{c['crop_size']}"
            f"_soft{c['mask_sigma']:g}_ch{idx}_prod")


@dataclass
class PatchedImage:
    """A CZI, squeezed and split into an NxN patch grid -- the shared starting
    point for both full-image classification and on-demand single-cell crop
    lookup (see :func:`crop_cell`), so both read the exact same pixels."""
    patches: np.ndarray  # (N, 5, H_patch, W_patch), pre-illumination-correction
    image_shape: tuple[int, int, int]  # (5, H, W) of the full squeezed image


def load_patches(image_path: str | Path, patch_grid: int) -> PatchedImage:
    img = squeeze_img(read_czi(image_path))
    patches = patch_image(img, patch_grid)
    return PatchedImage(patches=patches, image_shape=tuple(img.shape))


def crop_cell(
    patched: PatchedImage, patch_idx: int, bb_position: tuple[int, int, int, int]
) -> np.ndarray:
    """Raw (uncorrected) 5-channel crop for one cell -- exactly the pixels
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

    def normalized_patch(self, patch_idx: int) -> np.ndarray:
        """One patch as (5, H, W) in [0, 1] -- corrected and percentile-normalised.

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
    # CPU threads for per-patch cell extraction; None = this process's core
    # allocation. CellPose and the classifiers stay on one thread regardless --
    # one device, one model.
    n_workers: int | None = None
    # Retain the five normalised per-channel crops on every cell (~1.3 MB each).
    # Only worth it for a caller that renders them without a PatchPreprocessor.
    keep_display_crops: bool = False


@dataclass
class CellResult:
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


@dataclass
class LoadedModels:
    device: str
    cellpose_model: object
    classifiers: dict[str, CellClassifier]
    raw_cfgs: dict[str, RawCropConfig]
    thresholds: dict[str, float]


def load_models(
    model_dir: str | Path, device: str | None = None,
    on_progress: Callable[[str, int | None, int | None], None] | None = None,
) -> LoadedModels:
    """Load the frozen production classifiers + decision thresholds.

    ``model_dir`` must contain one subdirectory per marker (named via
    :func:`prod_exp_name`) holding ``best_model.pt``, plus a
    ``thresholds.json`` with ``{"OPC": <float>, "B3Tub": <float>}`` (produced
    by ``scripts/select_classifier_thresholds.py``). Thresholds are loaded
    once here, never recomputed at inference time -- a deployed install has
    no access to the held-out validation CSVs that selection needs.
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
    raw_cfgs: dict[str, RawCropConfig] = {}
    for marker in MARKERS:
        report(f"Loading {marker} classifier...", None, None)
        cfg = build_raw_cfg(marker)
        model = CellClassifier(
            input_channels=cfg.n_channels, num_classes=2, pretrained=False,
            dropout=0.2, freeze_backbone=False, architecture="resnet18",
        )
        ckpt_path = model_dir / prod_exp_name(marker) / "best_model.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        classifiers[marker] = model
        raw_cfgs[marker] = cfg

    return LoadedModels(
        device=device, cellpose_model=cellpose_model,
        classifiers=classifiers, raw_cfgs=raw_cfgs, thresholds=thresholds,
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


@torch.no_grad()
def _classify_cells(
    cells: list[dict], models: LoadedModels, batch_size: int = 64
) -> dict[str, np.ndarray]:
    """Score every cell with each marker's own preprocessing (raw_cell_to_tensor)."""
    probs: dict[str, np.ndarray] = {}
    for marker in MARKERS:
        model, cfg = models.classifiers[marker], models.raw_cfgs[marker]
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
    """
    report = on_progress or (lambda *_: None)
    make_deterministic(SEED)

    if patched is None:
        report("Reading CZI image...", None, None)
        patched = load_patches(input.image_path, input.patch_grid)
    patches = patched.patches

    if preprocessor is None:
        preprocessor = PatchPreprocessor.fit(patched, MAX_ITERATIONS, on_progress)

    extractor = FastCellExtractor(
        CellExtractionConfig(rfp=input.rfp, keep_display_crops=input.keep_display_crops),
        CHANNEL_NAMES, TARGET_CHANNELS,
    )
    detection = ParallelPatchDetector(
        models.cellpose_model, extractor,
        flow_threshold=input.flow_threshold,
        cellprob_threshold=input.cellprob_threshold,
        n_workers=input.n_workers,
    ).run(patches, preprocessor.basics, on_progress=report)
    all_cells = detection.cells
    report(f"Detected {detection.timing_line()}", None, None)

    report(f"Classifying {len(all_cells)} detected cells...", None, None)
    probs = _classify_cells(all_cells, models)
    opc_probs, b3tub_probs = probs["OPC"], probs["B3Tub"]
    opc_preds = opc_probs >= models.thresholds["OPC"]
    b3tub_preds = b3tub_probs >= models.thresholds["B3Tub"]

    results = []
    for i, cell in enumerate(all_cells):
        scores = _channel_scores(cell["result"])
        # Two rules, one call. The legacy rule's own positivity flag is not used
        # here: analyze_image applies eps to the *unrounded* mean while the gate
        # below is on score(%), and deriving both from one comparison is what
        # keeps them from disagreeing at the boundary.
        legacy_pos = scores["rfp_score"] > input.rfp.legacy_threshold
        neighbour_pos = bool(cell.get("rfp_neighbour_pos", False))
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
    return ClassifyImageOutput(image_shape=patched.image_shape, cells=results)
