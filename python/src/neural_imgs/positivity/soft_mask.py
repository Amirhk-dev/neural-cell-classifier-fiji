"""Soft-mask intensity heuristic -- the classifiers' cell isolation, applied to a
raw channel.

Why this exists
---------------
:class:`~neural_imgs.positivity.heuristic.LocalContrastHeuristic` (the original
OPC rule) calls a cell positive when

    mean(channel over nucleus dilated by 3 px)  >  p95(channel over the crop) + eps

Both halves of that comparison fail for a *cytoplasmic* marker such as the RFP
transduction reporter in a crowded field:

* **The foreground is nuclear.** A 3 px dilation of the DAPI mask measures the
  nucleus, but RFP fills the cytoplasm -- often with the nucleus as its dimmest
  part. A brightly transduced cell can therefore have a low masked mean.
* **The background is the crop's p95, and neighbours own it.** The reference is
  the 95th percentile of every pixel in the crop, the neighbouring cells
  included. One bright neighbour lifts p95 above the target cell's own
  intensity and the score goes to zero or negative. This is measured, not
  hypothetical: spearman(score, n_cells_in_region) runs -0.06..-0.50 among
  positives, and it is why the mirror-FDR noise test is only valid on sparse
  crops.

Together they produce exactly the reported failure -- a cell with obvious RFP
signal called negative.

What this scorer does instead
-----------------------------
It borrows the isolation step the OPC/B3-Tub ResNet-18s already use
(:func:`neural_imgs.training.raw_dataset.soft_mask_weights`): a Gaussian-
feathered version of *this* cell's DAPI mask, full weight on the nucleus and a
smooth decay outward, so the cell's own cytoplasm/neurites survive while
neighbours are attenuated toward zero. The score becomes

    score = (weighted mean of the channel over w >= foreground_weight
             - percentile_bg(channel over w <= background_weight)) * 100

i.e. *this cell including its cytoplasm* against *the far field with this cell
and its halo excluded*. Cytoplasmic signal now counts, and the background is a
low percentile (median by default) of the periphery rather than a p95 that a
neighbour can hijack.

What it deliberately does NOT borrow
------------------------------------
The classifier preprocessing also applies a **per-crop percentile contrast
stretch** (``RawCropConfig.stretch_percentiles``). That is right for a CNN,
which reads shape and texture, and fatal for an intensity heuristic: stretching
each crop to its own [0, 1] range rescales a dim negative cell to look exactly
like a bright positive one, so every cell would clear any threshold. This
scorer therefore runs on the **globally normalised** pixels (BaSiC illumination
correction + per-patch percentile normalisation) that the original heuristic
also used, which keeps intensities comparable across cells of an image.

Because the background estimator changed (a low percentile of the periphery
instead of the crop's p95), scores from this scorer are on a **different scale**
from ``analyze_image``'s ``score(%)`` and its threshold has to be recalibrated
rather than carried over. :func:`mirror_fdr_table` is the same-image noise test
to calibrate with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_imgs.positivity.base import (
    BaseMarkerScorer,
    MarkerScore,
    MarkerScoringInput,
    MarkerScoringOutput,
)


@dataclass
class SoftMaskContrastConfig:
    """Knobs of the soft-mask heuristic.

    Parameters
    ----------
    crop_size:
        Side of the native-pixel window cut around the nucleus centroid. The
        default matches ``PROD_CONFIG``'s classifier crop, so the heuristic and
        the classifiers see the same field of view.
    mask_sigma:
        Gaussian falloff of the soft mask, in native px -- how far out of the
        nucleus the cell's own signal is still counted. 16 is the B3-Tub
        production value (a cytoplasmic/neurite marker, like RFP).
    foreground_weight:
        Pixels with soft weight >= this are "this cell". The foreground mean is
        weighted by the soft weights inside that region, so the nucleus counts
        more than the outer ring.
    background_weight:
        Pixels with soft weight <= this are the local background. Everything
        between the two weights is a transition band and is used by neither.
    background_percentile:
        Percentile of the background region taken as the background level.
        50 (median) is deliberately low: it estimates the *field*, not the
        brightest thing in it, which is what stops a neighbouring cell from
        setting the reference.
    threshold:
        Cut on ``(foreground - background) * 100``. Calibrate it per experiment
        -- see :func:`mirror_fdr_table`; it is NOT comparable to the
        ``analyze_image`` gate.
    min_background_px:
        If the low-weight region is smaller than this (a big sigma in a small
        crop), fall back to the ``min_background_px`` lowest-weight valid pixels
        so the score stays defined instead of going NaN.
    """

    crop_size: int = 256
    mask_sigma: float = 32.0
    foreground_weight: float = 0.10
    background_weight: float = 0.02
    background_percentile: float = 50.0
    threshold: float = 5.0
    min_background_px: int = 500


@dataclass
class SoftMaskComponents:
    """The two halves of every score, kept for calibration and montages.

    ``(foreground - background) * 100`` is the score itself; having the parts
    separately is what lets a "why was this cell negative?" question be answered
    (dim cell vs. bright surround) instead of guessed.
    """

    foreground: dict[str, np.ndarray]   # channel name -> (n_cells,) weighted mean
    background: dict[str, np.ndarray]   # channel name -> (n_cells,) local level
    n_foreground_px: np.ndarray         # (n_cells,)
    n_background_px: np.ndarray         # (n_cells,)


class SoftMaskContrastHeuristic(BaseMarkerScorer):
    """Score cells by their soft-masked intensity above their own far field.

    Unlike :class:`LocalContrastHeuristic`, which rebuilds a 512 view from the
    full image, this scorer works directly off the **native crops the cells
    already carry**, so it needs no image/label map: each cell dict must hold

    * ``native_channels``: a sequence of ``(H, W)`` float arrays in [0, 1], one
      per channel, globally normalised (BaSiC + percentile normalisation) at
      native resolution, and
    * ``native_mask``: ``(H, W)`` this cell's DAPI mask at the same resolution.

    :func:`cells_with_native_views` builds the first key from the notebook /
    pipeline cell dicts, which already store those pixels as five separate
    ``*_crop`` arrays. It keeps them as references rather than stacking them
    into one array: on a crowded image that copy is ~1.3 MB per cell, which is
    gigabytes over a whole image and enough to OOM an 8 GB job. Only the
    requested channels are ever windowed.
    """

    def __init__(self, config: SoftMaskContrastConfig) -> None:
        self._config = config
        self._components: SoftMaskComponents | None = None

    @property
    def components(self) -> SoftMaskComponents:
        """Per-cell foreground/background levels behind the scores."""
        if self._components is None:
            raise RuntimeError("call score() first")
        return self._components

    def _window(self, plane: np.ndarray, cy: int, cx: int) -> np.ndarray:
        """One ``(H, W)`` plane, windowed to ``crop_size`` around the centroid."""
        center_window, _ = _numpy_helpers()
        h = w = int(self._config.crop_size)
        return center_window(plane[None].astype(np.float32), cy, cx, h, w)[0]

    def _weights(
        self, mask: np.ndarray
    ) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
        """Centroid, the windowed soft weights, and the valid-pixel mask.

        ``center_window`` zero-pads where the requested window runs off the crop
        (cells near a patch edge). Those zeros are not background measurements,
        so they are tracked and excluded rather than averaged in.
        """
        _, soft_mask_weights = _numpy_helpers()
        cfg = self._config
        cy, cx = _centroid(mask)
        msk = self._window(mask, cy, cx)
        valid = self._window(np.ones(mask.shape, dtype=np.float32), cy, cx) > 0
        weights = soft_mask_weights(msk > 0, cfg.mask_sigma)
        return (cy, cx), weights, valid

    def score(self, input: MarkerScoringInput) -> MarkerScoringOutput:
        cfg = self._config
        channels = list(input.target_channels)
        names = {c: input.channel_names[c] for c in channels}
        n = len(input.cells)

        raw = {c: np.full(n, np.nan, dtype=np.float64) for c in channels}
        fg = {names[c]: np.full(n, np.nan, dtype=np.float64) for c in channels}
        bg = {names[c]: np.full(n, np.nan, dtype=np.float64) for c in channels}
        n_fg = np.zeros(n, dtype=int)
        n_bg = np.zeros(n, dtype=int)
        labels = np.empty(n, dtype=int)

        for row, cell in enumerate(input.cells):
            labels[row] = int(cell["label"])
            planes, mask = cell["native_channels"], cell["native_mask"]
            if mask.sum() == 0:
                continue
            (cy, cx), weights, valid = self._weights(mask)

            fg_sel = valid & (weights >= cfg.foreground_weight)
            bg_sel = valid & (weights <= cfg.background_weight)
            if not fg_sel.any():
                continue
            if bg_sel.sum() < cfg.min_background_px:
                # Too little far field at this sigma/crop: take the lowest-weight
                # valid pixels instead of returning NaN for the whole cell.
                w_valid = np.where(valid, weights, np.inf).ravel()
                take = min(cfg.min_background_px, int(valid.sum()))
                idx = np.argpartition(w_valid, take - 1)[:take]
                bg_sel = np.zeros(weights.size, dtype=bool)
                bg_sel[idx] = True
                bg_sel = bg_sel.reshape(weights.shape)

            n_fg[row] = int(fg_sel.sum())
            n_bg[row] = int(bg_sel.sum())
            w_fg = weights[fg_sel]
            for c in channels:
                ch = self._window(planes[c], cy, cx)
                foreground = float(np.average(ch[fg_sel], weights=w_fg))
                background = float(np.percentile(ch[bg_sel], cfg.background_percentile))
                fg[names[c]][row] = foreground
                bg[names[c]][row] = background
                raw[c][row] = (foreground - background) * 100.0

        self._components = SoftMaskComponents(
            foreground=fg, background=bg, n_foreground_px=n_fg, n_background_px=n_bg)

        scores = {}
        for c in channels:
            values = raw[c]
            score = MarkerScore(
                channel_name=names[c], channel_index=c,
                method="softmask", score_name="soft contrast(%)",
                scores=values,
                positive=np.nan_to_num(values, nan=-np.inf) > cfg.threshold,
                threshold=float(cfg.threshold),
            )
            scores[score.key] = score

        return MarkerScoringOutput(labels=labels, scores=scores)


def _numpy_helpers():
    """``(center_window, soft_mask_weights)`` from the classifier's preprocessing.

    Imported on call, not at module scope: both are pure numpy, but they live in
    a module that pulls in torch, and ``neural_imgs.positivity`` stays importable
    without it (see its ``__init__``). Sharing them with the classifier is the
    point -- the isolation this scorer applies must be the same one the models
    were trained with.
    """
    from neural_imgs.training.raw_dataset import center_window, soft_mask_weights

    return center_window, soft_mask_weights


def _centroid(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return mask.shape[0] // 2, mask.shape[1] // 2
    return int(round(ys.mean())), int(round(xs.mean()))


def cells_with_native_views(
    cells: list[dict],
    crop_keys: tuple[str, ...] = ("dapi_crop", "opc_crop", "rfp_crop",
                                  "b3tub_crop", "bf_crop"),
) -> list[dict]:
    """Add the ``native_channels`` key :class:`SoftMaskContrastHeuristic` expects.

    The notebook and :mod:`neural_imgs.inference.fixed_model_pipeline` keep the
    globally normalised native pixels as one array per channel (they are what the
    montages display). Which key holds which channel is a view-level concern, not
    the scorer's, so the mapping happens here.

    The list holds **references** to the arrays the cells already carry -- no
    pixels are copied. Stacking them into one ``(C, H, W)`` array instead would
    duplicate ~1.3 MB per cell, i.e. several GB on a crowded image.
    """
    for cell in cells:
        if "native_channels" not in cell:
            cell["native_channels"] = [cell[k] for k in crop_keys]
    return cells


def mirror_fdr_table(scores: np.ndarray, thresholds) -> list[dict]:
    """Decoy-tail noise estimate for a signed contrast score, per threshold.

    A real marker can only push a cell *above* its own surround, so the negative
    tail is a free same-image estimate of how often local-background noise alone
    clears ``+t``:

        mirror FDR(t) = N(score < -t) / N(score > t)

    Falls with t  => signal. Flat or rising => no threshold separates signal from
    noise, and tuning the cut only trades one arbitrary number for another.

    **Run this on sparse crops only** (``n_cells_in_region <= 2``). The reference
    level is local, so a neighbouring positive cell deflates a real positive's
    score into the negative tail and the test reads as noise on crowded fields --
    an artifact that has already caused one wrong conclusion on this dataset.

    Returns one row per threshold with the counts kept alongside the ratio, so a
    "0% FDR" resting on three cells is visible as such.
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    rows = []
    for t in thresholds:
        n_pos = int((s > t).sum())
        n_neg = int((s < -t).sum())
        rows.append({
            "threshold": float(t),
            "n_above": n_pos,
            "n_below_mirror": n_neg,
            "mirror_fdr": (n_neg / n_pos) if n_pos else float("nan"),
        })
    return rows
