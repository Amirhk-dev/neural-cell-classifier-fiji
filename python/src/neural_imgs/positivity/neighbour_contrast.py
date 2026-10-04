"""The legacy RFP rule with a background a neighbouring cell cannot hijack.

``analyze_image`` scores a cell as ``mean(dilated nucleus) - p95(crop)``. The
foreground half is fine. The background half is not: the p95 is the brightest
few percent of *everything* in the crop, so one bright neighbour sets the
reference and a genuinely transduced cell scores negative. On the OPC control
112 cells sit below -10 for that reason, and the score's correlation with
``n_cells_in_region`` is -0.47.

Replacing the foreground was tried first and lost: a Gaussian-feathered mask
wide enough to reach a cell's own cytoplasm also reaches its neighbours', which
reversed the crowding bias to +0.36 and diluted the transduced population (see
:mod:`neural_imgs.positivity.soft_mask`). This scorer therefore keeps the
foreground **exactly** as the legacy rule has it and changes only the pool the
percentile is taken over: pixels near another cell's nucleus are dropped.

Two details make it comparable to the rule it replaces:

* **It runs in native pixels.** The legacy rule resizes each crop so its short
  side is 512, centre-crops, and zeroes a 100 px border -- geometry that only
  existed to make the 512 view. Here the same field of view is the central
  ``window_fraction`` of the native crop's short side, which is what that border
  actually keeps.
* **The percentile is 86.5, not 95.** ``analyze_image`` zeroes that border and
  then takes the percentile over the *whole* array, zeros included; with 63% of
  the 512 view zeroed its "p95" is really the ~p86.5 of the central window.
  Using 95 here would be a different, much stricter rule.

With ``exclude_radius = 0, exclude_foreground = False`` the result reproduces
``analyze_image`` at spearman **+0.986** (OPC control and experiment), which is what makes the
comparison below meaningful. Measured effect of turning the exclusion on
(r = 60 px, matched positive counts, control + experiment):

=========================  ==========  ==========
.                          r = 0       r = 60
=========================  ==========  ==========
spearman(score, crowding)  -0.474      **-0.004**
control cells recovered    0 / 112     **34 / 112**
experiment conversion      79.6%       77.8%
control conversion         0.82%       1.65%
=========================  ==========  ==========

Conversion (B3-Tub+ among RFP+) is the externally anchored check -- the
biologists expect ~0% in the control-vector control and 67-82% in an experiment
-- and it stays in band. ``r`` sits on a plateau: 50-80 px all give
|crowding| < 0.05 and recover 31-35 cells, so 60 is the middle rather than an
edge.

**It is a partial fix.** 34 of 112, not 112 of 112: the rest are contaminated by
neighbour *cytoplasm*, which CellPose never segmented, so no nucleus-radius
exclusion can reach it.
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

#: Central fraction of a crop's short side that the legacy rule's 100 px border
#: on a 512 view actually keeps.
LEGACY_WINDOW_FRACTION = (512 - 2 * 100) / 512

#: Percentile of that central window equivalent to the legacy rule's "p95 over
#: the border-zeroed 512 array" (the zeros are 1 - LEGACY_WINDOW_FRACTION**2 of
#: the pixels and rank below everything else).
LEGACY_EQUIVALENT_PERCENTILE = 86.5


@dataclass
class NeighbourContrastConfig:
    """Knobs of the rule. Defaults reproduce the legacy operating point.

    Parameters
    ----------
    dilation_radius:
        Native px the nucleus grows by before its mean is taken. 2 native px is
        the legacy rule's 3 px measured on the 512 view.
    window_fraction / percentile:
        The background window and the percentile taken over it; the defaults are
        the legacy equivalents derived above. Change them together or the
        operating point moves.
    exclude_radius:
        Native px around ANY other cell's nucleus whose pixels are dropped from
        the background pool. 0 reproduces the legacy rule; 60 is the plateau
        centre. It stands in for the neighbour's whole cell body, since only
        nuclei are segmented.
    exclude_foreground:
        Drop the cell's own dilated nucleus from its background pool. The legacy
        rule does NOT do this -- its percentile runs over every pixel, the cell
        included -- so the configuration that exactly reproduces it is
        ``exclude_radius=0, exclude_foreground=False``. On the OPC images the
        difference is negligible (the nucleus is a small, not-especially-bright
        fraction of the crop), but on a cell whose nucleus IS the brightest
        thing present it matters, so it is a knob rather than a hard-coded
        choice.
    min_background_px:
        If the exclusion leaves fewer pixels than this, fall back to the
        unexcluded window rather than scoring the cell from a handful of pixels.
    threshold:
        Cut on the score, in the same units as the legacy ``score(%)``.
    """

    dilation_radius: int = 2
    window_fraction: float = LEGACY_WINDOW_FRACTION
    percentile: float = LEGACY_EQUIVALENT_PERCENTILE
    exclude_radius: int = 60
    exclude_foreground: bool = True
    min_background_px: int = 200
    threshold: float = 5.0


@dataclass
class NeighbourContrastComponents:
    """Foreground and background level per cell, plus how much was excluded.

    ``excluded_frac`` is the diagnostic that says whether the exclusion did
    anything for a given cell: 0 means it had no neighbours and its score is the
    legacy score.
    """

    foreground: dict[str, np.ndarray]
    background: dict[str, np.ndarray]
    excluded_frac: np.ndarray
    n_background_px: np.ndarray


class NeighbourContrastHeuristic(BaseMarkerScorer):
    """Legacy foreground, neighbour-free background.

    Each cell dict must carry, all at native resolution and the same shape:

    * ``native_channels`` -- one ``(H, W)`` array per channel, globally
      normalised (:func:`~neural_imgs.positivity.soft_mask.cells_with_native_views`
      builds this from the ``*_crop`` keys without copying pixels);
    * ``native_mask`` -- this cell's DAPI mask;
    * ``neighbour_mask`` -- ``True`` where any OTHER cell's nucleus is. An
      all-``False`` mask is legal and makes the cell score exactly as the legacy
      rule scores it.
    """

    def __init__(self, config: NeighbourContrastConfig) -> None:
        self._config = config
        self._components: NeighbourContrastComponents | None = None

    @property
    def components(self) -> NeighbourContrastComponents:
        if self._components is None:
            raise RuntimeError("call score() first")
        return self._components

    def _window(self, shape: tuple[int, int]) -> np.ndarray:
        """The central part of the crop the legacy border-zeroing keeps."""
        h, w = shape
        keep = int(round(self._config.window_fraction * min(h, w)))
        out = np.zeros((h, w), dtype=bool)
        y0, x0 = (h - keep) // 2, (w - keep) // 2
        out[y0:y0 + keep, x0:x0 + keep] = True
        return out

    def _regions(self, mask: np.ndarray, neighbours: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, float]:
        """Foreground, background pool, and the fraction the exclusion removed."""
        from scipy.ndimage import binary_dilation, distance_transform_edt
        from skimage.morphology import disk

        cfg = self._config
        foreground = binary_dilation(mask > 0, disk(cfg.dilation_radius))
        window = self._window(mask.shape)
        if cfg.exclude_foreground:
            window = window & ~foreground
        background = window
        excluded = 0.0
        if cfg.exclude_radius and neighbours.any():
            far = distance_transform_edt(~neighbours) > cfg.exclude_radius
            trimmed = window & far
            excluded = 1.0 - trimmed.sum() / max(1, window.sum())
            # A cell ringed by neighbours can lose its whole window; scoring it
            # off a handful of pixels is worse than scoring it the legacy way.
            if trimmed.sum() >= cfg.min_background_px:
                background = trimmed
            else:
                excluded = float("nan")
        return foreground, background, excluded

    def score(self, input: MarkerScoringInput) -> MarkerScoringOutput:
        cfg = self._config
        channels = list(input.target_channels)
        names = {c: input.channel_names[c] for c in channels}
        n = len(input.cells)

        raw = {c: np.full(n, np.nan) for c in channels}
        fg_level = {names[c]: np.full(n, np.nan) for c in channels}
        bg_level = {names[c]: np.full(n, np.nan) for c in channels}
        excluded = np.full(n, np.nan)
        n_bg = np.zeros(n, dtype=int)
        labels = np.empty(n, dtype=int)

        for row, cell in enumerate(input.cells):
            labels[row] = int(cell["label"])
            mask = cell["native_mask"]
            if mask.sum() == 0:
                continue
            neighbours = cell.get("neighbour_mask")
            if neighbours is None:
                neighbours = np.zeros(mask.shape, dtype=bool)
            fg, bg, excl = self._regions(mask, np.asarray(neighbours, dtype=bool))
            if not fg.any() or not bg.any():
                continue
            excluded[row] = excl
            n_bg[row] = int(bg.sum())
            for c in channels:
                plane = cell["native_channels"][c]
                f = float(plane[fg].mean())
                b = float(np.percentile(plane[bg], cfg.percentile))
                fg_level[names[c]][row] = f
                bg_level[names[c]][row] = b
                raw[c][row] = (f - b) * 100.0

        self._components = NeighbourContrastComponents(
            foreground=fg_level, background=bg_level,
            excluded_frac=excluded, n_background_px=n_bg)

        scores = {}
        for c in channels:
            values = raw[c]
            score = MarkerScore(
                channel_name=names[c], channel_index=c,
                method="neighbour", score_name="contrast(%)",
                scores=values,
                positive=np.nan_to_num(values, nan=-np.inf) > cfg.threshold,
                threshold=float(cfg.threshold),
            )
            scores[score.key] = score
        return MarkerScoringOutput(labels=labels, scores=scores)


def neighbour_mask_from_labels(label_crop: np.ndarray, label: int) -> np.ndarray:
    """``True`` where a CellPose label other than ``label`` (and not background) is.

    Both extraction paths already slice the patch's label map to the cell's
    bounding box, so this is the whole cost of supporting the rule.
    """
    return (label_crop != label) & (label_crop != 0)
