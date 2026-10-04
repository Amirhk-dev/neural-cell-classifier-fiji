"""The OPC-project intensity heuristic, applied to any marker channel.

This is the rule that decided ``RFP+`` in the OPC pipeline
(:func:`neural_imgs.processing.processing.analyze_image`, gated at
``CONFIDENCE_THRESHOLD = 10`` in
:mod:`neural_imgs.inference.fixed_model_pipeline`): a cell is positive in a
channel when the mean intensity inside its (slightly dilated) nucleus mask
stands out above the local background of its own crop.

    positive  <=>  mean(channel over dilated nucleus)  >  p95(channel over crop) + 0.1

Everything is measured on the **normalised [0, 1]** image, and on the same
512x512 view the OPC pipeline used: the nucleus bounding box grown by ``pad``,
resized so its short side is 512, centre-cropped to 512x512, then the outer
``border = 100`` px zeroed. That border matters -- it is 63% of the pixels, so
the "p95" is in practice the ~p87 of the central window -- which is why this
class rebuilds the 512 view rather than reusing native crops.

Porting the rule to an experiment with a different pixel size is therefore only
a matter of keeping the *geometry* equivalent, not the pixel counts: the OPC
CZIs have ~57 px nuclei and used ``pad = 100``, i.e. 1.75 nucleus diameters of
context on each side. :func:`pad_for_nucleus_diameter` reproduces that ratio at
any resolution, after which the 512-resize makes every downstream constant
(border, percentile, dilation radius) land on the same physical scale it did in
the OPC project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_imgs.processing.processing import (
    analyze_image,
    extract_cells_512_nodistort,
    zero_out_image_border,
)
from neural_imgs.positivity.base import (
    BaseMarkerScorer,
    MarkerScore,
    MarkerScoringInput,
    MarkerScoringOutput,
)

#: Context kept around the nucleus in the OPC pipeline, in nucleus diameters
#: (``pad = 100`` px against a ~57 px median nucleus diameter).
OPC_PAD_IN_NUCLEUS_DIAMETERS = 100.0 / 57.0


def pad_for_nucleus_diameter(diameter_px: float,
                             pad_in_diameters: float = OPC_PAD_IN_NUCLEUS_DIAMETERS) -> int:
    """Crop padding, in native px, that keeps the OPC-pipeline field of view."""
    return int(round(diameter_px * pad_in_diameters))


@dataclass
class LocalContrastConfig:
    """Knobs of the heuristic. The defaults are the OPC production values."""

    pad: int                              # native px of context around the nucleus bbox
    target: int = 512                     # side of the resized crop the rule runs on
    border: int = 100                     # outer ring of the resized crop zeroed before p95
    background_percentile: float = 95.0   # local background estimate
    dilation_radius: int = 3              # px (on the `target` grid) the nucleus grows by
    eps: float = 0.1                      # margin required over the background
    confidence_threshold: float = 10.0    # gate on (mean - background) * 100
    min_area: int = 100                   # must match the caller's cell extraction
    boundary_margin: int = 20             # must match the caller's cell extraction


class LocalContrastHeuristic(BaseMarkerScorer):
    """Score cells by how far each channel rises above its own local background.

    Constructed with the normalised image and label map the cells came from,
    because the rule is defined on a *resized, border-zeroed* view of each cell
    that has to be cut from the full image rather than derived from a native
    crop.
    """

    def __init__(self, image: np.ndarray, masks: np.ndarray,
                 config: LocalContrastConfig) -> None:
        self._image = image
        self._masks = masks
        self._config = config
        self._crops: list[dict] | None = None

    @property
    def crops(self) -> list[dict]:
        """The 512x512 border-zeroed crops the rule was evaluated on (for montages)."""
        if self._crops is None:
            raise RuntimeError("call score() first")
        return self._crops

    def _build_crops(self) -> list[dict]:
        cfg = self._config
        crops = extract_cells_512_nodistort(
            self._image, self._masks, pad=cfg.pad, min_area=cfg.min_area,
            boundary_margin=cfg.boundary_margin, target=cfg.target,
        )
        for crop in crops:
            crop["img"] = zero_out_image_border(crop["img"], border=cfg.border)
        return crops

    def score(self, input: MarkerScoringInput) -> MarkerScoringOutput:
        cfg = self._config
        crops = self._build_crops()
        by_label = {int(c["label"]): c for c in crops}

        wanted = [int(c["label"]) for c in input.cells]
        missing = [lab for lab in wanted if lab not in by_label]
        if missing:
            raise ValueError(
                f"{len(missing)} cells have no 512 crop (labels {missing[:5]}...) -- "
                "min_area / boundary_margin must match the caller's extraction"
            )
        self._crops = [by_label[lab] for lab in wanted]

        n = len(wanted)
        channels = list(input.target_channels)
        raw = {c: np.full(n, np.nan, dtype=np.float64) for c in channels}

        for row, crop in enumerate(self._crops):
            result = analyze_image(
                crop["img"], crop["mask"], list(input.channel_names), channels,
                dilation_radius=cfg.dilation_radius,
                percent=cfg.background_percentile, eps=cfg.eps,
            )
            for c in channels:
                name = input.channel_names[c]
                mean, background = result["masked_means"][name], result["percentile"][name]
                if not np.isnan(mean):
                    # The same number analyze_image reports as "confidence(%)" --
                    # kept for every cell, not only the positive ones, so the
                    # decision threshold can be seen against a distribution.
                    raw[c][row] = (mean - background) * 100.0

        scores = {}
        for c in channels:
            values = raw[c]
            positive = np.nan_to_num(values, nan=-np.inf) > max(
                cfg.confidence_threshold, cfg.eps * 100.0)
            score = MarkerScore(
                channel_name=input.channel_names[c], channel_index=c,
                method="heuristic", score_name="confidence(%)",
                scores=values, positive=positive,
                threshold=max(cfg.confidence_threshold, cfg.eps * 100.0),
            )
            scores[score.key] = score

        return MarkerScoringOutput(labels=np.array(wanted, dtype=int), scores=scores)
