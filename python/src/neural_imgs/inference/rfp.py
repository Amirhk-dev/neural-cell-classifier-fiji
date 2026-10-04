"""Which RFP rule the deployed pipeline uses, and its operating point.

RFP is the one marker with no trained classifier -- it is called from pixel
intensity -- and the project has three rules for it. This module fixes which
one production runs and at what gate, so the Fiji plugin, any script and
``notebooks/single_cell_classification_fixed_model.ipynb`` cannot drift apart.

The default is the **neighbour-excluded** rule at ``exclude_radius = 60``,
``threshold = 16.5``, matching the notebook's ``RFP_METHOD = "neighbour"``.
Three facts about that choice, all easy to get wrong:

* **The gate moves with the score.** Dropping neighbour-contaminated pixels
  lowers every crowded cell's background, so the whole distribution shifts up
  and the legacy gate of 5 is a *different* operating point on the new scale --
  it admits 100-250 extra low-confidence cells per image and drops conversion
  to 54-72%. 16.5 is the median of the five per-image gates that reproduce the
  legacy rule's positive count.
* **The legacy gate was 10 here, not 5.** ``analyze_image``'s ``eps`` and the
  old ``CONFIDENCE_THRESHOLD`` encode the same cut and the effective gate is the
  larger of the two, so the previous deployed default was 10 while the notebook
  had already moved to 5. :attr:`RfpConfig.analyze_eps` derives both from one
  number so they cannot disagree again.
* **Soft-mask is deliberately absent.** It was built, measured on all five
  images and rejected: its extra positives convert at the image's *background*
  B3-Tub+ rate (16-33% against 66-80% for the cells every rule agrees on), i.e.
  they are mostly false positives. Re-adding it needs new evidence, not a flag.

See :mod:`neural_imgs.positivity.neighbour_contrast` for the rule itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from neural_imgs.positivity import (
    MarkerScoringInput,
    NeighbourContrastConfig,
    NeighbourContrastHeuristic,
)

#: Index of RFP in CHANNEL_NAMES.
RFP_CHANNEL = 2

RfpMethod = Literal["neighbour", "legacy"]


@dataclass(frozen=True)
class RfpConfig:
    """The RFP call's rule and operating point. Defaults are the notebook's.

    Parameters
    ----------
    method:
        ``"neighbour"`` (production) or ``"legacy"``. Both scores are computed
        for every cell either way -- this only picks which one carries the
        ``rfp_pos`` call, so a run can be re-read under the other rule from the
        saved table without re-classifying.
    legacy_threshold:
        Gate on the legacy ``score(%)``. Also sets ``analyze_image``'s ``eps``.
    neighbour_threshold / exclude_radius:
        Gate and neighbour-exclusion radius (native px) of the neighbour rule.
    """

    method: RfpMethod = "neighbour"
    legacy_threshold: float = 5.0
    neighbour_threshold: float = 16.5
    exclude_radius: int = 60

    @property
    def analyze_eps(self) -> float:
        """``analyze_image``'s positivity margin, in score units / 100.

        Derived, never passed separately: ``analyze_image`` marks a cell
        positive at ``mean > p95 + eps`` while the gate below compares
        ``score = (mean - p95) * 100``, so an ``eps`` out of step with
        ``legacy_threshold`` silently imposes whichever is stricter.
        """
        return self.legacy_threshold / 100.0

    @property
    def threshold(self) -> float:
        """The gate actually applied, in the active method's own units."""
        return (self.neighbour_threshold if self.method == "neighbour"
                else self.legacy_threshold)

    def neighbour_config(self) -> NeighbourContrastConfig:
        return NeighbourContrastConfig(
            exclude_radius=self.exclude_radius, threshold=self.neighbour_threshold,
        )

    def fingerprint(self) -> str:
        """Cache identity: any change here changes which cells are RFP+."""
        return (f"method={self.method}|legacy={self.legacy_threshold:g}"
                f"|neighbour={self.neighbour_threshold:g}|r={self.exclude_radius:d}")

    def describe(self) -> str:
        if self.method == "neighbour":
            return (f"RFP: neighbour-excluded rule, r={self.exclude_radius} px, "
                    f"gate {self.neighbour_threshold:g}")
        return f"RFP: legacy rule, gate {self.legacy_threshold:g}"


def score_rfp_neighbour(
    cells: list[dict], config: RfpConfig, channel_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour-excluded RFP score + call for ``cells``.

    Each cell must carry ``native_channels`` (index :data:`RFP_CHANNEL` filled),
    ``native_mask`` and ``neighbour_mask`` -- see
    :class:`~neural_imgs.inference.patch_extraction.FastCellExtractor`, which
    populates them as *views* and drops them again as soon as this has run.

    Safe to call per patch: the rule scores each cell only from that cell's own
    arrays, so scoring 16 patches separately and concatenating gives exactly the
    same numbers as one whole-image call -- which is what keeps peak memory flat
    instead of holding every cell's crops until the end.
    """
    if not cells:
        return np.empty(0), np.empty(0, dtype=bool)
    scored = NeighbourContrastHeuristic(config.neighbour_config()).score(
        MarkerScoringInput(cells=cells, channel_names=tuple(channel_names),
                           target_channels=(RFP_CHANNEL,))
    )[f"{channel_names[RFP_CHANNEL]}_neighbour"]
    return scored.scores, scored.positive
