"""Per-channel positivity calls for single cells: intensity heuristic + trained models."""

from neural_imgs.positivity.base import (
    BaseMarkerScorer,
    MarkerScore,
    MarkerScoringInput,
    MarkerScoringOutput,
)
from neural_imgs.positivity.heuristic import (
    LocalContrastConfig,
    LocalContrastHeuristic,
    pad_for_nucleus_diameter,
)
from neural_imgs.positivity.neighbour_contrast import (
    NeighbourContrastComponents,
    NeighbourContrastConfig,
    NeighbourContrastHeuristic,
    neighbour_mask_from_labels,
)
from neural_imgs.positivity.soft_mask import (
    SoftMaskComponents,
    SoftMaskContrastConfig,
    SoftMaskContrastHeuristic,
    cells_with_native_views,
    mirror_fdr_table,
)

__all__ = [
    "BaseMarkerScorer", "MarkerScore", "MarkerScoringInput", "MarkerScoringOutput",
    "LocalContrastConfig", "LocalContrastHeuristic", "pad_for_nucleus_diameter",
    "NeighbourContrastConfig", "NeighbourContrastHeuristic",
    "NeighbourContrastComponents", "neighbour_mask_from_labels",
    "SoftMaskContrastConfig", "SoftMaskContrastHeuristic", "SoftMaskComponents",
    "cells_with_native_views", "mirror_fdr_table",
    # neural_imgs.positivity.classifier is imported on demand -- it pulls in torch.
]
