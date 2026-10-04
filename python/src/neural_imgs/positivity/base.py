"""Typed contract for deciding whether a single cell is positive in a marker channel.

Two very different things answer the same question in this project -- the
intensity heuristic from the OPC pipeline (``processing.analyze_image``) and the
trained per-marker ResNet-18s -- so both are expressed here as *scorers*: given
a set of single cells and a channel, produce one continuous score per cell plus
the boolean decision its threshold implies. Keeping the output shape identical
is what lets the two be tabulated, plotted and cross-tabulated side by side
without the caller special-casing either one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class MarkerScore:
    """One scorer's verdict on every cell, for one channel.

    Parameters
    ----------
    channel_name / channel_index:
        The channel that was *measured* (e.g. ``"MAP2"`` / ``3``).
    method:
        How it was measured -- ``"heuristic"`` or ``"clf:<marker>"``. A single
        channel can carry several of these at once (the whole point of this
        module), so it is part of the identity of a score, not metadata.
    score_name:
        Human label for the units of ``scores`` (``"confidence(%)"``, ``"prob"``).
    scores:
        ``(n_cells,)`` float, higher = more likely positive. NaN where the score
        is undefined (e.g. an empty mask).
    positive:
        ``(n_cells,)`` bool -- the decision ``scores >= threshold`` (plus any
        extra condition the scorer imposes; see the concrete implementations).
    threshold:
        The cut applied to ``scores``.
    """

    channel_name: str
    channel_index: int
    method: str
    score_name: str
    scores: np.ndarray
    positive: np.ndarray
    threshold: float

    @property
    def key(self) -> str:
        """Short column stem, e.g. ``"MAP2_clf:B3Tub"``."""
        return f"{self.channel_name}_{self.method}"

    @property
    def n_positive(self) -> int:
        return int(self.positive.sum())

    @property
    def fraction_positive(self) -> float:
        return float(self.positive.mean()) if self.positive.size else float("nan")

    def as_columns(self) -> dict[str, np.ndarray]:
        """Two DataFrame columns: the score and the decision."""
        return {f"{self.key}_score": self.scores, f"{self.key}_pos": self.positive}


@dataclass
class MarkerScoringInput:
    """Cells to score and which channels to score them in.

    ``cells`` are the dicts produced by
    :func:`neural_imgs.processing.processing.extract_cells_nodistort_noresize`
    (native-resolution crops), each carrying at least ``label``, ``mask`` and
    ``raw_img``. Scorers that need a different view of the same cell (the
    heuristic wants the 512-resized, border-zeroed crop; the classifier wants a
    crop resampled to the training pixel size) build it themselves from the
    image the scorer was constructed with, so the caller never has to hold two
    parallel cell lists.
    """

    cells: list[dict]
    channel_names: tuple[str, ...]
    target_channels: tuple[int, ...]


@dataclass
class MarkerScoringOutput:
    """Every score produced for one image, keyed by :attr:`MarkerScore.key`."""

    labels: np.ndarray                      # (n_cells,) CellPose label per row
    scores: dict[str, MarkerScore]

    def __getitem__(self, key: str) -> MarkerScore:
        return self.scores[key]

    def for_channel(self, channel_name: str) -> list[MarkerScore]:
        return [s for s in self.scores.values() if s.channel_name == channel_name]

    def as_columns(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for score in self.scores.values():
            out.update(score.as_columns())
        return out


class BaseMarkerScorer(ABC):
    @abstractmethod
    def score(self, input: MarkerScoringInput) -> MarkerScoringOutput:
        """Score every cell in every requested channel."""
