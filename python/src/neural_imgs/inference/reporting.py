"""Post-classification reporting: UpSet co-positivity plot + side quantification.

Mirrors the UpSet cell in ``notebooks/single_cell_classification_fixed_model.ipynb``
exactly (fixed bar order, renamed sets, bar-gap/font styling) so a plugin- or
script-generated plot looks identical to the notebook's, whichever CZI it's run on.
The three reprogramming ratios printed underneath that cell are produced here too
(:class:`QuantificationSummary`), so the plot never travels without the numbers
that interpret it.

Uses matplotlib's object-oriented API (``Figure`` + an explicit Agg canvas)
rather than ``pyplot``: the figure is only ever written to disk, and pyplot
would try to create a GUI figure manager, which fails outright on a non-main
thread (e.g. the Appose worker driving the Fiji plugin on macOS).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from upsetplot import UpSet

from neural_imgs.inference.fixed_model_pipeline import CellResult

# Fixed bar order -- identical across every image regardless of that image's own
# counts (grouped by set size: none -> singles -> doubles -> triple), so figures
# across images are directly comparable bar-for-bar.
COMBO_ORDER = [
    (False, False, False),   # none
    (True,  False, False),   # OPC+
    (False, False, True),    # Transduced+
    (False, True,  False),   # Neuron+
    (True,  False, True),    # OPC+, Transduced+
    (True,  True,  False),   # OPC+, Neuron+
    (False, True,  True),    # Transduced+, Neuron+
    (True,  True,  True),    # OPC+, Transduced+, Neuron+
]

# Same (OPC, Neuron, Transduced) keys as COMBO_ORDER, spelled out for figure
# titles and log lines.
CASE_LABELS = {
    (False, False, False): "none",
    (True,  False, False): "OPC+",
    (False, False, True):  "Transduced+",
    (False, True,  False): "Neuron+",
    (True,  False, True):  "OPC+, Transduced+",
    (True,  True,  False): "OPC+, Neuron+",
    (False, True,  True):  "Transduced+, Neuron+",
    (True,  True,  True):  "OPC+, Transduced+, Neuron+",
}

FONT_SIZE = 15


def combo_of(cell: CellResult) -> tuple[bool, bool, bool]:
    """This cell's (OPC, Neuron, Transduced) case -- Neuron = B3-Tub+, Transduced = RFP+."""
    return (bool(cell.opc_pos), bool(cell.b3tub_pos), bool(cell.rfp_pos))


def combo_slug(combo: tuple[bool, bool, bool]) -> str:
    """Filename-safe form of a combo, e.g. ``(True, False, True)`` -> ``"101"``."""
    return "".join("1" if flag else "0" for flag in combo)


def _add_bar_gaps(ax, shrink: float = 0.8, horizontal: bool = False) -> None:
    """Shrink each bar (keeping it centered) to open up visible gaps between bars."""
    for bar in ax.patches:
        if horizontal:
            h = bar.get_height() * shrink
            bar.set_y(bar.get_y() + (bar.get_height() - h) / 2)
            bar.set_height(h)
        else:
            w = bar.get_width() * shrink
            bar.set_x(bar.get_x() + (bar.get_width() - w) / 2)
            bar.set_width(w)


def _set_text_fontsize(ax, fontsize: int) -> None:
    for txt in ax.texts:
        txt.set_fontsize(fontsize)


def upset_indicator_counts(cells: list[CellResult]) -> pd.Series:
    """Co-positivity counts per :data:`COMBO_ORDER` case, sets named OPC / Neuron
    (= B3-Tub+) / Transduced (= RFP+), matching the notebook's convention."""
    combo_counts = Counter(combo_of(c) for c in cells)
    return pd.Series(
        [combo_counts.get(combo, 0) for combo in COMBO_ORDER],
        index=pd.MultiIndex.from_tuples(COMBO_ORDER, names=["OPC", "Neuron", "Transduced"]),
        name="size",
    )


def save_upset_plot(
    cells: list[CellResult], output_path: str | Path, image_label: str = ""
) -> Path:
    """Build and save the OPC / Neuron / Transduced UpSet plot for ``cells``.

    ``image_label`` takes the place of the notebook's ``img {IMG_IDX}`` in the
    title -- pass the CZI's name so a figure is identifiable on its own.
    """
    output_path = Path(output_path)
    upset_data = upset_indicator_counts(cells)

    mpl.rcParams.update({"font.size": FONT_SIZE})
    fig = Figure(figsize=(14, 6))
    FigureCanvasAgg(fig)
    axes = UpSet(
        upset_data, subset_size="auto", show_counts=True,
        sort_by="input", sort_categories_by="input", include_empty_subsets=False,
    ).plot(fig)
    _add_bar_gaps(axes["intersections"], shrink=0.8, horizontal=False)
    _add_bar_gaps(axes["totals"], shrink=0.8, horizontal=True)
    _set_text_fontsize(axes["intersections"], FONT_SIZE)
    _set_text_fontsize(axes["totals"], FONT_SIZE)
    where = f" — {image_label}" if image_label else ""
    fig.suptitle(
        f"Num. of Detected cells{where}   ({len(cells)} cells)   —   "
        f"Neuron = B3-Tub+   |   Transduced = RFP+",
        fontsize=FONT_SIZE,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    return output_path


@dataclass(frozen=True)
class QuantificationSummary:
    """The three reprogramming ratios reported alongside the UpSet plot.

    All three are fractions of the **transduced** (RFP+) population, which is the
    only subset the experiment actually manipulated -- so each is ``nan``, not
    zero, when nothing was transduced: "no cells to measure" and "measured 0%"
    are different findings and must not print the same.
    """
    n_total: int
    n_transduced: int
    n_opc_transduced: int
    n_neuron_transduced: int
    opc_state_ratio: float
    transduction_performance: float
    conversion_ratio: float

    def lines(self) -> list[str]:
        """The notebook's printed block, one string per line."""
        def pct(value: float) -> str:
            return "n/a" if math.isnan(value) else f"{value:.1%}"
        return [
            f"OPC state ratio          = (OPC & Transduced) / Transduced "
            f"= {self.n_opc_transduced}/{self.n_transduced} = {pct(self.opc_state_ratio)}",
            f"Transduction performance = Transduced / Detected cells "
            f"= {self.n_transduced}/{self.n_total} = {pct(self.transduction_performance)}",
            f"Conversion ratio         = (Neuron & Transduced) / Transduced "
            f"= {self.n_neuron_transduced}/{self.n_transduced} = {pct(self.conversion_ratio)}",
        ]


def quantify(cells: list[CellResult]) -> QuantificationSummary:
    """Side quantification of a classified image (see :class:`QuantificationSummary`)."""
    n_total = len(cells)
    n_transduced = sum(1 for c in cells if c.rfp_pos)
    n_opc_transduced = sum(1 for c in cells if c.opc_pos and c.rfp_pos)
    n_neuron_transduced = sum(1 for c in cells if c.b3tub_pos and c.rfp_pos)
    nan = float("nan")
    return QuantificationSummary(
        n_total=n_total,
        n_transduced=n_transduced,
        n_opc_transduced=n_opc_transduced,
        n_neuron_transduced=n_neuron_transduced,
        opc_state_ratio=n_opc_transduced / n_transduced if n_transduced else nan,
        transduction_performance=n_transduced / n_total if n_total else nan,
        conversion_ratio=n_neuron_transduced / n_transduced if n_transduced else nan,
    )


def save_quantification(
    summary: QuantificationSummary, output_path: str | Path, image_label: str = ""
) -> Path:
    """Write ``summary`` next to the UpSet plot as a plain-text report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"Reprogramming quantification — {image_label}" if image_label \
        else "Reprogramming quantification"
    output_path.write_text("\n".join([header, ""] + summary.lines()) + "\n")
    return output_path


#: Per-cell CSV columns, in order. Matches the notebook's results table, plus
#: the two RFP columns the notebook writes under the same names -- so a plugin
#: CSV and a notebook CSV for the same image can be diffed column-for-column.
CELL_CSV_COLUMNS = (
    "cell_id", "patch_idx", "bb_y0", "bb_y1", "bb_x0", "bb_x1",
    "opc_prob", "opc_pos", "b3tub_prob", "b3tub_pos",
    "rfp_pos", "rfp_method", "rfp_neighbour_score", "rfp_neighbour_pos",
    "rfp_legacy_pos", "rfp_score", "opc_channel_score", "b3tub_channel_score",
    "n_cells_in_region",
)


def cells_to_frame(cells: list[CellResult]) -> pd.DataFrame:
    """One row per cell, in :data:`CELL_CSV_COLUMNS` order."""
    rows = []
    for c in cells:
        y0, y1, x0, x1 = c.bb_position
        rows.append({
            "cell_id": int(c.cell_id), "patch_idx": int(c.patch_idx),
            "bb_y0": int(y0), "bb_y1": int(y1), "bb_x0": int(x0), "bb_x1": int(x1),
            "opc_prob": round(float(c.opc_prob), 4), "opc_pos": bool(c.opc_pos),
            "b3tub_prob": round(float(c.b3tub_prob), 4), "b3tub_pos": bool(c.b3tub_pos),
            "rfp_pos": bool(c.rfp_pos), "rfp_method": str(c.rfp_method),
            "rfp_neighbour_score": round(float(c.rfp_neighbour_score), 4),
            "rfp_neighbour_pos": bool(c.rfp_neighbour_pos),
            "rfp_legacy_pos": bool(c.rfp_legacy_pos),
            "rfp_score": round(float(c.rfp_score), 4),
            "opc_channel_score": round(float(c.opc_channel_score), 4),
            "b3tub_channel_score": round(float(c.b3tub_channel_score), 4),
            "n_cells_in_region": int(c.n_cells_in_region),
        })
    return pd.DataFrame(rows, columns=list(CELL_CSV_COLUMNS))


def save_cells_csv(cells: list[CellResult], output_path: str | Path) -> Path:
    """Write the per-cell table next to the run's other outputs.

    Written on the Python side rather than from the Fiji results table because
    it is the side that has every column: the table shown in Fiji is a view for
    the user, while this file is the run's record and has to survive a batch
    where no table is ever opened.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cells_to_frame(cells).to_csv(output_path, index=False)
    return output_path
