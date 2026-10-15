"""Post-classification reporting: UpSet co-positivity plot + side quantification.

Mirrors the UpSet cell in ``notebooks/single_cell_classification_fixed_model.ipynb``
exactly (fixed bar order, renamed sets, bar-gap/font styling) so a plugin- or
script-generated plot looks identical to the notebook's, whichever CZI it's run on.
The three reprogramming ratios printed underneath that cell are produced here too
(:class:`QuantificationSummary`), so the plot never travels without the numbers
that interpret it.

Both shrink with the run's channel layout. A run that measured only two of the
three markers plots two sets and four cases, and reports only the ratios those
two can support -- naming the third as *not measured* rather than printing a
ratio computed as if every cell had been found negative in it. Which markers a
run measured comes from its :class:`~neural_imgs.channels.MarkerAxes`; with all
three present every figure and every line is byte-for-byte what it always was.

Uses matplotlib's object-oriented API (``Figure`` + an explicit Agg canvas)
rather than ``pyplot``: the figure is only ever written to disk, and pyplot
would try to create a GUI figure manager, which fails outright on a non-main
thread (e.g. the Appose worker driving the Fiji plugin on macOS).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from upsetplot import UpSet

from neural_imgs.channels import FULL_AXES, MarkerAxes, set_label
from neural_imgs.inference.fixed_model_pipeline import CellResult

# Fixed bar order for the three-marker case -- identical across every image
# regardless of that image's own counts (grouped by set size: none -> singles ->
# doubles -> triple), so figures across images are directly comparable
# bar-for-bar. Generated from MarkerAxes rather than written out, so a run with
# fewer markers gets the same ordering rule instead of a second hand-kept list.
COMBO_ORDER = FULL_AXES.combo_order()

# Same (OPC, Neuron, Transduced) keys as COMBO_ORDER, spelled out for figure
# titles and log lines.
CASE_LABELS = {combo: FULL_AXES.label(combo) for combo in COMBO_ORDER}

FONT_SIZE = 15


def combo_of(
    cell: CellResult, axes: MarkerAxes = FULL_AXES
) -> tuple[bool, ...]:
    """This cell's case over ``axes`` -- Neuron = B3-Tub+, Transduced = RFP+.

    With the default axes this is the familiar ``(OPC, Neuron, Transduced)``
    triple; with fewer markers it is the same flags minus the ones the run did
    not measure, so nothing is silently counted as negative.
    """
    return axes.flags_of({
        "OPC": bool(cell.opc_pos),
        "B3Tub": bool(cell.b3tub_pos),
        "RFP": bool(cell.rfp_pos),
    })


def combo_slug(combo: tuple[bool, ...]) -> str:
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


def upset_indicator_counts(
    cells: list[CellResult], axes: MarkerAxes = FULL_AXES
) -> pd.Series:
    """Co-positivity counts per case, sets named OPC / Neuron (= B3-Tub+) /
    Transduced (= RFP+), matching the notebook's convention.

    ``axes`` is the run's measured markers; the index has one level per measured
    marker, in the published order.
    """
    order = axes.combo_order()
    combo_counts = Counter(combo_of(c, axes) for c in cells)
    return pd.Series(
        [combo_counts.get(combo, 0) for combo in order],
        index=pd.MultiIndex.from_tuples(order, names=list(axes.set_names)),
        name="size",
    )


def save_upset_plot(
    cells: list[CellResult], output_path: str | Path, image_label: str = "",
    axes: MarkerAxes = FULL_AXES,
) -> Path:
    """Build and save the OPC / Neuron / Transduced UpSet plot for ``cells``.

    ``image_label`` takes the place of the notebook's ``img {IMG_IDX}`` in the
    title -- pass the CZI's name so a figure is identifiable on its own.
    ``axes`` restricts the sets to the markers the run measured; with none of
    them there is nothing to intersect and this raises rather than drawing an
    empty figure.
    """
    if not axes:
        raise ValueError(
            "no marker channels in this run, so there are no co-positivity "
            "sets to plot (the cell count is still reported)")
    output_path = Path(output_path)
    upset_data = upset_indicator_counts(cells, axes)

    mpl.rcParams.update({"font.size": FONT_SIZE})
    fig = Figure(figsize=(14, 6))
    FigureCanvasAgg(fig)
    # `plot_axes` are matplotlib axes; `axes` above is the MarkerAxes. Named
    # apart on purpose -- the two were one name here until the title below
    # needed the marker names after the plot had been drawn.
    plot_axes = UpSet(
        upset_data, subset_size="auto", show_counts=True,
        sort_by="input", sort_categories_by="input", include_empty_subsets=False,
    ).plot(fig)
    _add_bar_gaps(plot_axes["intersections"], shrink=0.8, horizontal=False)
    _add_bar_gaps(plot_axes["totals"], shrink=0.8, horizontal=True)
    _set_text_fontsize(plot_axes["intersections"], FONT_SIZE)
    _set_text_fontsize(plot_axes["totals"], FONT_SIZE)
    where = f" — {image_label}" if image_label else ""
    # The legend names only the sets actually plotted: a key for "Transduced =
    # RFP+" on a figure with no Transduced set reads as a missing bar.
    legend = "   |   ".join(
        f"{name} = {marker}+" for marker, name in zip(axes.markers, axes.set_names)
        if name != marker
    )
    fig.suptitle(
        f"Num. of Detected cells{where}   ({len(cells)} cells)"
        + (f"   —   {legend}" if legend else ""),
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

    A ratio whose markers the run did not measure at all is ``nan`` for a second,
    different reason, and :meth:`lines` does not print it as a ratio -- it names
    the marker as not measured instead. Printing ``0/0 = n/a`` for a channel that
    was never acquired invites reading it as "none found".
    """
    n_total: int
    n_transduced: int
    n_opc_transduced: int
    n_neuron_transduced: int
    opc_state_ratio: float
    transduction_performance: float
    conversion_ratio: float
    #: Markers this image was actually scored for. Defaults to all three, so an
    #: older caller keeps the published three-line block verbatim.
    axes: MarkerAxes = FULL_AXES
    #: Positive counts per measured marker, keyed by marker role.
    n_positive: "dict[str, int]" = field(default_factory=dict)

    def lines(self) -> list[str]:
        """The notebook's printed block, one string per line.

        With all three markers measured this is exactly the published block. With
        fewer it becomes the detected count, each measured marker's positive
        count, whichever ratios survive, and a closing line naming what was not
        measured -- so a shorter report is never mistaken for a worse result.
        """
        def pct(value: float) -> str:
            return "n/a" if math.isnan(value) else f"{value:.1%}"

        markers = self.axes.markers
        if markers == FULL_AXES.markers:
            return [
                f"OPC state ratio          = (OPC & Transduced) / Transduced "
                f"= {self.n_opc_transduced}/{self.n_transduced} = {pct(self.opc_state_ratio)}",
                f"Transduction performance = Transduced / Detected cells "
                f"= {self.n_transduced}/{self.n_total} = {pct(self.transduction_performance)}",
                f"Conversion ratio         = (Neuron & Transduced) / Transduced "
                f"= {self.n_neuron_transduced}/{self.n_transduced} = {pct(self.conversion_ratio)}",
            ]

        # Same column as the published block: label padded to 25, then "= ".
        def row(label: str, body: str) -> str:
            return f"{label:<25}= {body}"

        out = [row("Detected cells", str(self.n_total))]
        for marker, name in zip(markers, self.axes.set_names):
            count = self.n_positive.get(marker, 0)
            share = count / self.n_total if self.n_total else float("nan")
            out.append(row(f"{name}+", f"{count}/{self.n_total} = {pct(share)}"))
        if "RFP" in markers:
            out.append(row(
                "Transduction performance",
                f"Transduced / Detected cells = {self.n_transduced}/{self.n_total} "
                f"= {pct(self.transduction_performance)}"))
            if "OPC" in markers:
                out.append(row(
                    "OPC state ratio",
                    f"(OPC & Transduced) / Transduced "
                    f"= {self.n_opc_transduced}/{self.n_transduced} "
                    f"= {pct(self.opc_state_ratio)}"))
            if "B3Tub" in markers:
                out.append(row(
                    "Conversion ratio",
                    f"(Neuron & Transduced) / Transduced "
                    f"= {self.n_neuron_transduced}/{self.n_transduced} "
                    f"= {pct(self.conversion_ratio)}"))
        missing = [m for m in FULL_AXES.markers if m not in markers]
        if missing:
            out.append(
                "Not measured in this run: "
                + ", ".join(set_label(m) for m in missing)
                + " -- every ratio needing one is omitted, not reported as zero")
        return out


def quantify(
    cells: list[CellResult], axes: MarkerAxes = FULL_AXES
) -> QuantificationSummary:
    """Side quantification of a classified image (see :class:`QuantificationSummary`).

    ``axes`` is the run's measured markers; a ratio needing an unmeasured one is
    ``nan`` and is reported as not measured rather than as a result.
    """
    measured = set(axes.markers)
    n_total = len(cells)
    n_transduced = sum(1 for c in cells if c.rfp_pos)
    n_opc_transduced = sum(1 for c in cells if c.opc_pos and c.rfp_pos)
    n_neuron_transduced = sum(1 for c in cells if c.b3tub_pos and c.rfp_pos)
    nan = float("nan")

    def ratio(numerator: int, denominator: int, needs: "tuple[str, ...]") -> float:
        if not measured.issuperset(needs):
            return nan
        return numerator / denominator if denominator else nan

    positives = {
        "OPC": sum(1 for c in cells if c.opc_pos),
        "B3Tub": sum(1 for c in cells if c.b3tub_pos),
        "RFP": n_transduced,
    }
    return QuantificationSummary(
        n_total=n_total,
        n_transduced=n_transduced,
        n_opc_transduced=n_opc_transduced,
        n_neuron_transduced=n_neuron_transduced,
        opc_state_ratio=ratio(n_opc_transduced, n_transduced, ("OPC", "RFP")),
        transduction_performance=ratio(n_transduced, n_total, ("RFP",)),
        conversion_ratio=ratio(n_neuron_transduced, n_transduced, ("B3Tub", "RFP")),
        axes=axes,
        n_positive={m: positives[m] for m in axes.markers},
    )


def save_quantification(
    summary: QuantificationSummary, output_path: str | Path, image_label: str = "",
    channel_note: str = "",
) -> Path:
    """Write ``summary`` next to the UpSet plot as a plain-text report.

    ``channel_note`` is the run's channel layout
    (:meth:`~neural_imgs.channels.ChannelLayout.describe`), recorded in the file
    because the numbers below cannot be read without knowing which markers were
    measured -- and the file outlives the Log that said so.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"Reprogramming quantification — {image_label}" if image_label \
        else "Reprogramming quantification"
    body = [header, ""]
    if channel_note:
        body += [channel_note, ""]
    output_path.write_text("\n".join(body + summary.lines()) + "\n")
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


def cells_to_frame(
    cells: list[CellResult], axes: MarkerAxes = FULL_AXES
) -> pd.DataFrame:
    """One row per cell, in :data:`CELL_CSV_COLUMNS` order.

    The columns are always the same set, whatever ``axes`` says -- a stable
    header is what lets two runs' CSVs be diffed or concatenated. What changes
    is the *values*: a marker the run did not measure gets **blank** cells
    rather than ``False`` and ``0.0``, because a column of ``False`` for a
    channel that was never acquired reads as "no cell was positive".
    """
    unmeasured = {m for m in FULL_AXES.markers if m not in axes.markers}

    def call(marker: str, value: bool) -> "bool | None":
        return None if marker in unmeasured else bool(value)

    def score(marker: str, value: float) -> float:
        return float("nan") if marker in unmeasured else round(float(value), 4)

    rows = []
    for c in cells:
        y0, y1, x0, x1 = c.bb_position
        rows.append({
            "cell_id": int(c.cell_id), "patch_idx": int(c.patch_idx),
            "bb_y0": int(y0), "bb_y1": int(y1), "bb_x0": int(x0), "bb_x1": int(x1),
            "opc_prob": score("OPC", c.opc_prob), "opc_pos": call("OPC", c.opc_pos),
            "b3tub_prob": score("B3Tub", c.b3tub_prob),
            "b3tub_pos": call("B3Tub", c.b3tub_pos),
            "rfp_pos": call("RFP", c.rfp_pos),
            "rfp_method": "" if "RFP" in unmeasured else str(c.rfp_method),
            "rfp_neighbour_score": score("RFP", c.rfp_neighbour_score),
            "rfp_neighbour_pos": call("RFP", c.rfp_neighbour_pos),
            "rfp_legacy_pos": call("RFP", c.rfp_legacy_pos),
            "rfp_score": score("RFP", c.rfp_score),
            "opc_channel_score": score("OPC", c.opc_channel_score),
            "b3tub_channel_score": score("B3Tub", c.b3tub_channel_score),
            "n_cells_in_region": int(c.n_cells_in_region),
        })
    return pd.DataFrame(rows, columns=list(CELL_CSV_COLUMNS))


def save_cells_csv(
    cells: list[CellResult], output_path: str | Path, axes: MarkerAxes = FULL_AXES
) -> Path:
    """Write the per-cell table next to the run's other outputs.

    Written on the Python side rather than from the Fiji results table because
    it is the side that has every column: the table shown in Fiji is a view for
    the user, while this file is the run's record and has to survive a batch
    where no table is ever opened.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cells_to_frame(cells, axes).to_csv(output_path, index=False)
    return output_path
