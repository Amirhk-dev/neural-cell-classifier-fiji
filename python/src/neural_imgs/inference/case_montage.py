"""One representative single cell per UpSet case, shown across all 5 channels.

Mirrors the "One example cell per UpSet case" cell of
``notebooks/single_cell_classification_fixed_model.ipynb``: for each of the 8
:data:`~neural_imgs.inference.reporting.COMBO_ORDER` cases, pick one cell and
render DAPI + mask followed by DAPI / OPC / RFP / B3-Tub / BF, titled with the
case, the cell id and that cell's classifier probabilities. It is the visual
audit of the UpSet bars: what the models actually saw for each call.

The pixels are the notebook's, not an approximation of them: panels come from
:meth:`~neural_imgs.inference.fixed_model_pipeline.PatchPreprocessor.normalized_patch`,
i.e. the same BaSiC-corrected, percentile-normalised patch the notebook slices
its ``dapi_crop``/``opc_crop``/... out of and the same one CellPose ran on.

One thing degrades rather than fails: **the mask contour** is drawn only when
the cell carries a ``native_mask``. Results restored from a cache entry written
without its mask sidecar may not have one (see
:mod:`neural_imgs.inference.result_cache`); those panels render without the
outline.

As with :mod:`neural_imgs.inference.reporting`, figures are built through
matplotlib's OO API -- never pyplot -- so this is safe on the Appose worker
thread that drives the Fiji plugin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from skimage.measure import find_contours

from neural_imgs.inference.fixed_model_pipeline import (
    CHANNEL_NAMES,
    CellResult,
    PatchPreprocessor,
)
from neural_imgs.inference.reporting import (
    CASE_LABELS,
    COMBO_ORDER,
    combo_of,
    combo_slug,
)


@dataclass(frozen=True)
class CaseExample:
    """The cell chosen to stand for one UpSet case.

    ``cell`` is ``None`` when the image contains no cell of this case -- the case
    is still reported (as a skip) so a montage set always covers all 8 cases.
    """
    combo: tuple[bool, bool, bool]
    label: str
    cell: CellResult | None
    n_candidates: int
    example_index: int


class CaseExamplePicker:
    """Picks one cell per case, in the fixed :data:`COMBO_ORDER` order.

    ``example_index`` maps a case to *which* of its matching cells to take
    (0 = first, the default for cases left out); the index wraps, so any value
    is safe and can be bumped to page through a case's other cells.
    """

    def __init__(self, example_index: dict[tuple[bool, bool, bool], int] | None = None) -> None:
        self.example_index = dict(example_index or {})

    def pick(self, cells: list[CellResult]) -> list[CaseExample]:
        by_combo: dict[tuple[bool, bool, bool], list[CellResult]] = {}
        for cell in cells:
            by_combo.setdefault(combo_of(cell), []).append(cell)

        examples: list[CaseExample] = []
        for combo in COMBO_ORDER:
            candidates = by_combo.get(combo, [])
            which = (self.example_index.get(combo, 0) % len(candidates)) if candidates else 0
            examples.append(CaseExample(
                combo=combo,
                label=CASE_LABELS[combo],
                cell=candidates[which] if candidates else None,
                n_candidates=len(candidates),
                example_index=which,
            ))
        return examples


@dataclass(frozen=True)
class CellDisplay:
    """Display-ready pixels for one cell: ``channels`` is (5, H, W) in [0, 1]
    (DAPI/OPC/RFP/B3Tub/BF), ``mask`` the native DAPI mask or None."""
    channels: np.ndarray
    mask: np.ndarray | None


class CellDisplayProvider:
    """Cuts each cell's display crop out of its own preprocessed patch.

    Holds **one** normalised patch at a time: at 5 channels of float that array
    is hundreds of MB, so caching several would be worse than recomputing.
    Ask for cells grouped by patch (:meth:`display_for_all` does that) and each
    patch is preprocessed exactly once.
    """

    def __init__(self, preprocessor: PatchPreprocessor) -> None:
        self.preprocessor = preprocessor
        self._patch_idx: int | None = None
        self._patch: np.ndarray | None = None

    def _normalized_patch(self, patch_idx: int) -> np.ndarray:
        if self._patch_idx != patch_idx:
            self._patch = self.preprocessor.normalized_patch(patch_idx)
            self._patch_idx = patch_idx
        return self._patch

    def display_for(self, cell: CellResult) -> CellDisplay:
        y0, y1, x0, x1 = cell.bb_position
        pre = self._normalized_patch(cell.patch_idx)
        return CellDisplay(
            channels=pre[:, y0:y1, x0:x1].copy(), mask=cell.native_mask,
        )

    def display_for_all(self, cells: list[CellResult]) -> dict[int, CellDisplay]:
        """Displays for ``cells``, keyed by cell id, preprocessing each patch once."""
        return {
            cell.cell_id: self.display_for(cell)
            for cell in sorted(cells, key=lambda c: c.patch_idx)
        }


@dataclass(frozen=True)
class CaseMontageOutput:
    """What :meth:`CaseMontageWriter.write` produced.

    ``paths`` maps case label -> PNG written; ``skipped`` lists the case labels
    with no cell in this image; ``report_lines`` is the notebook's per-case
    "N cells -> showing #i (cell id)" block, ready to log verbatim.
    """
    paths: dict[str, Path]
    skipped: list[str]
    report_lines: list[str]


class CaseMontageWriter:
    """Renders and saves one PNG per UpSet case."""

    def __init__(self, provider: CellDisplayProvider) -> None:
        self.provider = provider

    def write(
        self, examples: list[CaseExample], out_dir: str | Path,
        stem: str, image_label: str = "",
    ) -> CaseMontageOutput:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # All the crops first, grouped by patch, so no patch is preprocessed
        # twice just because two cases happened to pick cells from it.
        displays = self.provider.display_for_all(
            [e.cell for e in examples if e.cell is not None]
        )

        paths: dict[str, Path] = {}
        skipped: list[str] = []
        lines: list[str] = []
        for example in examples:
            if example.cell is None:
                skipped.append(example.label)
                lines.append(f"{example.label:<28} no cells — skipped")
                continue
            lines.append(
                f"{example.label:<28} {example.n_candidates:>5} cells -> "
                f"showing #{example.example_index} (cell {example.cell.cell_id})"
            )
            path = out_dir / f"{stem}_upset_case_{combo_slug(example.combo)}.png"
            figure = self._render(example, displays[example.cell.cell_id], image_label)
            figure.savefig(path, dpi=100, bbox_inches="tight")
            paths[example.label] = path
        return CaseMontageOutput(paths=paths, skipped=skipped, report_lines=lines)

    def _render(
        self, example: CaseExample, display: CellDisplay, image_label: str
    ) -> Figure:
        cell = example.cell
        contours = (
            find_contours(display.mask.astype(float), 0.5) if display.mask is not None else []
        )

        fig = Figure(figsize=(16, 3.2))
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 6)

        # Column 0 is the mask panel; the same contour is drawn on every channel
        # so the signal in each one can be read against the cell outline.
        panels = [(display.channels[0], "DAPI + mask")] + [
            (display.channels[c], name) for c, name in enumerate(CHANNEL_NAMES)
        ]
        for ax, (channel, name) in zip(axes, panels):
            ax.imshow(channel, cmap="gray", vmin=0, vmax=1)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color="lime", linewidth=1.0)
            ax.axis("off")
            ax.set_title(name, fontsize=11)

        where = f" — {image_label}" if image_label else ""
        fig.suptitle(
            f"{example.label} — cell {cell.cell_id}{where}\n"
            f"OPC {cell.opc_prob:.2f} / B3 {cell.b3tub_prob:.2f} / "
            f"RFP {int(cell.rfp_pos)}",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        return fig


def save_case_montages(
    cells: list[CellResult], preprocessor: PatchPreprocessor,
    out_dir: str | Path, stem: str, image_label: str = "",
    example_index: dict[tuple[bool, bool, bool], int] | None = None,
) -> CaseMontageOutput:
    """Convenience wrapper: pick one cell per UpSet case and save all 8 montages.

    ``preprocessor`` must be the one fitted on this image -- reuse the fit from
    the classification run rather than making a second one.
    """
    examples = CaseExamplePicker(example_index).pick(cells)
    writer = CaseMontageWriter(CellDisplayProvider(preprocessor))
    return writer.write(examples, out_dir, stem, image_label)
