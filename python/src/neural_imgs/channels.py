"""Which image channel carries which marker, and which markers a run can call.

The pipeline used to assume one fixed acquisition: exactly five channels in the
order DAPI, OPC, RFP, B3-Tub, BF. Every stage encoded that assumption as a bare
integer -- CellPose ran on ``img[0]``, the OPC classifier read channel 1, the
RFP rule channel 2 -- so an image acquired without one of those channels, or
with them in another order, could not be classified at all: it failed on
``patch_image``'s ``assert C == 5`` before anything was detected.

:class:`ChannelLayout` replaces those integers with a single declared mapping,
and :class:`MarkerAxes` derives from it *which* markers the run can actually
call. Two independent things follow from one declaration:

* **Channel order is free.** ``DAPI`` may sit anywhere; a channel the run has no
  use for is declared :data:`UNUSED` and is read, illumination-corrected and
  then ignored.
* **Markers are optional.** A layout without an ``OPC`` channel runs the rest of
  the pipeline and reports OPC as *not measured* rather than as negative -- the
  distinction matters, because "no OPC channel" and "no OPC+ cells" are
  different findings that must not print the same.

``DAPI`` is the one channel that cannot be omitted: nuclei are detected on it,
so without it there are no single cells for anything downstream to score.

The default layout is the historical one, so a caller that declares nothing
behaves exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

#: The nucleus channel. Required: CellPose detects on it and every cell's mask,
#: crop centre and bounding box come from that detection.
DAPI = "DAPI"

#: Marker roles a cell can be called positive in, in the order their flags are
#: stored (``(opc_pos, b3tub_pos, rfp_pos)``).
MARKER_ROLES = ("OPC", "B3Tub", "RFP")

#: Markers with a trained ResNet-18 behind them; see ``PROD_CONFIG``.
CLASSIFIER_MARKERS = ("OPC", "B3Tub")

#: Brightfield. Carried for display only -- nothing scores it.
BF = "BF"

#: Every role a channel may be given.
ROLES = (DAPI,) + MARKER_ROLES + (BF,)

#: A channel present in the file that this run does not use.
UNUSED = "-"

#: The historical fixed acquisition, and the default when nothing is declared.
DEFAULT_ORDER = (DAPI, "OPC", "RFP", "B3Tub", BF)

#: Display names of the three marker roles, as the UpSet plot and the
#: quantification name them. "Neuron" and "Transduced" are what the biologists
#: call these two readouts; the role names are what the code calls the channels.
SET_NAMES = {"OPC": "OPC", "B3Tub": "Neuron", "RFP": "Transduced"}

#: Order the marker sets are *enumerated* in, which is not the order their flags
#: are stored in (:data:`MARKER_ROLES`). Fixing both separately is what keeps the
#: UpSet bar order identical to the published figures.
DISPLAY_ORDER = ("OPC", "RFP", "B3Tub")

#: Spellings accepted for each role, lower-cased and stripped of ``-``, ``_``
#: and spaces. Only for callers that write a layout by hand (a script, a
#: notebook, the Python API); the Fiji plugin always sends canonical role names,
#: so this table has no second copy on the Java side to drift from.
_ALIASES = {
    "dapi": DAPI, "nuclei": DAPI, "nucleus": DAPI, "hoechst": DAPI, "dna": DAPI,
    "opc": "OPC", "olig2": "OPC", "o4": "OPC",
    "rfp": "RFP", "tdtomato": "RFP", "mcherry": "RFP", "red": "RFP",
    "b3tub": "B3Tub", "b3": "B3Tub", "biiitub": "B3Tub", "biiitubulin": "B3Tub",
    "tubb3": "B3Tub", "tuj1": "B3Tub", "betatubulin": "B3Tub",
    "bf": BF, "brightfield": BF, "trans": BF, "phase": BF,
}

#: Spellings that mean "present in the file, not used by this run".
_UNUSED_ALIASES = frozenset({"-", "", "none", "unused", "ignore", "other", "x", "na"})


def set_label(marker: str) -> str:
    """``"Neuron (B3Tub)"`` -- the readout's name plus the channel's, when they
    differ. Both halves are needed in a log line: the reader knows the marker by
    one name and the dialog field by the other."""
    name = SET_NAMES[marker]
    return name if name == marker else f"{name} ({marker})"


def _normalise(token: str) -> str:
    return token.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _role_of(token: str) -> str:
    """The role ``token`` names, or :data:`UNUSED`. Raises on anything else."""
    key = _normalise(token)
    if key in _UNUSED_ALIASES:
        return UNUSED
    # Canonical role names are matched first and case-insensitively, so the
    # exact strings this module hands out always parse back.
    for role in ROLES:
        if key == _normalise(role):
            return role
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(
        f"unknown channel name {token.strip()!r}; expected one of "
        f"{', '.join(ROLES)} or {UNUSED!r} for a channel this run does not use"
    )


@dataclass(frozen=True)
class ChannelLayout:
    """What each channel of one image is, in file order.

    ``roles[i]`` is the role of the image's channel ``i``: a name from
    :data:`ROLES`, or :data:`UNUSED`. Exactly one channel must be
    :data:`DAPI`; every other role may appear at most once, or not at all.

    A layout may be *shorter* than the image it is applied to -- see
    :meth:`for_image`. That is the normal case when the channels a run cares
    about sit at the front of the file: the trailing channels are read and then
    ignored, rather than having to be spelled out as :data:`UNUSED`.
    """

    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.roles:
            raise ValueError("a channel layout needs at least one channel")
        assigned = [r for r in self.roles if r != UNUSED]
        duplicates = sorted({r for r in assigned if assigned.count(r) > 1})
        if duplicates:
            raise ValueError(
                f"each channel role may be used at most once; "
                f"{', '.join(duplicates)} appears more than once"
            )
        if DAPI not in assigned:
            raise ValueError(
                "no DAPI channel declared. Nuclei are detected on DAPI, so "
                "without it there are no cells to classify -- every other "
                "channel is optional, DAPI is not"
            )

    # -- construction -------------------------------------------------------

    @classmethod
    def default(cls) -> "ChannelLayout":
        """The historical five-channel acquisition, DAPI/OPC/RFP/B3Tub/BF."""
        return cls(roles=DEFAULT_ORDER)

    @classmethod
    def parse(cls, spec: "str | Sequence[str] | ChannelLayout | None") -> "ChannelLayout":
        """A layout from ``"DAPI,OPC,RFP,B3Tub,BF"`` or the same as a sequence.

        ``None`` or an empty/blank string gives :meth:`default`, so an older
        caller that passes nothing keeps the old behaviour. Names are matched
        case-insensitively and through :data:`_ALIASES`; ``-`` marks a channel
        the run does not use.
        """
        if spec is None:
            return cls.default()
        if isinstance(spec, ChannelLayout):
            return spec
        tokens = list(spec.split(",")) if isinstance(spec, str) else list(spec)
        if not any(token.strip() for token in tokens):
            return cls.default()
        return cls(roles=tuple(_role_of(token) for token in tokens))

    def for_image(self, n_image_channels: int) -> "ChannelLayout":
        """This layout padded out to ``n_image_channels``.

        Raises when the layout describes *more* channels than the image has --
        that is a genuine mismatch between what the user declared and what was
        acquired, and classifying it anyway would score the wrong pixels. Fewer
        is fine and is padded with :data:`UNUSED`.
        """
        if n_image_channels < len(self.roles):
            raise ValueError(
                f"the channel layout describes {len(self.roles)} channels "
                f"({self.describe_order()}) but the image has only "
                f"{n_image_channels}. Fix the channel assignment to match the "
                f"image, or use an image with all the declared channels"
            )
        if n_image_channels == len(self.roles):
            return self
        return ChannelLayout(
            roles=self.roles + (UNUSED,) * (n_image_channels - len(self.roles)))

    # -- what is where ------------------------------------------------------

    @property
    def n_channels(self) -> int:
        return len(self.roles)

    def index_of(self, role: str) -> "int | None":
        """Channel index carrying ``role``, or None when the run has no such channel."""
        return self.roles.index(role) if role in self.roles else None

    def has(self, role: str) -> bool:
        return role in self.roles

    @property
    def dapi(self) -> int:
        """The nucleus channel's index. Always present -- see :meth:`__post_init__`."""
        return self.roles.index(DAPI)

    @property
    def markers(self) -> tuple[str, ...]:
        """Marker roles this layout can call, in :data:`MARKER_ROLES` order."""
        return tuple(m for m in MARKER_ROLES if m in self.roles)

    @property
    def classifier_markers(self) -> tuple[str, ...]:
        """Markers with both a channel here and a trained classifier."""
        return tuple(m for m in CLASSIFIER_MARKERS if m in self.roles)

    @property
    def missing_markers(self) -> tuple[str, ...]:
        """Marker roles with no channel in this layout -- i.e. not measured."""
        return tuple(m for m in MARKER_ROLES if m not in self.roles)

    @property
    def target_channels(self) -> tuple[int, ...]:
        """Indices ``analyze_image`` scores, ascending: the marker channels."""
        return tuple(sorted(self.roles.index(m) for m in self.markers))

    @property
    def unused_channels(self) -> tuple[int, ...]:
        return tuple(i for i, role in enumerate(self.roles) if role == UNUSED)

    @property
    def names(self) -> tuple[str, ...]:
        """Per-channel display names, unique by construction.

        Unused channels are named ``ch<index>`` rather than all sharing ``-``:
        these names are dictionary keys in ``analyze_image`` and axis titles in
        the montages, both of which need them distinct.
        """
        return tuple(
            f"ch{i + 1}" if role == UNUSED else role
            for i, role in enumerate(self.roles)
        )

    def crop_key(self, index: int) -> str:
        """The notebook's per-channel display-crop key, e.g. ``"dapi_crop"``."""
        return f"{self.names[index].lower()}_crop"

    # -- identity and reporting ---------------------------------------------

    def describe_order(self) -> str:
        return ",".join(self.roles)

    def fingerprint(self) -> str:
        """Cache identity: a different layout scores different pixels."""
        return f"channels={self.describe_order()}"

    def describe(self) -> str:
        """One line for the log, naming what is measured and what is not."""
        parts = [f"Channels: {self.describe_order()}"]
        missing = self.missing_markers
        if missing:
            parts.append("not measured: " + ", ".join(set_label(m) for m in missing))
        unused = self.unused_channels
        if unused:
            parts.append(
                "ignored: channel " + ", ".join(str(i + 1) for i in unused))
        return "  |  ".join(parts)


#: The layout every pre-existing caller gets, and what the Fiji dialog starts from.
DEFAULT_LAYOUT = ChannelLayout.default()


@dataclass(frozen=True)
class MarkerAxes:
    """The marker sets one run reports on, and the cases they combine into.

    This is the presentation side of a :class:`ChannelLayout`: the UpSet plot,
    the per-case montages and the quantification all enumerate the same cases,
    and all three have to shrink the same way when a marker was not measured.
    Deriving them here rather than in each of the three is what keeps a
    two-marker run's figures consistent with its numbers.

    With all three markers the enumeration reproduces the published bar order
    exactly: none, OPC, Transduced, Neuron, then the three pairs in the same
    sequence, then all three.
    """

    markers: tuple[str, ...]

    @classmethod
    def from_layout(cls, layout: ChannelLayout) -> "MarkerAxes":
        return cls(markers=layout.markers)

    @classmethod
    def full(cls) -> "MarkerAxes":
        return cls(markers=MARKER_ROLES)

    def __bool__(self) -> bool:
        """False when nothing is measured -- a DAPI-only run, which counts
        nuclei and has no cases to plot."""
        return bool(self.markers)

    @property
    def set_names(self) -> tuple[str, ...]:
        """Display names of the active markers, in flag order."""
        return tuple(SET_NAMES[m] for m in self.markers)

    @property
    def display_markers(self) -> tuple[str, ...]:
        """Active markers in :data:`DISPLAY_ORDER` -- the enumeration order."""
        return tuple(m for m in DISPLAY_ORDER if m in self.markers)

    def flags_of(self, flags: dict[str, bool]) -> tuple[bool, ...]:
        """``flags`` (keyed by marker role) reduced to the active markers."""
        return tuple(bool(flags[m]) for m in self.markers)

    def combo_order(self) -> list[tuple[bool, ...]]:
        """Every case, grouped by how many sets it is in, smallest first."""
        display = self.display_markers
        order: list[tuple[bool, ...]] = []
        for size in range(len(display) + 1):
            for chosen in combinations(display, size):
                order.append(tuple(m in chosen for m in self.markers))
        return order

    def label(self, combo: Iterable[bool]) -> str:
        """``"OPC+, Transduced+"``, or ``"none"`` for the all-negative case."""
        chosen = [
            SET_NAMES[marker] + "+"
            for marker, flag in zip(self.markers, combo) if flag
        ]
        if not chosen:
            return "none"
        # Named in enumeration order, so a label reads the same way the bars do.
        rank = {SET_NAMES[m]: i for i, m in enumerate(self.display_markers)}
        return ", ".join(sorted(chosen, key=lambda name: rank[name[:-1]]))

    def slug(self, combo: Iterable[bool]) -> str:
        """Filename-safe form, e.g. ``(True, False, True)`` -> ``"101"``."""
        return "".join("1" if flag else "0" for flag in combo)


#: The three-marker axes -- what a default-layout run reports on.
FULL_AXES = MarkerAxes.full()
