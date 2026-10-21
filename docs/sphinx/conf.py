"""Sphinx configuration for the Neural Cell Classifier manual.

Two things are documented here from one source tree:

* the **plugin manual** -- hand-written pages under ``usage/``, which is where
  the long-form answers live so that neither the Fiji dialog nor the README has
  to carry them;
* the **Python API** -- pulled straight out of ``python/src/neural_imgs`` by
  autodoc, because those modules are a byte-identical copy of the research
  repository's inference subset (see ``docs/DEVELOPING.md``) and must not be
  edited to suit a docs build.

That second point is why everything heavy is mocked below rather than
installed: a docs build must never need PyTorch, CellPose or BaSiCPy, and it
must never be a reason to touch a vendored module.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# -- Where the code is -------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "python" / "src"))

# -- Project ----------------------------------------------------------------

project = "Neural Cell Classifier"
author = "Amirhossein Kardoost"
copyright = f"{date.today().year}, {author}"

# The plugin build id, which is what a user is asked to quote in a bug report.
# Read out of the Java source so there is no third place to bump it: the pom
# names the jar, BUILD_ID is what the running plugin claims to be, and this is
# what the manual claims to describe.
_JAVA = _REPO / "src/main/java/com/neuralimgs/fiji/NeuralClassifyPlugin.java"
_marker = 'static final String BUILD_ID = "'
release = next(
    (
        line.split(_marker, 1)[1].split('"', 1)[0]
        for line in _JAVA.read_text(encoding="utf-8").splitlines()
        if _marker in line
    ),
    "unknown",
)
version = release

# -- Extensions -------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autodoc ----------------------------------------------------------------

# The docstrings in neural_imgs are already reStructuredText with Sphinx roles
# (:mod:, :class:, :attr:, literal blocks, "#:" attribute comments), so autodoc
# needs no preprocessing -- only the imports mocked out.
autodoc_mock_imports = [
    "basicpy",
    "cellpose",
    "czifile",
    "matplotlib",
    "pandas",
    "scipy",
    "skimage",
    "tifffile",
    "torch",
    "torchvision",
    "tqdm",
    "upsetplot",
    "yaml",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
# Dataclass defaults are part of the contract here -- RfpConfig's gate, the
# CellPose thresholds, CellExtractionConfig's geometry -- so the signature is
# worth showing in full, and showing it verbatim rather than re-evaluated.
autodoc_typehints = "signature"
autodoc_preserve_defaults = True
autosummary_generate = False

napoleon_google_docstring = True
napoleon_numpy_docstring = True
# OFF deliberately, and it matters. Several of these modules document related
# knobs on one line -- "pad / min_area / boundary_margin / target:" in
# CellExtractionConfig, and the same style in rfp and positivity. With
# use_param on, napoleon emits ":param pad / min_area / boundary_margin /
# target:", and Sphinx's field parser reads a multi-word param argument as
# "<type> <name>" -- so that line renders as the parameter *target* of type
# "pad / min_area / boundary_margin /". Off, napoleon emits a definition list
# and the names survive as written. Grouping them is the authors' style in a
# vendored tree, so the renderer is what has to give.
napoleon_use_param = False

# neural_imgs.positivity.neighbour_contrast writes "|crowding| < 0.05" meaning
# the absolute value, which reStructuredText reads as a substitution reference.
# The fix belongs here rather than in the module: those files are a
# byte-identical copy of the research repository (see docs/DEVELOPING.md), and
# a docs build is not a reason to break that. Defining the substitution renders
# exactly what was meant.
rst_prolog = r"""
.. |crowding| replace:: \|crowding\|
"""

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

_GH = "https://github.com/Amirhk-dev/neural-cell-classifier-fiji"
extlinks = {
    "repofile": (_GH + "/blob/main/%s", "%s"),
    "issue": (_GH + "/issues/%s", "issue #%s"),
}

# -- HTML -------------------------------------------------------------------

# furo if it is installed, otherwise a theme that ships with Sphinx, so a build
# never fails on a missing theme.
try:
    import furo  # noqa: F401

    html_theme = "furo"
    html_theme_options = {
        "source_repository": _GH,
        "source_branch": "main",
        "source_directory": "docs/sphinx/",
        "footer_icons": [
            {
                "name": "GitHub",
                "url": _GH,
                "html": "GitHub",
                "class": "",
            }
        ],
    }
except ImportError:  # pragma: no cover - cosmetic fallback
    html_theme = "alabaster"
    html_theme_options = {"description": "A Fiji plugin for single-cell marker calls"}

html_title = f"{project} {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = True


# -- Two renderer fixes for docstrings that must not be edited ---------------
#
# The modules under python/src/neural_imgs are a byte-identical copy of the
# research repository's inference subset, so anything that renders badly has to
# be fixed here. See docs/DEVELOPING.md for why that copy is byte-identical.

_ATTRIBUTE_LIKE = frozenset({"attribute", "data", "property"})

# Original lines of the attribute-like docstring currently being processed,
# keyed by (what, name). Written before napoleon runs, read back after.
_verbatim_attribute_docstrings: dict[tuple[str, str], list[str]] = {}


def _keep_attribute_docstring(app, what, name, obj, options, lines):
    """Remember an attribute's docstring before napoleon rewrites it."""
    if what in _ATTRIBUTE_LIKE:
        _verbatim_attribute_docstrings[(what, name)] = lines[:]


def _restore_attribute_docstring(app, what, name, obj, options, lines):
    """Put it back, i.e. skip napoleon for attributes and module data.

    Napoleon parses an attribute docstring as ``type: description`` and splits
    it on the first colon, with no option to turn that off. These modules use
    ``#:`` comments that are ordinary prose, and prose contains colons, so

        #: The nucleus channel. Required: CellPose detects on it and every
        #: cell's mask, crop centre and bounding box come from that detection.

    rendered with "The nucleus channel. Required" as the *type* of ``DAPI`` and
    the rest as its description -- losing the first half of the sentence and
    inventing a type. The sections napoleon is here for (``Parameters`` and
    friends) only ever appear in function and class docstrings, so bypassing it
    for attribute-like objects costs nothing.
    """
    original = _verbatim_attribute_docstrings.pop((what, name), None)
    if original is not None:
        lines[:] = original


# Targets these docstrings reference that nitpick mode cannot resolve, so that
# `make strict` reports real problems only:
nitpick_ignore_regex = [
    # Third-party types, mocked above: no inventory to resolve them against,
    # and intersphinx for each would make a docs build need the network.
    (r"py:.*", r"(torch|pandas|cellpose|basicpy|skimage|scipy|matplotlib)\..*"),
    # Private and dunder names, referenced from the module that owns them.
    (r"py:.*", r"_[A-Za-z_]+"),
    (r"py:.*", r"__\w+__"),
    # Names that live in the private research repository's training half, which
    # is not part of this distribution (see docs/DEVELOPING.md).
    (r"py:.*", r"(CellClassifierDataset|LocalContrastHeuristic)"),
    # Short names written unqualified in prose; the qualified target exists.
    (r"py:.*", r"(ClassifyImageOutput|ChannelLayout\.for_image)"),
    # Half of a generic. Sphinx cross-references each comma-separated piece of
    # a union annotation separately, so "int | tuple[int, int]" also asks for a
    # class called "tuple[int". It renders correctly; only the lookup is bogus.
    (r"py:.*", r"[^\]]*\[[^\]]*"),
]


def setup(app):
    # Napoleon connects at the default priority of 500, so 100 runs before it
    # and 900 after.
    app.connect("autodoc-process-docstring", _keep_attribute_docstring, priority=100)
    app.connect("autodoc-process-docstring", _restore_attribute_docstring, priority=900)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
