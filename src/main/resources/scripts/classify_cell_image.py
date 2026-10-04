# Appose worker script for NeuralClassifyPlugin.
#
# Classifies ONE image or a whole FOLDER of them. Either way the work goes
# through neural_imgs.inference.batch.BatchClassifier, so a single image is just
# a batch of one and there is no second code path to keep in step.
#
# Appose injects one global variable per key in the Java-side `inputs` map
# passed to `python.task(script, inputs)`:
#   image_path         : str   -- CZI to classify; "" when input_dir is used
#   input_dir          : str   -- folder of CZIs to classify; "" when image_path is used
#   patch_grid         : int   -- NxN tiling grid (patch_image()'s patch_size arg)
#   flow_threshold     : float -- CellPose flow threshold (nucleus detection)
#   cellprob_threshold : float -- CellPose cell probability threshold
#   rfp_method              : str   -- "neighbour" (production) or "legacy"
#   rfp_neighbour_threshold : float -- gate on the neighbour rule's score
#   rfp_legacy_threshold    : float -- gate on the legacy rule's score. The two
#                                 rules are NOT on the same scale (different
#                                 background estimator), so they carry their own
#                                 gates rather than sharing one field that would
#                                 silently mean something else after a switch.
#   rfp_exclude_radius      : int   -- native px around other nuclei dropped from
#                                 the background pool (neighbour rule only)
#   model_dir          : str   -- per-marker checkpoints + thresholds.json
#   out_dir            : str   -- OUTPUT ROOT. One subfolder per image is created
#                                 inside it, named after the image, holding that
#                                 image's cache, CSV, quantification and figures.
#   generate_upset     : bool  -- build + save the UpSet co-positivity plot
#   show_case_montage  : bool  -- also save one example cell per UpSet case
#   reuse_cache        : bool  -- load previously saved results instead of
#                                 re-running the (multi-hour) classification
#   prefetch           : int   -- images allowed to be read + BaSiC-fitted ahead
#                                 of the GPU lane; the batch's memory knob
#   n_workers          : int   -- CPU threads for per-patch extraction; 0 = auto
#
# This worker process is persistent across task() calls within one Appose
# Service, but each task() call gets a FRESH exec() binding -- only names
# explicitly passed to `task.export(...)` survive into the next task's globals().
# MODELS is exported because loading it costs minutes; see get_cell_crop.py for
# how single-cell views are served afterwards.

from pathlib import Path

from neural_imgs.inference.batch import (
    BatchClassifier,
    BatchInput,
    discover_images,
)
from neural_imgs.inference.fixed_model_pipeline import load_models
from neural_imgs.inference.rfp import RfpConfig


def _report(message, current=None, maximum=None):
    kwargs = {"message": message}
    if current is not None:
        kwargs["current"] = current
    if maximum is not None:
        kwargs["maximum"] = maximum
    task.update(**kwargs)  # noqa: F821


# -- what to run on -----------------------------------------------------------

if input_dir:  # noqa: F821
    _paths = [str(p) for p in discover_images(input_dir)]  # noqa: F821
    if not _paths:
        raise RuntimeError(f"No .czi images directly inside {input_dir}")
    _report(f"Found {len(_paths)} images in {input_dir}")
else:
    _paths = [image_path]  # noqa: F821

_rfp = RfpConfig(
    method=rfp_method,  # noqa: F821
    neighbour_threshold=float(rfp_neighbour_threshold),  # noqa: F821
    legacy_threshold=float(rfp_legacy_threshold),  # noqa: F821
    exclude_radius=int(rfp_exclude_radius),  # noqa: F821
)
_report(_rfp.describe())

# Loaded once per worker process and reused for every image in every later run
# -- the whole reason the batch is one process rather than one per image.
if "MODELS" not in globals() or MODELS_DIR != model_dir:  # noqa: F821
    MODELS = load_models(model_dir, on_progress=_report)  # noqa: F821
    MODELS_DIR = model_dir  # noqa: F821
    task.export(MODELS=MODELS, MODELS_DIR=MODELS_DIR)  # noqa: F821

batch_input = BatchInput(
    image_paths=_paths,
    out_root=out_dir,  # noqa: F821
    model_dir=model_dir,  # noqa: F821
    patch_grid=patch_grid,  # noqa: F821
    flow_threshold=flow_threshold,  # noqa: F821
    cellprob_threshold=cellprob_threshold,  # noqa: F821
    rfp=_rfp,
    reuse_cache=reuse_cache,  # noqa: F821
    generate_upset=generate_upset,  # noqa: F821
    show_case_montage=show_case_montage,  # noqa: F821
    n_workers=int(n_workers) or None,  # noqa: F821
    prefetch=max(1, int(prefetch)),  # noqa: F821
)

result = BatchClassifier(
    MODELS,  # noqa: F821
    on_progress=_report,
    # Reported as each image lands rather than at the end: a folder run is
    # hours, and a user watching the Log should see image 3 finish while image 4
    # is still going.
    on_image_done=lambda outcome: _report(outcome.summary_line()),
).run(batch_input)

# -- outputs ------------------------------------------------------------------
#
# Every value is coerced to a plain Python scalar. Appose serialises task outputs
# as JSON and has no encoding for a numpy scalar -- it passes one across as an
# opaque object handle, which the Java side then fails to cast to a number.

task.outputs["batch_summary"] = result.summary_lines()  # noqa: F821
task.outputs["n_images"] = len(result.outcomes)  # noqa: F821
task.outputs["n_failed"] = result.n_failed  # noqa: F821

task.outputs["images"] = [  # noqa: F821
    {
        "image_path": o.image_path,
        "image_name": Path(o.image_path).name,
        "out_dir": o.out_dir,
        "status": o.status,
        "detail": o.detail,
        "n_cells": int(o.n_cells),
        "n_opc_pos": int(o.n_opc_pos),
        "n_b3tub_pos": int(o.n_b3tub_pos),
        "n_rfp_pos": int(o.n_rfp_pos),
        "elapsed_s": float(o.elapsed_s),
        "csv_path": o.csv_path,
        "cache_path": o.cache_path,
        "quantification": list(o.quantification),
        "quantification_path": o.quantification_path,
        "upset_plot_path": o.upset_plot_path,
        "case_montage_paths": list(o.case_montage_paths),
        "warnings": list(o.warnings),
    }
    for o in result.outcomes
]

# One flat cell list across the whole batch, each row tagged with the image it
# came from -- that tag is what lets the Fiji table stay a single sortable table
# and what lets "Show Cell..." know which CZI to re-open for a given cell.
task.outputs["cells"] = [  # noqa: F821
    {
        "image_path": path,
        "image_name": Path(path).name,
        "cell_id": int(c.cell_id),
        "patch_idx": int(c.patch_idx),
        "bb_y0": int(c.bb_position[0]), "bb_y1": int(c.bb_position[1]),
        "bb_x0": int(c.bb_position[2]), "bb_x1": int(c.bb_position[3]),
        "opc_prob": float(c.opc_prob), "opc_pos": bool(c.opc_pos),
        "b3tub_prob": float(c.b3tub_prob), "b3tub_pos": bool(c.b3tub_pos),
        "rfp_pos": bool(c.rfp_pos),
        "rfp_method": str(c.rfp_method),
        # float(), not the raw value: a numpy scalar crosses Appose as an opaque
        # WorkerObject and fails the Java Number cast.
        "rfp_neighbour_score": float(c.rfp_neighbour_score),
        "rfp_neighbour_pos": bool(c.rfp_neighbour_pos),
        "rfp_legacy_pos": bool(c.rfp_legacy_pos),
        "rfp_score": float(c.rfp_score),
        "opc_channel_score": float(c.opc_channel_score),
        "b3tub_channel_score": float(c.b3tub_channel_score),
        "n_cells_in_region": int(c.n_cells_in_region),
    }
    for path, output in result.cells_by_image.items()
    for c in output.cells
]

# Lets get_cell_crop.py re-open whichever image a requested cell belongs to.
task.export(PATCH_GRID=patch_grid)  # noqa: F821
