# Appose worker script for NeuralClassifyPlugin's "Show Cell" action.
#
# Inputs (from Java): image_path (str), patch_idx, bb_y0, bb_y1, bb_x0, bb_x1 (int).
# Output: cell_crop, a (5, H, W) uint16 Appose NDArray -- DAPI/OPC/RFP/B3Tub/BF,
# the exact raw pixels the classifiers scored for this cell.
#
# The image is named per request rather than inherited from the last run: a
# batch classifies a whole folder, so "the image" is whichever one the selected
# cell came from. Exactly one decoded image is kept between calls -- clicking
# through cells of one image re-reads nothing, and moving to another image drops
# the previous one instead of accumulating ~840 MB per image in the worker.

import appose

from neural_imgs.inference.fixed_model_pipeline import (
    CHANNEL_NAMES,
    crop_cell,
    load_patches,
)

_grid = PATCH_GRID if "PATCH_GRID" in globals() else 4  # noqa: F821
_key = (image_path, _grid)  # noqa: F821

_have = "PATCHED_IMAGE" in globals() and "PATCHED_IMAGE_KEY" in globals()
if not _have or PATCHED_IMAGE_KEY != _key:  # noqa: F821
    task.update(message=f"Reading {image_path} for cell view...")  # noqa: F821
    PATCHED_IMAGE = load_patches(*_key)  # noqa: F821
    task.export(PATCHED_IMAGE=PATCHED_IMAGE, PATCHED_IMAGE_KEY=_key)  # noqa: F821

crop = crop_cell(
    PATCHED_IMAGE, patch_idx, (bb_y0, bb_y1, bb_x0, bb_x1)  # noqa: F821
).copy()  # contiguous, so it can be written straight into shared memory

nd = appose.NDArray(str(crop.dtype), list(crop.shape))
nd.ndarray()[:] = crop

task.outputs["cell_crop"] = nd  # noqa: F821
task.outputs["channel_names"] = list(CHANNEL_NAMES)  # noqa: F821
