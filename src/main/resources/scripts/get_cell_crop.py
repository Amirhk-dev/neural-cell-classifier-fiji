# Appose worker script for NeuralClassifyPlugin's "Show Cell" action.
#
# Inputs (from Java): image_path (str), patch_idx, bb_y0, bb_y1, bb_x0, bb_x1 (int).
# Output: cell_crop, a (C, H, W) uint16 Appose NDArray in the run's channel order,
# the exact raw pixels the classifiers scored for this cell, plus channel_names
# so the composite's channels can be labelled and coloured.
#
# C and the channel order are whatever the classify run declared, not a fixed
# five: CHANNEL_ORDER is exported by classify_cell_image.py and read back here,
# so a single-cell view can never come back with the planes in a different order
# from the one that was scored. Without that export (a worker that somehow
# served this first) it falls back to the historical DAPI/OPC/RFP/B3Tub/BF.
#
# The image is named per request rather than inherited from the last run: a
# batch classifies a whole folder, so "the image" is whichever one the selected
# cell came from. Exactly one decoded image is kept between calls -- clicking
# through cells of one image re-reads nothing, and moving to another image drops
# the previous one instead of accumulating ~840 MB per image in the worker.

import appose

from neural_imgs.channels import ChannelLayout
from neural_imgs.inference.fixed_model_pipeline import crop_cell, load_patches

_grid = PATCH_GRID if "PATCH_GRID" in globals() else 4  # noqa: F821
_order = CHANNEL_ORDER if "CHANNEL_ORDER" in globals() else None  # noqa: F821
_layout = ChannelLayout.parse(_order)
# The layout is part of the cache key: re-reading the same file under a changed
# channel declaration must produce a differently ordered array, so serving the
# previously decoded one would be wrong.
_key = (image_path, _grid, _layout.describe_order())  # noqa: F821

_have = "PATCHED_IMAGE" in globals() and "PATCHED_IMAGE_KEY" in globals()
if not _have or PATCHED_IMAGE_KEY != _key:  # noqa: F821
    task.update(message=f"Reading {image_path} for cell view...")  # noqa: F821
    PATCHED_IMAGE = load_patches(image_path, _grid, _layout)  # noqa: F821
    task.export(PATCHED_IMAGE=PATCHED_IMAGE, PATCHED_IMAGE_KEY=_key)  # noqa: F821

crop = crop_cell(
    PATCHED_IMAGE, patch_idx, (bb_y0, bb_y1, bb_x0, bb_x1)  # noqa: F821
).copy()  # contiguous, so it can be written straight into shared memory

nd = appose.NDArray(str(crop.dtype), list(crop.shape))
nd.ndarray()[:] = crop

task.outputs["cell_crop"] = nd  # noqa: F821
# The image's own reconciled layout, not the declared one: load_patches padded
# the declaration out to the file's real channel count, so this always has one
# name per plane of the array above.
task.outputs["channel_names"] = list(PATCHED_IMAGE.layout.names)  # noqa: F821
