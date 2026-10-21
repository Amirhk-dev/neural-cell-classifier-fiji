Tiling, detection and performance
=================================

Patch grid
----------

**Patch grid (NxN)** is a *tile count per side*, not a pixel size: ``4`` splits
the image into a 4×4 = 16-tile grid before running BaSiC and CellPose per tile.
A whole number of at least 1; a fractional entry is rejected rather than
quietly truncated.

Tiling is what keeps one tile's pixels, its BaSiC correction and its CellPose
call inside memory, and cells are numbered tile by tile. It mirrors how the
training data was built, so changing it changes **results**, not just speed.

The two CellPose thresholds
---------------------------

They decide which nuclei are detected on the DAPI channel, and therefore which
cells everything downstream classifies and counts.

.. list-table::
   :header-rows: 1
   :widths: 26 12 18 44

   * - Field
     - Default
     - Range
     - What it asks
   * - Cell probability threshold
     - ``0.1``
     - −3.0 … 2.0
     - Does this pixel look like it belongs to a cell? **Lower → more cells
       detected.**
   * - Flow threshold
     - ``0.2``
     - 0.0 … 1.0
     - Are the proposed mask's flows consistent with what CellPose predicted?
       **Lower → fewer masks accepted (stricter).**

Both accept any decimal in range (``0.15``, ``-1.25``, …), not just the
tenth-steps the usual tables list; out-of-range or unparseable entries are
rejected with a message rather than silently clamped. Cell diameter is not
passed — CellPose auto-estimates it.

Because these two decide which nuclei exist at all, they are part of the
saved-results key: changing either makes a previous run stale and the image is
re-classified (:ref:`saved-results`).

CPU or GPU
----------

Device selection is automatic: CUDA if present, else Apple Silicon MPS, else
CPU.

On an Intel Mac (no MPS) that means CPU-only and is noticeably slow — CellPose
4's model (Cellpose-SAM) has a much heavier backbone than classic CellPose, so
BaSiC + CellPose + two ResNet-18 passes per cell on CPU alone can be tens of
minutes *per tile*. Try ``Patch grid = 2`` first to sanity-check.

On Apple Silicon, MPS should be dramatically faster; if it still crawls,
confirm in the Log that CellPose picked MPS — it prints ``Using device: ...``
on load — rather than silently falling back.

.. _what-runs-in-parallel:

What runs in parallel
---------------------

**Inside one image the tiles are pipelined, not serial.** CellPose runs on one
tile at a time — one GPU, one model copy, so a second concurrent call would
contend rather than add throughput — while that tile's cell extraction and RFP
scoring run on a CPU thread pool, overlapping the next tile's CellPose call.
**Extraction threads** sizes that pool; ``0`` sizes it from this process's core
allocation.

**Across images in folder mode**, reading the next CZI and fitting its BaSiC
illumination model overlap the current image's GPU work. **Images prepared
ahead** bounds how many may wait: each one holds its whole tile array (~840 MB
for a 16-tile OPC CZI), so this is the run's memory knob. ``1`` is enough to
keep the GPU lane fed.

Cell ids are assembled back in tile order, so the numbers are identical to a
fully serial run.

See :mod:`neural_imgs.inference.batch` and
:mod:`neural_imgs.inference.patch_extraction`.
