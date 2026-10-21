The dialog, field by field
==========================

``Plugins > Neural Image Processor > Classify Cells (OPC/B3Tub/RFP)...``

Only the first group is required. Everything below it is tuning, already set to
the published defaults — except **Channels**, which describes *your* images and
is the one group a new dataset usually needs changed.

.. code-block:: text

    ┌─ Neural Image Classifier (1.2.0) ──────────────────────────────┐
    │ Input and output                                               │
    │           Input: [Single image  ▾]                             │
    │       CZI image: [____________________________] [Browse]       │
    │                  One Zeiss .czi file.                          │
    │    Input folder: [____________________________] [Browse]       │
    │                  Every .czi directly inside it; not recursive. │
    │ Model directory: [____________________________] [Browse]       │
    │ Output directory:[____________________________] [Browse]       │
    │                  Downloaded on first run. ...                  │
    │                                                                │
    │ Channels                                                       │
    │   Channel number of each marker, counting from 1.              │
    │   0 = not in my images; DAPI is required.                      │
    │   DAPI [1]  OPC [2]  RFP [3]  B3-Tub [4]  BF [5]               │
    │                                                                │
    │ Detection                                                      │
    │   Patch grid (NxN) [4]  Cell probability [0.10]  Flow [0.20]   │
    │                                                                │
    │ RFP positivity (an intensity rule, not a classifier)           │
    │   Rule [neighbour ▾]     Exclusion radius [60] px              │
    │   Gate, neighbour [16.50]   Gate, legacy [5.00]                │
    │                                                                │
    │ Performance                                                    │
    │   Images prepared ahead [1]  Extraction threads [0] 0 = auto   │
    │                                                                │
    │ Outputs                                                        │
    │   [x] UpSet plot          [ ] Example cell per UpSet case       │
    │   [ ] Cell ROIs in ROI Mgr[x] Load saved results if available   │
    │                                                                │
    │                               [Help]  [Cancel]  [OK]           │
    └────────────────────────────────────────────────────────────────┘

The **Help** button opens the same guidance inside Fiji, with no browser and no
network.

Input and output
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - What it is
   * - **Input**
     - ``Single image`` or ``Folder of images``. The row the selected mode does
       not use is **greyed out** — field, ``Browse`` button, caption and hint —
       so only the path that will actually be read is editable. The greyed
       field keeps its text, and both persist between runs, so switching mode
       never means retyping a path.
   * - **CZI image**
     - One Zeiss ``.czi``. Which channel carries which marker is set in
       **Channels**; see :doc:`channels`.
   * - **Input folder**
     - Every ``.czi`` *directly* inside it is classified. Not recursive, hidden
       files skipped, sorted by name.
   * - **Model directory**
     - Pre-filled with ``~/.neural-imgs/models``, downloaded on first use.
       Change it only to use your own retrained checkpoints; a complete
       directory is never overwritten. See :doc:`models`.
   * - **Output directory**
     - The output **root**. One subfolder per image is created inside it, named
       after the image. See :doc:`outputs`.

Channels
--------

Five fields on one row, one per marker, each holding that marker's channel
number in your image counting from 1 — ``DAPI 1, OPC 2, RFP 3, B3-Tub 4,
BF 5`` by default, the acquisition the models were trained on.

``0`` means *not in my images*: the run goes ahead and that marker is reported
as **not measured**. ``DAPI`` cannot be ``0``. :doc:`channels` has the whole
story, because it is the one setting that changes what the results mean.

Detection
---------

.. list-table::
   :header-rows: 1
   :widths: 26 12 20 42

   * - Field
     - Default
     - Range
     - What it asks
   * - **Patch grid (NxN)**
     - ``4``
     - ≥ 1, whole
     - Tiles per side, so ``4`` means 16 tiles. A *count*, not a pixel size.
   * - **Cell probability**
     - ``0.1``
     - −3.0 … 2.0
     - Does this pixel look like it belongs to a cell? Lower → more cells
       detected.
   * - **Flow**
     - ``0.2``
     - 0.0 … 1.0
     - Are the proposed mask's flows consistent with CellPose's prediction?
       Lower → fewer masks accepted.

Both thresholds accept any decimal in range, not just the tenth-steps the usual
tables list; out-of-range or unparseable entries are rejected with a message
rather than silently clamped. See :doc:`tuning`.

RFP positivity
--------------

``Rule`` (default ``neighbour``), ``Exclusion radius`` (default ``60`` px) and
one gate per rule — ``16.5`` for ``neighbour``, ``5`` for ``legacy``. The gates
are separate fields because the two rules are on **different scales**; see
:doc:`rfp`.

Performance
-----------

``Images prepared ahead`` (default ``1``) is the memory knob and
``Extraction threads`` (default ``0`` = auto) is the CPU knob. See
:ref:`what-runs-in-parallel`.

Outputs
-------

Four checkboxes: ``UpSet plot`` (on), ``Example cell per UpSet case`` (off),
``Cell ROIs in the ROI Manager`` (off), ``Load saved results if available``
(on). :doc:`outputs` says what each produces.

After the run
-------------

Click **OK**. When the run finishes **the dialog comes back**, pre-filled with
what you just used, so the next image is one field-change away; **Cancel**
finishes. The dialog is non-modal, so the Log window and Results table stay
readable while it is up. A failed run returns to the dialog too, rather than
ending the session.

.. _show-cell:

Looking at one cell
-------------------

``Plugins > Neural Image Processor > Show Cell (all channels)...`` opens a
composite of every channel your run declared, for one cell — the exact raw
pixels the classifiers scored, in the same channel order that scored them.

Select its ROI in the ROI Manager first to pre-fill the id, or just type one.
Toggle channels with the ``C`` slider (``Image > Color > Channels Tool``) like
any Fiji composite. It needs a ``Classify Cells...`` run earlier in the same
session and does not re-run the pipeline. After a folder run it also asks
**which image**, since cell ids restart per image.
