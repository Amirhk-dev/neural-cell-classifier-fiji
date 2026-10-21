Channels and optional markers
=============================

The **Channels** group is one field per marker, holding that marker's channel
number in your image counting from 1, and ``0`` for a channel your images do
not have.

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - Your images
     - Channels fields
     - What runs
   * - DAPI, OPC, RFP, B3Tub, BF (the default)
     - ``1, 2, 3, 4, 5``
     - everything
   * - DAPI, RFP, B3Tub — **no OPC**
     - ``1, 0, 2, 3, 0``
     - Neuron + Transduced calls, conversion ratio, transduction performance
   * - DAPI, OPC, B3Tub — no RFP
     - ``1, 2, 0, 3, 0``
     - OPC + Neuron calls and counts; no ratios (all three are fractions of the
       transduced population)
   * - B3Tub, RFP, DAPI (reordered)
     - ``3, 0, 2, 1, 0``
     - Neuron + Transduced, same as row 2
   * - DAPI only
     - ``1, 0, 0, 0, 0``
     - nucleus detection and a cell count

``0`` is "not in my images", not "skip this classifier"
-------------------------------------------------------

A marker set to ``0`` is reported as **not measured**, which is deliberately
different from *measured and found negative*:

* **no column** for it in the Results table or the Fiji CSV (the per-image CSV
  keeps the column for a stable header, but leaves it **blank** rather than
  ``False``);
* **no set** for it in the UpSet plot — three markers give the published 8
  cases, two give 4, one gives 2, none skips the plot entirely;
* **no ratio** that needs it: with no OPC channel the OPC state ratio is
  omitted from the quantification rather than printed as ``0/402 = 0.0%``, and a
  closing ``Not measured in this run: ...`` line names what was left out;
* **no per-case montage** panel claiming a probability it never computed.

*"There was no OPC channel"* and *"no cell was OPC+"* are different findings,
and the outputs keep them apart.

Everything the missing channel is not needed for is unchanged: tiling, BaSiC,
CellPose detection, cell extraction, cell ids, ROIs, and the remaining markers'
classifications are bit-for-bit what a full run would produce for them.

Only DAPI is required
---------------------

Nuclei are detected on DAPI, and every single cell in the run — its mask, its
crop, its bounding box, its id — comes from that detection, so there is nothing
to classify without it. The dialog rejects ``DAPI 0`` with that explanation.

What else the fields allow
--------------------------

* The numbers **need not be in order** (``DAPI 3`` is fine) or contiguous — an
  undeclared channel below your highest one is read, illumination-corrected and
  then ignored.
* Channels **above** the highest number you use need not be mentioned at all,
  so one assignment can fit both a 3-channel and a 5-channel image.
* **Two markers on the same channel** is rejected before the run starts.
* An image with **fewer** channels than you assigned is a real mismatch: it is
  reported per image, naming both counts, *before* that image's multi-hour work
  begins — and in a folder run the other images still go ahead.

.. warning::

   A channel set to ``0`` is not a classifier being switched off — the channel
   is simply not there to score. If a channel **is** present but your model
   directory ships no checkpoint for it, that marker is *also* reported as not
   measured, and the Log says which when the models load. See :doc:`models`.

The channel assignment is part of the saved-results key
(:ref:`saved-results`), because a run that measured two markers and a run that
measured three produce different tables from the same image. Changing it
re-classifies, and the Log says ``channel assignment changed (... -> ...)``.

Under the hood
--------------

The Java side collects the five numbers and turns them into one canonical
``channel_order`` string — e.g. ``DAPI,-,RFP,B3Tub`` — which
:mod:`neural_imgs.channels` parses and validates. Parsing lives in one place on
purpose: there is no second spelling table to drift.

See :class:`neural_imgs.channels.ChannelLayout` and
:class:`neural_imgs.channels.MarkerAxes`.
