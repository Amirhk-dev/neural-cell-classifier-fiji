The RFP rules
=============

RFP is the one marker with no trained classifier — it is called from pixel
intensity. The production default is **neighbour-excluded, radius 60 px,
gate 16.5**.

.. list-table::
   :header-rows: 1
   :widths: 24 52 24

   * - Rule
     - What the cell is compared against
     - Default gate
   * - ``neighbour`` (production)
     - its crop's background **with pixels near other nuclei dropped**
     - ``16.5``
   * - ``legacy``
     - its crop's 95th percentile, neighbours included
     - ``5``

Why the legacy rule was replaced
--------------------------------

Its flaw is measured, not theoretical: one bright neighbour owns the crop's
percentile, so a genuinely transduced cell scores negative. On the control
image 112 cells sit below ``-10`` for that reason, and the score correlates
with crowding at ``-0.47``. Excluding neighbour pixels takes that to ``-0.004``
and recovers ~110 cells across five images, while conversion rate — the
biologists' anchor — stays inside its 67–82% band.

Both rules travel with every cell
---------------------------------

Both are scored for every cell and both appear in the results table
(``rfp_neighbour_score`` / ``rfp_neighbour_pos``, ``rfp_score`` /
``rfp_legacy_pos``), with ``rfp_method`` recording which made the ``rfp_pos``
call. So a saved table can be re-read under the other rule without
re-classifying anything.

.. warning::

   **The two gates are not interchangeable.** Different background estimators
   mean different scales: the neighbour score sits higher, so running it at the
   legacy gate of ``5`` admits 100–250 extra low-confidence cells per image.
   That is why there is one field per rule rather than one shared field.

The rule that is deliberately absent
------------------------------------

A third rule, soft-mask, was built, measured on all five images and
**rejected** — its extra positives convert at the image's *background* B3-Tub+
rate (16–33% against 66–80% for cells every rule agrees on), i.e. they are
mostly false positives. Re-adding it needs new evidence, not a flag.

See :mod:`neural_imgs.inference.rfp` for the fixed operating point, and
:mod:`neural_imgs.positivity.neighbour_contrast` for the rule itself.
