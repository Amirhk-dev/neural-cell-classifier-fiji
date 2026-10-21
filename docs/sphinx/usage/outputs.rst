What a run produces
===================

On screen
---------

* **One combined Results table** across every image processed, with an
  ``image`` column first (``cell_id`` restarts at 0 per image, so the two
  together identify a cell), plus ``opc_prob``, ``b3tub_prob``, ``rfp_pos``,
  both RFP scores, the bounding box and ``n_cells_in_region``. A marker you set
  to ``0`` gets **no columns at all** rather than a column of zeros
  (:doc:`channels`).
* The **UpSet plot** opens as an image window for a single-image run. For a
  folder run the paths are logged instead — twenty images would otherwise open
  sixty windows at the end of an overnight run.
* **Colour-coded ROIs** in the ROI Manager, if ticked: cyan = OPC+,
  yellow = B3-Tub+, magenta = both, grey = neither.

On disk
-------

Folder mode treats **Output directory** as a *root* and gives every image its
own subfolder, named after the image::

    <output directory>/
    ├── all_images_classifications.csv          ← the combined table, matching Fiji's
    ├── 7d_OPC_reprogramming_20x_#1_.../
    │   ├── ..._fixed_model_classifications.csv
    │   ├── ..._quantification.txt
    │   ├── ..._upset_plot.png
    │   ├── ..._upset_case_<combo>.png          ← if "Example cell per case"
    │   ├── ..._classification_cache.json       ← saved results
    │   └── ..._classification_masks.npz
    └── 7d_OPC_reprogramming_20x_#2_.../
        └── ...

That is not cosmetic: saved results are looked up by **image stem**, so two
images written flat into one directory would overwrite each other's cache and
CSV. One folder per image is also what makes a batch re-runnable image by image
— re-point the plugin at the same folder and everything already done comes back
from cache in seconds, while only the new images are classified.

The three **reprogramming ratios** (OPC state ratio, transduction performance,
conversion ratio) are printed in the Log and saved as
``<image-name>_quantification.txt``.

The UpSet plot
--------------

Sets are named **OPC**, **Neuron** (B3-Tub+) and **Transduced** (RFP+) — one
set per marker you measured — with a fixed bar order so plots are directly
comparable across images. Unticking it skips the plot only; classification,
ROIs and CSV are unaffected.

One example cell per UpSet case
-------------------------------

Off by default. For each case the plugin picks one representative cell and
renders ``DAPI + mask`` followed by every channel of the run, with the DAPI mask
outlined in green on every panel, titled with the case, the cell id and that
cell's scores (``OPC 0.87 / B3 0.12 / RFP 1``, naming only the markers you
measured). Three markers gives the published 8 figures; two gives 4.

``<combo>`` in the filename is the ``OPC/Neuron/Transduced`` flags as ``0``/``1``
— e.g. ``101`` = OPC+, Transduced+. Cases with no cell in this image are
skipped and named in the Log, alongside how many candidates each case had.

The panels are cut from the same BaSiC-corrected, percentile-normalised tile the
cells were detected on, so they are the pixels that were classified rather than
a lookalike. The green outline is drawn whenever the cell's mask is available —
always for a fresh run, and for a cached run whenever the
``_classification_masks.npz`` sidecar sits next to the cache file.

.. note::

   On a **cached** run this step has to read the CZI and re-fit BaSiC, since a
   cache hit skips both — a few minutes, against the hours a re-classification
   costs, and both are then reused for the rest of the Fiji session. A fresh run
   reuses the fit it already made, so the montages cost it almost nothing.

The Log
-------

``Window > Log`` carries one block per image: its counts, the three ratios,
where its folder is, whether the numbers were computed or re-loaded, and any
non-fatal problem. **One image failing does not stop the batch** — it is logged
as ``FAILED`` and the rest still run.

.. _saved-results:

Saved results
-------------

Classifying one CZI takes hours, but the answer is fully determined by its
inputs, so each run writes it into the output directory you chose:

* ``<image-name>_classification_cache.json`` — every cell's id, bounding box,
  probabilities and calls, plus the key described below (~200 bytes per cell);
* ``<image-name>_classification_masks.npz`` — the per-cell DAPI masks,
  bit-packed and compressed. Only the per-case montages' outlines need these;
  if the file is missing, the montages simply render without the green contour.

With **Load saved results if available** ticked, a later run looks for the JSON
*in the output directory you give it* — so pointing at the same output
directory is what makes an image "already done". It then rebuilds ROIs, Results
table, CSV, UpSet plot, quantification and (if asked) the per-case montages
from it in seconds; only the classification itself is skipped. A cache hit never
even reads the CZI.

What the key covers
^^^^^^^^^^^^^^^^^^^

Saved results are reused **only** when they provably describe the same
computation. The file stores a key of:

* the image file's **size in bytes**,
* the patch grid,
* both CellPose thresholds,
* the RFP rule and its gate,
* **the channel assignment**,
* a fingerprint of the model bundle — the contents of ``thresholds.json`` plus
  each checkpoint's byte size, deliberately not mtimes, so re-copying an
  identical bundle does not force a needless re-run.

If any of those differs the file is ignored and the image is re-classified, and
the Log prints the reason, e.g. ``re-classified because saved results do not
match: patch grid 4 -> 8``. A corrupt or older-format file reads as unreadable
and also just triggers a re-run; it never breaks anything.

**Moving the CZI to another folder does not trigger a re-run.** Where a file
sits is not an input to the classification, so a relocated image still matches;
the Log just notes ``image moved from <old path>``. What identifies the image is
its *filename* (which is how the cache file is found) plus its byte size. The
one case this cannot distinguish is two genuinely different CZIs that share both
a filename and an exact byte size and are classified into the same output
directory — untick the checkbox if that is ever a real risk.

Untick to force a fresh run regardless, and delete the two files to throw the
saved results away — both are safe to delete at any time.

See :mod:`neural_imgs.inference.result_cache`.
