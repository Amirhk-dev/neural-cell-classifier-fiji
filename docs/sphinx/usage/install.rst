Install
=======

What you need
-------------

* **Fiji** from https://fiji.sc, recent enough to bundle **Java 21 or newer**
  (``Help > About ImageJ...``). Current downloads do; only a long-unused
  ``Fiji.app`` still on Java 8 would not.
* **Internet access on the first run only** — 1–2 GB, once per machine.
* **About 5 GB of free disk** for the cached Python environment and models.
* **A GPU is optional.** CUDA and Apple Silicon (MPS) are detected and used
  automatically; without either it runs on CPU, correctly but slowly
  (:doc:`tuning`).

Steps
-----

#. Download ``Neural_Image_Classifier-<version>.jar`` from the `latest release
   <https://github.com/Amirhk-dev/neural-cell-classifier-fiji/releases/latest>`_.
#. Get it into ``Fiji.app/plugins/``, either way:

   * **From inside Fiji:** ``Plugins > Install...``, pick the ``.jar``, accept
     the default save location.
   * **By hand:** drop it into ``Fiji.app/plugins/`` yourself. On macOS that is
     right-click the Fiji app → *Show Package Contents* →
     ``Contents/Java/Fiji.app/plugins/``.

#. **Restart Fiji.** ``Help > Refresh Menus`` adds the menu entries but does
   **not** reload plugin code.

You should now have::

    Plugins > Neural Image Processor > Classify Cells (OPC/B3Tub/RFP)...
                                    > Show Cell (all channels)...
                                    > Install / Verify Models...

That is the whole install: no model bundle to request, no configuration file to
edit.

.. _first-run:

What the first run downloads
----------------------------

The first time you classify anything, two one-time downloads happen, each with
its own progress window.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - What
     - Size
     - Detail
   * - Python environment
     - ~1–2 GB
     - An isolated conda environment (Python 3.11 + PyTorch + CellPose +
       BaSiCPy) built by `Appose <https://github.com/apposed/appose>`_. Size
       depends on platform — a CUDA-capable machine pulls the larger GPU
       wheels. It does **not** touch any Python you already have.
   * - Model bundle
     - ~90 MB
     - The two trained classifiers and their decision thresholds, into
       ``~/.neural-imgs/models/``.

Expect several minutes. Every later run finds both cached and starts in
seconds.

To get that out of the way before a session — or just to find out whether the
models are already there — run ``Plugins > Neural Image Processor > Install /
Verify Models...``. It reports what is installed and downloads only what is
missing.

.. note::

   **Behind a restrictive proxy?** If ``github.com`` release downloads are
   blocked, an admin can host ``models.zip`` internally and start Fiji with
   ``-Dneuralimgs.models.url=https://internal.host/models.zip``. The integrity
   check is not relaxed for a mirror: the file must be the same bytes.
