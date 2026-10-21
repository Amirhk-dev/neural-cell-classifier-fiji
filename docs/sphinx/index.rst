Neural Cell Classifier
======================

Single-cell **OPC / Neuron (β-III-tubulin) / Transduced (RFP)** calls on Zeiss
``.czi`` images, from Fiji's ``Plugins`` menu — one image or a whole folder,
without touching Python or a terminal.

Each run reads the CZI, splits it into tiles, corrects illumination (BaSiC),
detects nuclei on DAPI (CellPose), and classifies every detected cell with two
trained ResNet-18 models (OPC, β-III-tubulin) plus an RFP intensity rule.

Two things worth knowing before the first run:

* **You declare which channel is which**, and markers you do not have are
  optional — see :doc:`usage/channels`.
* **Install is one file.** Drop the ``.jar`` into Fiji; the first run fetches
  its own Python environment and the trained models and caches both.

.. toctree::
   :maxdepth: 2
   :caption: Using the plugin

   usage/install
   usage/dialog
   usage/channels
   usage/outputs
   usage/tuning
   usage/rfp
   usage/models
   usage/troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Python API

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   developing

Where things live
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Piece
     - What it is
   * - ``Neural_Image_Classifier-<version>.jar``
     - The plugin. A thin orchestration layer: collect dialog input, hand off
       to Python, render what comes back.
   * - ``models.zip`` (~90 MB)
     - The two trained classifiers and their thresholds, fetched on first run
       into ``~/.neural-imgs/models``.
   * - ``neural_imgs`` (Python)
     - The pipeline itself — tiling, BaSiC, CellPose, the classifiers, the RFP
       rule, reporting and the result cache. See :doc:`api/index`.

The Java side deliberately contains no reimplementation of BaSiC, CellPose or
PyTorch: there is one code path for the science, and it is the Python one.
