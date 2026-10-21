Python API
==========

The pipeline the plugin drives. These pages are generated straight from the
modules under ``python/src/neural_imgs``, which are a byte-identical copy of
the research repository's inference subset -- so what you read here is what
runs.

The entry point for driving it from a script is
:class:`neural_imgs.inference.batch.BatchClassifier`:

.. code-block:: python

   from neural_imgs.inference.batch import BatchClassifier, BatchInput
   from neural_imgs.inference.fixed_model_pipeline import load_models
   from neural_imgs.inference.rfp import RfpConfig

   models = load_models("~/.neural-imgs/models")   # the bundle the plugin fetches
   result = BatchClassifier(models).run(BatchInput(
       image_paths=["image.czi"],
       out_root="out",
       model_dir="~/.neural-imgs/models",
       patch_grid=4,
       rfp=RfpConfig(),                            # neighbour rule, r=60, gate 16.5
   ))
   print("\n".join(result.summary_lines()))

Declaring the channels
----------------------

:doc:`neural_imgs.channels`
   Which image channel carries which marker, and which markers a run can call.

.. toctree::
   :hidden:

   neural_imgs.channels

Running a classification
------------------------

:doc:`neural_imgs.inference.batch`
   Classify a folder of images, with the CPU stages overlapped across them.

:doc:`neural_imgs.inference.fixed_model_pipeline`
   The fixed production pipeline: tiling, BaSiC, CellPose, the classifiers.

:doc:`neural_imgs.inference.patch_extraction`
   Turning one image's tiles into scored single cells, CPU work overlapped.

:doc:`neural_imgs.inference.result_cache`
   The on-disk cache of a completed run, and the key that decides a hit.

.. toctree::
   :hidden:

   neural_imgs.inference.batch
   neural_imgs.inference.fixed_model_pipeline
   neural_imgs.inference.patch_extraction
   neural_imgs.inference.result_cache

Calling RFP
-----------

:doc:`neural_imgs.inference.rfp`
   Which RFP rule production uses, and its operating point.

:doc:`neural_imgs.positivity`
   Per-channel positivity calls for single cells.

:doc:`neural_imgs.positivity.base`
   The typed contract a positivity rule implements.

:doc:`neural_imgs.positivity.heuristic`
   The OPC-project intensity heuristic, applied to any marker channel.

:doc:`neural_imgs.positivity.neighbour_contrast`
   The production rule: a background a neighbouring cell cannot hijack.

:doc:`neural_imgs.positivity.soft_mask`
   The soft-mask heuristic -- built, measured and deliberately not shipped.

.. toctree::
   :hidden:

   neural_imgs.inference.rfp
   neural_imgs.positivity
   neural_imgs.positivity.base
   neural_imgs.positivity.heuristic
   neural_imgs.positivity.neighbour_contrast
   neural_imgs.positivity.soft_mask

Reporting
---------

:doc:`neural_imgs.inference.reporting`
   The UpSet co-positivity plot and the side quantification.

:doc:`neural_imgs.inference.case_montage`
   One representative cell per UpSet case, across every channel.

.. toctree::
   :hidden:

   neural_imgs.inference.reporting
   neural_imgs.inference.case_montage

Image handling
--------------

:doc:`neural_imgs.io.read_image`
   Reading a CZI.

:doc:`neural_imgs.processing.processing`
   Tiling, illumination correction and normalisation.

.. toctree::
   :hidden:

   neural_imgs.io.read_image
   neural_imgs.processing.processing

Model and preprocessing
-----------------------

:doc:`neural_imgs.training.model`
   The classifier: ResNet-18 and MobileNetV3-Small backbones.

:doc:`neural_imgs.training.raw_dataset`
   Native-resolution single-cell preprocessing -- what the checkpoints were trained under.

:doc:`neural_imgs.utils.utils`
   Small shared helpers.

.. toctree::
   :hidden:

   neural_imgs.training.model
   neural_imgs.training.raw_dataset
   neural_imgs.utils.utils
