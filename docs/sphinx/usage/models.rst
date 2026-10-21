Models
======

The bundle
----------

``models.zip`` (~90 MB) holds the two trained classifiers and their decision
thresholds. The plugin pins three things about it — its URL, its SHA-256 and its
byte size — and verifies all three, so a truncated or substituted download
fails loudly instead of classifying with the wrong weights.

It is fetched into **Model directory** (``~/.neural-imgs/models`` by default) on
the first run, or on demand via ``Plugins > Neural Image Processor > Install /
Verify Models...``. A directory that already holds a complete bundle is never
overwritten.

The default location is under your home directory rather than beside the jar
because ``Fiji.app/plugins/`` is not reliably writable — a macOS app bundle, a
shared or read-only install — and the download has to succeed without an
administrator.

Your own retrained checkpoints
------------------------------

Point **Model directory** at a directory holding ``thresholds.json`` plus one
subdirectory per marker containing ``best_model.pt``, named exactly as
:func:`neural_imgs.inference.fixed_model_pipeline.prod_exp_name` derives it.

A directory holding **only one** of the two markers also works: the missing one
is reported in the Log as not measured, exactly as if its channel were absent
(:doc:`channels`).

.. important::

   The directory names are derived from the channel order the models were
   *trained* on and never change with your **Channels** settings — they
   identify a checkpoint, not a plane of your image.

See :func:`neural_imgs.inference.fixed_model_pipeline.load_models`.
