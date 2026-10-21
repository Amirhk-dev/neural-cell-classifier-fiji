Troubleshooting
===============

First run and downloads
-----------------------

**First run takes a long time / looks stuck**
  Expected — it is building the Python environment. Watch the progress window's
  status text. If it genuinely stalls, check internet access.

**Model download failed: HTTP 404**
  The release asset for this plugin version is not reachable. Check internet
  access, then `open an issue
  <https://github.com/Amirhk-dev/neural-cell-classifier-fiji/issues>`_.

**Downloaded model bundle is corrupt**
  The download was truncated or intercepted. Delete ``~/.neural-imgs/models/``
  and run ``Install / Verify Models...`` again.

**The models need to be downloaded into: ... but that location is not writable**
  Point **Model directory** at somewhere inside your home folder.

Speed
-----

**Runs but is slow**
  You are on CPU — expected on a laptop with no NVIDIA GPU and no Apple
  Silicon; see :doc:`tuning`. Try ``Patch grid = 2`` first to sanity-check. A
  *second* run on the same image should be near-instant; if it is not, the Log
  says why the saved results were rejected.

**An image I already classified is being re-classified**
  Something in the saved-results key changed and the Log names it
  (:ref:`saved-results`). Most likely a different output directory, a different
  patch grid, a changed CellPose threshold, or a changed channel assignment.

Channels and columns
--------------------

**the channel layout describes 5 channels (...) but the image has only 3**
  The **Channels** fields ask for a channel the image does not have. Set the
  markers your images lack to ``0`` (:doc:`channels`). In a folder run only that
  image fails; the rest continue.

**A marker's columns are missing from the Results table**
  Its **Channels** field is ``0``, so it was not measured. The Log's per-image
  line and the quantification both name what was left out. If the channel *is*
  set but the marker is still missing, the model directory shipped no
  checkpoint for it — the Log says so when the models load (:doc:`models`).

**no DAPI channel declared**
  ``DAPI`` is the one channel that cannot be ``0``; nuclei are detected on it.

Install and versions
--------------------

**java.lang.UnsupportedClassVersionError: ... class file version 65.0 ... only recognizes class file versions up to 52.0**
  Fiji's bundled Java is too old — 52 = Java 8, and this plugin needs 21,
  because Appose does. Download a fresh Fiji from https://fiji.sc, replace the
  old ``Fiji.app``, reinstall the jar and restart. Check with
  ``Help > About ImageJ...``.

**Menu entries missing after copying the jar in**
  Restart Fiji. ``Refresh Menus`` does not reload plugin classes.

**Which build am I running?**
  The dialog title and the first Log line both carry the build id. Fiji keeps a
  plugin's classes loaded until it restarts, so "I copied the new jar in" and
  "Fiji is running the new jar" are different claims.

Still stuck
-----------

`Open an issue
<https://github.com/Amirhk-dev/neural-cell-classifier-fiji/issues>`_ with the
build id from the dialog title and the Fiji Log contents of the failing run.
