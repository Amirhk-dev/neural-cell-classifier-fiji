# Neural Cell Classifier — a Fiji plugin

Single-cell **OPC / Neuron (β-III-tubulin) / Transduced (RFP)** calls on Zeiss
`.czi` images, from Fiji's `Plugins` menu — one image or a whole folder, without
touching Python or a terminal.

Each run reads the CZI, tiles it, corrects illumination (BaSiC), detects nuclei
on DAPI (CellPose), and classifies every detected cell with two trained
ResNet-18 models plus an RFP intensity rule. Results come back as a Fiji results
table, CSVs, an UpSet co-positivity plot, the three reprogramming ratios, and
optional per-cell ROIs and example-cell figures.

- **Install is one file.** Drop the `.jar` into Fiji and run it — the first run
  downloads its own Python environment and the trained models and caches both.
  No admin rights, nothing to copy by hand.
- **You declare which channel is which**, and markers you do not have are
  optional: `0` means "not in my images" and the run still goes ahead, reporting
  that marker as *not measured* rather than negative. Only DAPI is required.

📖 **[Full manual](docs/sphinx/index.rst) — install, every dialog field, channel layouts,
saved results, the RFP rules, troubleshooting, and the Python API.**

---

## Install

You need **Fiji** ([fiji.sc](https://fiji.sc)) on **Java 21 or newer**
(`Help > About ImageJ...`), about **5 GB of free disk**, and internet access on
the first run only. A GPU is optional — CUDA and Apple Silicon (MPS) are used
automatically, otherwise it runs on CPU, correctly but slowly.

1. Download `Neural_Image_Classifier-<version>.jar` from the
   [latest release](https://github.com/Amirhk-dev/neural-cell-classifier-fiji/releases/latest).
2. `Plugins > Install...` inside Fiji, or drop the jar into `Fiji.app/plugins/`
   yourself.
3. **Restart Fiji** — `Refresh Menus` adds the menu entries but does not reload
   plugin code.

```
Plugins > Neural Image Processor > Classify Cells (OPC/B3Tub/RFP)...
                                > Show Cell (all channels)...
                                > Install / Verify Models...
```

**The first run downloads ~1–2 GB once:** the isolated conda environment
(PyTorch + CellPose + BaSiCPy, built by [Appose](https://github.com/apposed/appose),
which never touches any Python you already have) and the ~90 MB model bundle
into `~/.neural-imgs/models/`. Expect several minutes; every later run starts in
seconds. `Install / Verify Models...` does the model half ahead of time, or just
reports what is installed.

See [the install page](docs/sphinx/usage/install.rst) for the macOS jar
location, internal mirrors behind a proxy, and the rest.

---

## Run it

`Plugins > Neural Image Processor > Classify Cells (OPC/B3Tub/RFP)...`

**Only the first group is required.** Everything below it is tuning, already at
the published defaults — except **Channels**, which describes *your* images.
The dialog's **Help** button carries the same guidance inside Fiji.

| Group | Fields | Default |
| --- | --- | --- |
| **Input and output** | Input mode, CZI image *or* input folder, model directory, output directory | — |
| **Channels** | one channel number per marker, counting from 1; `0` = not in my images | `1, 2, 3, 4, 5` |
| **Detection** | Patch grid (tiles per side), cell probability, flow | `4`, `0.1`, `0.2` |
| **RFP positivity** | rule, exclusion radius, one gate per rule | `neighbour`, `60` px, `16.5` / `5` |
| **Performance** | images prepared ahead (memory), extraction threads | `1`, `0` = auto |
| **Outputs** | UpSet plot, example cell per case, cell ROIs, load saved results | on, off, off, on |

Click **OK**. When the run finishes **the dialog comes back pre-filled**, so the
next image is one field-change away; **Cancel** finishes. The dialog is
non-modal, so the Log and Results table stay readable while it is up.

Field-by-field reference: **[the dialog page](docs/sphinx/usage/dialog.rst)**.

### What you get

- **One combined Results table** across every image, `image` first (`cell_id`
  restarts per image), with probabilities, both RFP scores, the bounding box and
  `n_cells_in_region`. A marker set to `0` gets no columns at all.
- A per-image folder under the output root holding its CSV, quantification,
  UpSet plot, example-cell figures and saved results, plus a combined
  `all_images_classifications.csv` at the root.
- The three **reprogramming ratios** in the Log and in
  `<image>_quantification.txt`.
- **Colour-coded ROIs** in the ROI Manager if ticked — cyan = OPC+,
  yellow = B3-Tub+, magenta = both, grey = neither.

The Log (`Window > Log`) carries one block per image: counts, ratios, where its
folder is, whether the numbers were computed or re-loaded, and any non-fatal
problem. **One image failing does not stop the batch.**

A second run over the same image and settings comes back from saved results in
seconds. Details of what invalidates them:
**[outputs and saved results](docs/sphinx/usage/outputs.rst)**.

### Looking at one cell

`Show Cell (all channels)...` opens a composite of every channel your run
declared for one cell — the exact raw pixels the classifiers scored. Select its
ROI in the ROI Manager to pre-fill the id. It needs a `Classify Cells...` run
earlier in the session and does not re-run the pipeline.

---

## Documentation

| | |
| --- | --- |
| [Install](docs/sphinx/usage/install.rst) | Requirements, steps, first-run downloads, proxies |
| [The dialog](docs/sphinx/usage/dialog.rst) | Every field, its default and its range |
| [Channels](docs/sphinx/usage/channels.rst) | Optional markers, reordered acquisitions, what *not measured* means |
| [Outputs](docs/sphinx/usage/outputs.rst) | Tables, figures, folder layout, saved results |
| [Tuning](docs/sphinx/usage/tuning.rst) | Patch grid, CellPose thresholds, CPU/GPU, what runs in parallel |
| [RFP rules](docs/sphinx/usage/rfp.rst) | Why `neighbour` replaced `legacy`, why the gates differ |
| [Models](docs/sphinx/usage/models.rst) | The bundle, and using your own checkpoints |
| [Troubleshooting](docs/sphinx/usage/troubleshooting.rst) | Error messages and what they mean |
| [Python API](docs/sphinx/api/index.rst) | Driving the same pipeline from a script |

Build the manual as HTML:

```bash
pip install -r docs/sphinx/requirements.txt
make -C docs/sphinx html        # -> docs/sphinx/_build/html/index.html
```

It needs neither a GPU nor the inference stack — the heavy imports are mocked.

## For developers

Repository layout, how the Java and Python halves talk, and building the jar:
[`docs/DEVELOPING.md`](docs/DEVELOPING.md). Cutting a release:
[`docs/PUBLISHING.md`](docs/PUBLISHING.md).

## Licence and contact

MIT — see [LICENSE](LICENSE). Issues and questions:
[GitHub issues](https://github.com/Amirhk-dev/neural-cell-classifier-fiji/issues)
— please quote the build id from the dialog title and the Fiji Log of the
failing run. CellPose, BaSiC, PyTorch and Fiji/ImageJ are used under their own
licences.
