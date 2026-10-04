# Neural Cell Classifier — a Fiji plugin

Single-cell **OPC / Neuron (β-III-tubulin) / Transduced (RFP)** calls on Zeiss
`.czi` images, from Fiji's `Plugins` menu — one image or a whole folder, without
touching Python or a terminal.

Each run reads the CZI, splits it into tiles, corrects illumination (BaSiC),
detects nuclei on DAPI (CellPose), and classifies every detected cell with two
trained ResNet-18 models (OPC, β-III-tubulin) plus an RFP intensity rule.
Results come back as a Fiji results table, CSVs, an UpSet co-positivity plot,
the three reprogramming ratios, and optional per-cell ROIs and example-cell
figures.

**Install is one file.** Drop the `.jar` into Fiji and run it — the first run
downloads its own Python environment and the trained models and caches both.
Nothing else to install, no files to copy by hand, no admin rights.

---

## 1. Install

### What you need

- **Fiji** from <https://fiji.sc>, recent enough to bundle **Java 21 or newer**
  (check `Help > About ImageJ...`). Current downloads do; only a long-unused
  `Fiji.app` still on Java 8 would not — see [Troubleshooting](#5-troubleshooting).
- **Internet access on first run only** (1–2 GB, once per machine — see
  [What the first run downloads](#2-what-the-first-run-downloads)).
- **Roughly 5 GB of free disk** for the cached Python environment and models.
- **A GPU is optional.** CUDA and Apple Silicon (MPS) are detected and used
  automatically; without either it runs on CPU, correctly but slowly (Q1).

### Steps

1. Download **`Neural_Image_Classifier-1.0.0.jar`** from the
   [latest release](https://github.com/Amirhk-dev/neural-cell-classifier-fiji/releases/latest).
2. Get it into `Fiji.app/plugins/`, either way:
   - **From inside Fiji:** `Plugins > Install...`, pick the `.jar`, and accept
     the default save location when prompted.
   - **By hand:** drop it into `Fiji.app/plugins/` yourself.
     - macOS: right-click the Fiji app → *Show Package Contents* →
       `Contents/Java/Fiji.app/plugins/` (or wherever your `Fiji.app` lives).
     - Windows / Linux: `<your Fiji install dir>/Fiji.app/plugins/`.
3. **Restart Fiji.** (`Help > Refresh Menus` adds the menu entries but does not
   reload plugin code, so prefer a restart.)

You should now have:

```
Plugins > Neural Image Processor > Classify Cells (OPC/B3Tub/RFP)...
                                > Show Cell (all channels)...
                                > Install / Verify Models...
```

That is the whole install. There is no model bundle to request and no
configuration file to edit.

---

## 2. What the first run downloads

The first time you classify anything, two one-time downloads happen, each with a
progress window:

| | Size | What it is |
| --- | --- | --- |
| **Python environment** | ~1–2 GB | An isolated conda environment (Python 3.11 + PyTorch + CellPose + BaSiCPy) built by [Appose](https://github.com/apposed/appose). Size depends on platform — a CUDA-capable machine pulls the larger GPU wheels. It does **not** touch any Python you already have. |
| **Model bundle** | ~90 MB | The two trained classifiers and their decision thresholds, into `~/.neural-imgs/models/`. |

Expect several minutes. Every later run finds both cached and starts in seconds.

If you would rather get that out of the way before a session — or you just want
to know whether the models are already there — run
**`Plugins > Neural Image Processor > Install / Verify Models...`**. It reports
what is installed and downloads only what is missing.

> **Behind a restrictive proxy?** If `github.com` release downloads are blocked,
> an admin can host `models.zip` internally and start Fiji with
> `-Dneuralimgs.models.url=https://internal.host/models.zip`. The integrity
> check is not relaxed for a mirror: the file must be the same bytes.

---

## 3. Run it

`Plugins > Neural Image Processor > Classify Cells (OPC/B3Tub/RFP)...`

The dialog's **top group is everything a run requires**; every group below it is
tuning whose defaults are the published ones, so a first run only needs the top.
Click the dialog's **Help** button for the same guidance inside Fiji.

**Required**

- **Input** — `Single image` or `Folder of images`. The row the selected mode
  does not use is **greyed out** (field, `Browse` button and caption), so only
  the path that will actually be read is editable. The greyed field keeps its
  text, and both persist between runs, so switching mode never means retyping a
  path.
- **CZI image** — one Zeiss `.czi`, **5 channels in the order DAPI, OPC, RFP,
  B3Tub, BF**.
- **Input folder** — every `.czi` *directly* inside it is classified. Not
  recursive (Q8).
- **Model directory** — pre-filled with `~/.neural-imgs/models`, downloaded on
  first use. Change it only to use your own retrained checkpoints; a complete
  directory is never overwritten.
- **Output directory** — the output **root**. One subfolder per image is created
  inside it, named after the image (Q8).

**Tiling and nucleus detection**

- **Patch grid (NxN)** (default `4`) — how many tiles per side; `4` means 16
  tiles. A tile *count*, not a pixel size. Whole number, at least 1 (Q3).
- **Cell probability threshold** (default `0.1`) and **Flow threshold**
  (default `0.2`) — CellPose's two nucleus-detection knobs (Q1).

**RFP positivity** — `RFP rule` (default `neighbour`), `RFP gate, neighbour
rule` (default `16.5`), `RFP gate, legacy rule` (default `5`), `Neighbour
exclusion radius` (default `60` px). The two gates are separate fields because
the rules are on **different scales** (Q9).

**Performance** — `Images prepared ahead` (default `1`, a memory knob) and
`Extraction threads` (default `0` = auto, a CPU knob) (Q10).

**Outputs** — `Generate UpSet plot` (on, Q4), `Show one example cell per UpSet
case` (off, Q5), `Add cell ROIs to the ROI Manager` (off, Q7), `Load saved
results if available` (on, Q6).

Click **OK**. When the run finishes **the dialog comes back**, pre-filled with
what you just used, so the next image is one field-change away; **Cancel**
finishes. The dialog is non-modal, so the Log window and Results table stay
readable while it is up.

### What you get

- **One combined Results table** across every image processed, with an `image`
  column first (`cell_id` restarts at 0 per image, so the two together identify
  a cell), plus `opc_prob`, `b3tub_prob`, `rfp_pos`, both RFP scores, the
  bounding box and `n_cells_in_region`.
- A per-image CSV, `<image-name>_fixed_model_classifications.csv`, inside that
  image's subfolder, and a combined `all_images_classifications.csv` at the
  output root.
- An **UpSet co-positivity plot**, `<image-name>_upset_plot.png` (if ticked).
  For a single image it also opens in a window; for a folder run the paths are
  logged instead — twenty images would otherwise open sixty windows at the end
  of an overnight run.
- The three **reprogramming ratios** (OPC state ratio, transduction performance,
  conversion ratio) printed in the Log and saved as
  `<image-name>_quantification.txt`.
- **8 per-case example cells**, `<image-name>_upset_case_<combo>.png` (if
  ticked, Q5).
- **Saved results**, `<image-name>_classification_cache.json` plus
  `<image-name>_classification_masks.npz`, so the same image never has to be
  classified twice (Q6).
- **Colour-coded ROIs** in the ROI Manager — cyan = OPC+, yellow = B3-Tub+,
  magenta = both, grey = neither — if ticked (Q7).

The Log window (`Window > Log`) carries one block per image: its counts, the
three ratios, where its folder is, whether the numbers were computed or
re-loaded, and any non-fatal problem. **One image failing does not stop the
batch** — it is logged as `FAILED` and the rest still run.

### Looking at one cell

`Plugins > Neural Image Processor > Show Cell (all channels)...` opens a
5-channel composite (DAPI/OPC/RFP/B3Tub/BF) for one cell — the exact raw pixels
the classifiers scored. Select its ROI in the ROI Manager first to pre-fill the
id, or just type one. Toggle channels with the "C" slider
(`Image > Color > Channels Tool`) like any Fiji composite. It needs a
`Classify Cells...` run to have happened earlier in the session and does not
re-run the pipeline. After a folder run it also asks **which image**, since cell
ids restart per image.

---

## 4. Where the numbers come from

The thresholds and parameters are frozen, not tuned per run: **OPC 0.45**,
**B3-Tub 0.4825**, CellPose **cellprob 0.1 / flow 0.2**, and the
**neighbour-excluded RFP rule at radius 60 px, gate 16.5**. They are the values
the published analysis uses, and the shipped checkpoints are the ones those
thresholds were selected for — `thresholds.json` travels inside the model bundle
with the weights so the two cannot drift apart.

The classifiers are ResNet-18, trained per marker on native-resolution
single-cell crops (200 px, soft-masked to the nucleus with a per-marker sigma).
Each was trained on a class-balanced set of biologist-labelled cells with a
single held-out validation split:

| Marker | Train / validation cells | Best validation accuracy |
| --- | --- | --- |
| OPC | 219 / 39 | 97.4% |
| B3-Tub | 224 / 40 | 92.5% |

**Read those accuracies with their sample sizes in mind** — 39 and 40 cells is a
small validation set, so the true error bars are wide, and the figures describe
the cells the models were trained and validated on rather than a guarantee for a
new experiment. Spot-check a new dataset (`Show one example cell per UpSet case`
is the quickest way) before trusting counts from it.

The Python pipeline in [`python/`](python/) is the same code the analysis ran,
vendored unchanged — see [`docs/DEVELOPING.md`](docs/DEVELOPING.md).

---

## 5. Troubleshooting

- **First run takes a long time / looks stuck** — expected; it is building the
  Python environment. Watch the progress window's status text. If it genuinely
  stalls, check internet access.
- **`Model download failed: HTTP 404`** — the release asset for this plugin
  version is not reachable. Check internet access, then
  [open an issue](https://github.com/Amirhk-dev/neural-cell-classifier-fiji/issues).
- **`Downloaded model bundle is corrupt`** — the download was truncated or
  intercepted. Delete `~/.neural-imgs/models/` and run
  `Install / Verify Models...` again.
- **`The models need to be downloaded into: ... but that location is not
  writable`** — point *Model directory* at somewhere inside your home folder.
- **Runs but is slow** — you are on CPU (expected on a laptop with no NVIDIA GPU
  and no Apple Silicon); see Q1. Try `Patch grid = 2` first to sanity-check. A
  *second* run on the same image should be near-instant; if it is not, the Log
  says why the saved results were rejected (Q6).
- **An image I already classified is being re-classified** — something in the
  saved-results key changed and the Log names it (Q6). Most likely a different
  output directory, a different patch grid, or a changed CellPose threshold.
- **`java.lang.UnsupportedClassVersionError: ... class file version 65.0 ...
  only recognizes class file versions up to 52.0`** — Fiji's bundled Java is too
  old (52 = Java 8; this plugin needs 21, because Appose does). Download a fresh
  Fiji from <https://fiji.sc>, replace the old `Fiji.app`, reinstall the jar and
  restart. Check with `Help > About ImageJ...`.
- **Menu entries missing after copying the jar in** — restart Fiji;
  `Refresh Menus` does not reload plugin classes.
- **Which build am I running?** The dialog title and the first Log line both
  carry the build id. Fiji keeps a plugin's classes loaded until it restarts, so
  "I copied the new jar in" and "Fiji is running the new jar" are different
  claims.

---

## Questions

**1. Can I set CellPose's parameters? CPU or GPU?**
Yes to the first. The dialog exposes both detection thresholds, defaulting to
the published values:

| Field | Default | Range | What it asks |
| --- | --- | --- | --- |
| Cell probability threshold | `0.1` | `-3.0 … 2.0` | Does this pixel look like it belongs to a cell? **Lower → more cells detected.** |
| Flow threshold | `0.2` | `0.0 … 1.0` | Are the proposed mask's flows consistent with what CellPose predicted? **Lower → fewer masks accepted (stricter).** |

Both accept any decimal in range (`0.15`, `-1.25`, …), not just the tenth-steps
the usual tables list; out-of-range or unparseable entries are rejected with a
message rather than silently clamped. Cell diameter is not passed (CellPose
auto-estimates it). Because these two decide which nuclei exist at all, they are
part of the saved-results key: changing either makes a previous run stale and
the image is re-classified (Q6).

Device selection is automatic: CUDA if present, else Apple Silicon MPS, else
CPU. On an Intel Mac (no MPS) that means CPU-only and is noticeably slow —
CellPose 4's model (Cellpose-SAM) has a much heavier backbone than classic
CellPose, so BaSiC + CellPose + two ResNet-18 passes per cell on CPU alone can
be tens of minutes *per tile*. Try `Patch grid = 2` first. On Apple Silicon MPS
should be dramatically faster; if it still crawls, confirm in the Log that
CellPose picked MPS (it prints `Using device: ...` on load) rather than silently
falling back.

**2. Does the plugin have a GUI?**
A minimal one: an ImageJ dialog with the fields above, plus progress windows for
the first-run downloads. Output is native Fiji UI — ROI overlay, Results table,
ordinary image windows for the UpSet plot and for "Show Cell" — not a custom
results screen.

**3. Can I set the patch size?**
Yes — **Patch grid (NxN)**, a whole number of at least 1 (a fractional entry is
rejected rather than quietly truncated). It is a *tile count per side*, not a
pixel size: `4` splits the image into a 4×4 = 16-tile grid before running
BaSiC/CellPose per tile. This mirrors how the training data was built, so
changing it changes results, not just speed.

**4. Does it generate an UpSet plot?**
Yes, optionally (on by default). Sets are named **OPC**, **Neuron** (B3-Tub+)
and **Transduced** (RFP+), with a fixed bar order so plots are directly
comparable across images. It opens as an image window (single-image runs) and is
saved as `<image-name>_upset_plot.png` next to the CSV. Untick to skip it —
classification, ROIs and CSV are unaffected.

**5. Can it show single cells with their probabilities?**
Two ways, both opt-in.

**a) One example cell per UpSet case** — the checkbox (off by default). For each
of the 8 cases it picks one representative cell and renders `DAPI + mask`
followed by all five channels, with the DAPI mask outlined in green on every
panel, titled with the case, the cell id and that cell's scores
(`OPC 0.87 / B3 0.12 / RFP 1`). Each is opened as its own window and saved as
`<image-name>_upset_case_<combo>.png`, where `<combo>` is the
`OPC/Neuron/Transduced` flags as `0`/`1` (e.g. `101` = OPC+, Transduced+). Cases
with no cell in this image are skipped and named in the Log, alongside how many
candidates each case had.

The panels are cut from the same BaSiC-corrected, percentile-normalised tile the
cells were detected on, so they are the pixels that were classified rather than
a lookalike. The green outline is drawn whenever the cell's mask is available —
always for a fresh run, and for a cached run whenever the
`_classification_masks.npz` sidecar sits next to the cache file.

On a **cached** run this step has to read the CZI and re-fit BaSiC, since a
cache hit skips both — a few minutes, against the hours a re-classification
costs, and both are then reused for the rest of the Fiji session. A fresh run
reuses the fit it already made, so the montages cost it almost nothing.

**b) Any cell you choose** — the **`Show Cell (all channels)...`** menu item.
It reuses the in-memory tile data the classification run already loaded, so it
never re-runs BaSiC/CellPose. The one case where it does read the CZI is right
after results were restored from a saved-results file (Q6), since that path
skips the image read entirely — it then happens once, on the first cell you open
(or on the per-case montages, whichever comes first), and the Log says so.

**6. Do I have to wait hours every time I look at the same image?**
No. Classifying one CZI takes hours, but the answer is fully determined by its
inputs, so each run writes it into the **output directory** you chose, named
after the image:

- `<image-name>_classification_cache.json` — every cell's id, bounding box,
  probabilities and calls, plus the key described below (~200 bytes per cell).
- `<image-name>_classification_masks.npz` — the per-cell DAPI masks, bit-packed
  and compressed. Only the per-case montages' outlines need these; if the file
  is missing, the montages simply render without the green contour.

With **"Load saved results if available"** ticked, a later run looks for the
JSON *in the output directory you give it* — so pointing at the same output
directory is what makes an image "already done". It then rebuilds ROIs, Results
table, CSV, UpSet plot, quantification and (if asked) the per-case montages from
it in seconds; only the classification itself is skipped.

Saved results are reused **only** when they provably describe the same
computation. The file stores a key of: the image file's **size in bytes**, the
patch grid, both CellPose thresholds, the RFP rule and its gate, and a
fingerprint of the model bundle (the contents of `thresholds.json` plus each
checkpoint's byte size — deliberately not mtimes, so re-copying an identical
bundle does not force a needless re-run). If any of those differs the file is
ignored and the image is re-classified, and the Log prints the reason, e.g.
`re-classified because saved results do not match: patch grid 4 -> 8`. A corrupt
or older-format file reads as unreadable and also just triggers a re-run; it
never breaks anything.

**Moving the CZI to another folder does not trigger a re-run.** Where a file
sits is not an input to the classification, so a relocated image still matches;
the Log just notes `image moved from <old path>`. What identifies the image is
its *filename* (which is how the cache file is found) plus its byte size. The
one case this cannot distinguish is two genuinely different CZIs that share both
a filename and an exact byte size and are classified into the same output
directory — untick the checkbox if that is ever a real risk.

Untick to force a fresh run regardless, and delete the two files to throw the
saved results away — both are safe to delete at any time.

**7. Why doesn't the ROI Manager open?**
It is off by default — tick **"Add cell ROIs to the ROI Manager"** to get it.
The ROIs are bounding boxes in *tile* coordinates and the plugin never opens the
CZI in Fiji, so there is no image on screen for them to overlay; popping the
window open at the end of every run was noise for anyone reading the Results
table and the CSV instead. Ticking it is also what lets
`Show Cell (all channels)...` pre-fill a cell id from the selected ROI.

**8. What does folder mode do with the output directory?**
Treats it as a **root** and gives every image its own subfolder, named after the
image:

```
<output directory>/
├── all_images_classifications.csv          ← the combined table, matching Fiji's
├── 7d_OPC_reprogramming_20x_#1_.../
│   ├── ..._fixed_model_classifications.csv
│   ├── ..._quantification.txt
│   ├── ..._upset_plot.png
│   ├── ..._classification_cache.json       ← saved results (Q6)
│   └── ..._classification_masks.npz
└── 7d_OPC_reprogramming_20x_#2_.../
    └── ...
```

Not cosmetic: saved results are looked up by **image stem**, so two images
written flat into one directory would overwrite each other's cache and CSV. One
folder per image is also what makes a batch re-runnable image by image —
re-point the plugin at the same folder and everything already done comes back
from cache in seconds, while only the new images are classified.

Only `.czi` files **directly inside** the folder are picked up — no recursion,
no hidden files. A folder of images is the unit you chose; silently walking into
subfolders would turn "this plate" into "everything under it". They are
processed in sorted order, so cell ids and the Log are reproducible run to run.

**9. Which RFP rule is used, and why are there two gates?**
RFP is the one marker with no trained classifier — it is called from pixel
intensity. The default is **neighbour-excluded, radius 60 px, gate 16.5**.

| Rule | What the cell is compared against | Default gate |
| --- | --- | --- |
| `neighbour` (production) | its crop's background **with pixels near other nuclei dropped** | `16.5` |
| `legacy` | its crop's 95th percentile, neighbours included | `5` |

The legacy rule's flaw is measured, not theoretical: one bright neighbour owns
the crop's percentile, so a genuinely transduced cell scores negative — on the
control image 112 cells sit below `-10` for that reason, and the score
correlates with crowding at `-0.47`. Excluding neighbour pixels takes that to
`-0.004` and recovers ~110 cells across five images, while conversion rate (the
biologists' anchor) stays inside its 67–82% band.

Both rules are scored for every cell and both travel in the results table
(`rfp_neighbour_score` / `rfp_neighbour_pos`, `rfp_score` / `rfp_legacy_pos`),
with `rfp_method` recording which made the `rfp_pos` call — so a saved table can
be re-read under the other rule without re-classifying anything.

**The two gates are not interchangeable.** Different background estimators mean
different scales: the neighbour score sits higher, so running it at the legacy
gate of `5` admits 100–250 extra low-confidence cells per image. That is why
there is one field per rule rather than one shared field.

A third rule, soft-mask, was built, measured on all five images and **rejected**
— its extra positives convert at the image's *background* B3-Tub+ rate (16–33%
against 66–80% for cells every rule agrees on), i.e. they are mostly false
positives — so it is deliberately not offered.

**10. What do the two performance fields do?**
Inside one image the tiles are **pipelined, not serial**: CellPose runs on one
tile at a time (one GPU, one model copy, so a second concurrent call would
contend rather than add throughput) while cell extraction for already-detected
tiles runs on **Extraction threads** CPU workers (`0` = size the pool from this
process's core allocation). In folder mode, **Images prepared ahead** images are
read and BaSiC-fitted ahead of the GPU lane; each one holds its whole tile array
(~840 MB for a 16-tile CZI), which makes this the memory knob. `1` is enough to
keep the GPU fed.

**11. Can I use my own retrained checkpoints?**
Yes — point **Model directory** at a directory holding `thresholds.json` plus
one subdirectory per marker containing `best_model.pt`, named exactly as
`prod_exp_name()` in
[`python/src/neural_imgs/inference/fixed_model_pipeline.py`](python/src/neural_imgs/inference/fixed_model_pipeline.py)
derives it. A directory that already has all of those is never overwritten by
the download.

---

## For developers

Building the jar, the repository layout, and how the model bundle is published:
[`docs/DEVELOPING.md`](docs/DEVELOPING.md).

## Licence and contact

MIT — see [LICENSE](LICENSE). Issues and questions:
[GitHub issues](https://github.com/Amirhk-dev/neural-cell-classifier-fiji/issues).
