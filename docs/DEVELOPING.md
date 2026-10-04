# Developing

## Layout

```
.
├── pom.xml                              Maven project for the Fiji plugin
├── src/main/java/com/neuralimgs/fiji/
│   ├── NeuralClassifyPlugin.java        the plugin: dialog, run loop, rendering
│   └── ModelBootstrap.java              downloads + verifies the model bundle
├── src/main/resources/
│   ├── plugins.config                   the three Fiji menu entries
│   └── scripts/
│       ├── environment.yml              the Python env Appose builds on first run
│       ├── classify_cell_image.py       Appose worker: classify an image or folder
│       └── get_cell_crop.py             Appose worker: one cell's raw pixels
├── python/                              the pipeline the plugin drives
│   ├── pyproject.toml
│   └── src/neural_imgs/...
├── tools/
│   ├── prepare_model_bundle.py          builds the models.zip release asset
│   ├── check-model-asset.sh             verifies a published bundle (curl only)
│   ├── publish-release.sh               one-shot release, needs `gh`
│   └── manual-test/                     drives the model download over localhost
└── docs/
    ├── DEVELOPING.md
    └── PUBLISHING.md                    how to cut a release (browser route)
```

Two artefacts ship, and they ship separately:

| Artefact | Where it lives | Reaches the user when |
| --- | --- | --- |
| `Neural_Image_Classifier-<version>.jar` | GitHub release asset | they download and install it |
| `models.zip` (~90 MB) | GitHub release asset, tag `models-vN` | the plugin fetches it on first run |

The Python package is **not** a third artefact: `environment.yml` pip-installs
`python/` straight from this repository, so a pipeline fix reaches existing
installs on their next environment build without a new jar.

## How the two halves talk

The Java side is deliberately a thin orchestration layer: collect dialog input,
hand off to Python, render what comes back. BaSiC, CellPose and PyTorch have no
practical Java equivalent, so there is no Java reimplementation of any of them
and no second code path to keep in step.

[Appose](https://github.com/apposed/appose) builds the conda environment and
runs a persistent Python worker. Both sides must agree on the Appose version —
the wire protocol is not guaranteed stable across releases — so
`appose.version` in `pom.xml` and the `appose==` pin in `environment.yml` are
one decision in two files. Change them together.

Two gotchas that cost real time when rediscovered:

- **The worker runs off the main thread.** No `pyplot` (it crashes on macOS);
  the reporting code uses `Figure` + `FigureCanvasAgg` directly for that reason.
- **numpy scalars do not cross the wire.** Appose serialises task outputs as
  JSON and hands Java an opaque object handle for a numpy scalar, which then
  fails the `Number` cast. Every value in `task.outputs` is coerced with
  `int()` / `float()` / `bool()` on purpose.

Each `task()` call gets a fresh `exec()` binding; only names passed to
`task.export(...)` survive into the next task. That is how the loaded models
(minutes to load) and the decoded image are reused across calls within one Fiji
session.

## Why `python/` is vendored, not depended on

`python/src/neural_imgs/` is a **byte-identical copy** of the inference subset
of the private research repository (`neural-image-processor`), reduced to the
import closure the two worker scripts actually reach:

```
inference/{batch,case_montage,fixed_model_pipeline,patch_extraction,reporting,result_cache,rfp}
positivity/{base,heuristic,neighbour_contrast,soft_mask}
io/read_image   processing/processing   training/{model,raw_dataset}   utils/utils
```

The import name stays `neural_imgs` and the files are copied rather than
rewritten because the preprocessing applied at inference **must** be the
preprocessing the shipped checkpoints were trained under. Identical files make
that claim checkable with `diff -r` instead of something to re-argue after every
edit.

Two files depart from that on purpose. The first is `training/__init__.py`,
trimmed to re-export only `model` and `raw_dataset`. The full package also exports the trainer,
evaluator, split manager and augmentation pipeline — none reachable from the
plugin, each pulling in a dependency (albumentations, scikit-learn) inference
never needs. Importing a submodule runs the package `__init__` first, so without
that trim every install would carry the training stack.

The second is `utils/utils.py`, where two commented-out absolute cluster paths
in `load_conf` were replaced with a usage example: a public repository should
not publish the internal filesystem layout. No executable line changed, and
`load_conf` is not reachable from either worker script.

So `diff -r` against the private repo's `src/neural_imgs/` should report exactly
those two files and nothing else. Anything more is drift to investigate.

To re-sync after a change upstream, copy the files again and re-check that the
closure is still closed:

```bash
PYTHONPATH=python/src python -c "
import sys, neural_imgs.inference, neural_imgs.training
from neural_imgs.inference.fixed_model_pipeline import load_models, crop_cell, load_patches
from neural_imgs.inference.batch import BatchClassifier, discover_images
assert 'neural-cell-classifier-fiji' in neural_imgs.__file__, 'shadowed by another install'
print(len([m for m in sys.modules if m.startswith('neural_imgs')]), 'modules, all vendored')
"
```

A new upstream import that is not in the list above will fail here rather than
at a biologist's first run.

## Building the jar

Needs JDK 21 (pom-scijava enforces `[21,)`, because Appose does) and Maven.

```bash
rm -rf target        # not `mvn clean` -- see below
mvn -B -DskipTests package
```

The deliverable is the **shaded** jar, `target/Neural_Image_Classifier-<version>.jar`
(~15 MB), not `original-*.jar`. Appose drags in groovy/ivy/jna/commons-* that a
stock Fiji does not ship, and the shade plugin folds them in so a user installs
one self-contained file instead of copying jars into `Fiji.app/jars/`.

Delete `target/` by hand rather than relying on `mvn clean`: a plain `package`
copies over a stale `target/classes`, which is how `scripts/__pycache__/*.pyc`
once ended up inside a shipped jar.

Bump `BUILD_ID` in `NeuralClassifyPlugin.java` on every jar you hand over. It is
shown in the dialog title and the first Log line, and it is the only way to tell
a bug from a stale class — Fiji keeps a plugin's classes loaded until it
*restarts*, so "I copied the new jar in" and "Fiji is running the new jar" are
different claims.

### On the HPC login node

`mvn` is not on the default `PATH` (it lives in the `neural_env` conda env),
that env's own JDK is 17, and there is no outbound Maven access:

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
rm -rf target
PATH="$JAVA_HOME/bin:$HOME/miniconda3/envs/neural_env/bin:$PATH" \
  mvn -o -B -DskipTests package
```

`-o` (offline) works because `~/.m2` is already populated.

## Publishing a model bundle

`ModelBootstrap` pins three things about the asset: its URL (from
`BUNDLE_VERSION`), its SHA-256, and its byte size. The script prints the last
two; they are copied, never hand-edited.

```bash
python tools/prepare_model_bundle.py \
    --src /path/to/classifier_raw/models \
    --out dist/models.zip
```

It keeps only what inference reads — a training checkpoint is 134 MB, of which
45 MB is the network; the optimizer, scheduler and history exist to *resume*
training and are never loaded by `load_models`. Weights are copied
tensor-for-tensor rather than re-saved from a rebuilt model, so the bundle
cannot silently disagree with the checkpoint it came from. Verify that if you
ever change the script:

```bash
PYTHONPATH=python/src python -c "
import torch, sys
from pathlib import Path
from neural_imgs.inference.fixed_model_pipeline import MARKERS, prod_exp_name
orig, new = Path(sys.argv[1]), Path(sys.argv[2])
for m in MARKERS:
    a = torch.load(orig/prod_exp_name(m)/'best_model.pt', map_location='cpu', weights_only=False)['model_state_dict']
    b = torch.load(new/prod_exp_name(m)/'best_model.pt', map_location='cpu', weights_only=False)['model_state_dict']
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), m
    print(m, 'bit-identical')
" /path/to/classifier_raw/models /path/to/extracted/bundle
```

Then paste the printed constants into `ModelBootstrap.java`, rebuild the jar,
and follow **[`PUBLISHING.md`](PUBLISHING.md)** — the browser route, which needs
no CLI tooling or access token.

Whichever route you take, the invariant is the same: the jar hardcodes the
bundle's URL and SHA-256, so **the model release must be published and public
before the jar that points at it**, or every first run dies on `HTTP 404`.
`tools/check-model-asset.sh` proves that with nothing but `curl` — it fetches
the asset the way a biologist's Fiji will (anonymously, through GitHub's
redirect) and checks it against the constants compiled into the jar.

If you have [`gh`](https://cli.github.com) installed,
`tools/publish-release.sh v1.0.0` does all of it in one shot: it reads
`BUNDLE_VERSION` and `BUNDLE_SHA256` back out of the Java source rather than
taking them as arguments, so it cannot publish a bundle under a version the jar
does not request or a zip whose hash the jar will reject, and it refuses to
publish the jar until the bundle is downloadable. `gh` is not installed on the
HPC node, so this is the optional path, not the default one.

Two tags, not one, because the artefacts change at different rates: a jar fix
that does not touch the weights should not make every user re-download 90 MB,
and a retrained model should not require a new jar unless `BUNDLE_VERSION`
changes.

### Changing which checkpoints ship

The bundle's directory names are not free-form: Python derives them from
`PROD_CONFIG` via `prod_exp_name()` in `fixed_model_pipeline.py`, and
`ModelBootstrap.REQUIRED` lists the same paths so the plugin can say "the OPC
checkpoint is missing" before starting a worker that would take a minute to
reach the same conclusion. Change `PROD_CONFIG` and all three must move
together: the config, `REQUIRED`, and a new `BUNDLE_VERSION`.

## Testing the model download without GitHub

`ModelBootstrap` reads the system property `neuralimgs.models.url`, which exists
for internal mirrors and makes the download path testable against a local
server. The integrity check is not relaxed: a mirror must serve the same bytes.

`tools/manual-test/` uses that to drive the whole install path against a
throwaway localhost server -- see its
[README](../tools/manual-test/README.md) for the four failure modes it covers
and how to run it. Worth doing after any change to `ModelBootstrap`, because
none of them are reachable until a user's first run: a redirect chain, a
truncated body, an interrupted download leaving a partial directory, and a
second call that must *not* re-download.

An install can also be pointed at a mirror by hand, e.g. to reproduce a
proxied site:

```bash
# Linux/macOS Fiji: pass it through to the JVM
./Fiji.app/ImageJ-linux64 -Dneuralimgs.models.url=http://127.0.0.1:8000/models.zip
```
