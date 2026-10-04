# Publishing

Two artefacts reach users, and **the order matters**:

| # | Artefact | Where | Size |
| --- | --- | --- | --- |
| 1 | `models.zip` | release tagged `models-v1` | ~90 MB |
| 2 | `Neural_Image_Classifier-1.0.0.jar` | release tagged `v1.0.0` | ~15 MB |

The jar hardcodes the model bundle's URL and SHA-256 (`ModelBootstrap.java`), so
**the model release must exist and be public before the jar is published.**
Otherwise every biologist's first run dies with `HTTP 404` — a failure none of
them can diagnose or work around.

This page is the browser route: no `gh`, no CLI tooling, no access token.
`tools/publish-release.sh` automates the same thing if you ever do install
[`gh`](https://cli.github.com); the steps below are the authority either way.

---

## Step 0 — build the two files

On the machine with the checkpoints (e.g. the HPC node):

```bash
# the model bundle -> dist/models.zip
python tools/prepare_model_bundle.py \
    --src /path/to/classifier_raw/models \
    --out dist/models.zip

# the jar -> target/Neural_Image_Classifier-1.0.0.jar
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
rm -rf target        # not `mvn clean` -- see DEVELOPING.md
PATH="$JAVA_HOME/bin:$HOME/miniconda3/envs/neural_env/bin:$PATH" \
  mvn -o -B -DskipTests package
```

`prepare_model_bundle.py` prints a SHA-256 and a byte count. Those two values
**must** already be in `ModelBootstrap.java` before you build the jar — if you
changed the bundle, paste them in and rebuild. To check they are in step:

```bash
sha256sum dist/models.zip
grep -A1 'BUNDLE_SHA256 =' src/main/java/com/neuralimgs/fiji/ModelBootstrap.java
```

### Getting the files to your laptop

The browser upload happens from your laptop, so pull both files out of the
remote first. In VS Code's Explorer, right-click → **Download**:

- `dist/models.zip`
- `target/Neural_Image_Classifier-1.0.0.jar`

(Or `scp`/`rsync` them, whichever you normally use.)

---

## Step 1 — create the repository

In a browser: <https://github.com/new>

- **Owner / name:** `Amirhk-dev` / `neural-cell-classifier-fiji`
- **Visibility:** **Public** — the plugin downloads its models anonymously, so a
  private repo means the download 404s for everyone.
- Do **not** add a README, .gitignore or licence — the repo already has all
  three and GitHub's would collide with the first push.

Click **Create repository**, then leave the page open: the next step needs the
URL it shows you.

---

## Step 2 — push the code

The repository is already committed locally; it just has no remote yet. From
the repository root, in **your own terminal** (the one whose Git credential
helper works):

```bash
cd /lustre/groups/aih/amirhossein.kardoost/codes/github/neural-cell-classifier-fiji
git remote add origin https://github.com/Amirhk-dev/neural-cell-classifier-fiji.git
git push -u origin main
```

If the push asks for a password, that is Git wanting a **personal access token**
rather than your account password — GitHub stopped accepting passwords over
HTTPS. Either let the VS Code credential helper handle it (it usually does), or
create a token at
<https://github.com/settings/tokens> with the `repo` scope and let your
credential helper store it. Never paste a token into a chat or a committed file.

Check the push landed: the repo page should show 43 files and the README
rendered.

---

## Step 3 — publish the model bundle (release 1 of 2)

In the browser, on the new repo: **Releases** → **Create a new release**
(or go straight to
`https://github.com/Amirhk-dev/neural-cell-classifier-fiji/releases/new`).

| Field | Value |
| --- | --- |
| **Tag** | `models-v1` — type it in and pick *"Create new tag: models-v1 on publish"* |
| **Target** | `main` |
| **Title** | `Model bundle v1` |
| **Attach files** | drag in **`models.zip`** |

> The tag must be **exactly** `models-v1` and the file must be named **exactly**
> `models.zip`. Both are compiled into the jar's download URL — a tag of
> `models-v1.0` or a file renamed to `models-v1.zip` by your browser produces a
> 404 for every user.

Description (optional, but useful):

```
OPC + B3-Tub production checkpoints and thresholds.json.

Downloaded automatically by the plugin on first run — there is nothing to do
with this file by hand.

SHA-256: 0aabef9b4ebb37b7baa7859eec4a975771ce1e657884490a7a0749c9f9794335
```

Leave **Set as a pre-release** unchecked, and click **Publish release**. Wait
for the upload to finish — a 90 MB attachment takes a moment, and the release
is not usable until it shows up under *Assets*.

### Verify it before going further

Back on the remote, from the repository root:

```bash
tools/check-model-asset.sh --full
```

This needs nothing but `curl`. It fetches the asset **exactly the way a
biologist's Fiji will** — anonymously, following GitHub's redirect to its asset
CDN — and checks the size and SHA-256 against the constants compiled into the
jar. It must print `OK` before you publish the jar. Drop `--full` for a
seconds-long reachability-and-size check that skips the 90 MB download.

If it reports `HTTP 404`, the usual causes in order: the repo is still private,
the tag is not exactly `models-v1`, or the asset is not named exactly
`models.zip`.

---

## Step 4 — publish the plugin (release 2 of 2)

Only once Step 3 verified `OK`. Same flow, new release:

| Field | Value |
| --- | --- |
| **Tag** | `v1.0.0` |
| **Target** | `main` |
| **Title** | `v1.0.0` |
| **Attach files** | drag in **`Neural_Image_Classifier-1.0.0.jar`** |

Description — this is what biologists read, so keep it to the install:

```
Install: download the .jar below, then in Fiji use `Plugins > Install...`
(or drop it into `Fiji.app/plugins/`) and restart Fiji.

The first run downloads its own Python environment and the trained models and
caches both — there is nothing else to install and no model files to copy.

See the README for the full guide: https://github.com/Amirhk-dev/neural-cell-classifier-fiji#readme
```

Tick **Set as the latest release** (the README's install link points at
`/releases/latest`), then **Publish release**.

---

## Step 5 — check it the way a biologist will

Worth doing once, on a machine that is *not* the development box:

1. Open `https://github.com/Amirhk-dev/neural-cell-classifier-fiji/releases/latest`
   in a browser where you are **logged out** of GitHub. If you can download the
   jar, so can they. (A logged-in browser can mask a repo that is still
   private — the single most common way this breaks.)
2. Install the jar into a Fiji that has never run the plugin, and start
   `Plugins > Neural Image Processor > Install / Verify Models...`. It should
   download the bundle and report success without you typing a path. That
   exercises the whole first-run path — release URL, redirect, hash check,
   extraction — in the one place it has to work.

---

## Later releases

- **Code or jar change only:** rebuild the jar, bump `BUILD_ID` in
  `NeuralClassifyPlugin.java` and the `<version>` in `pom.xml`, publish a new
  `vX.Y.Z` release. Leave `models-v1` alone — users keep the models they already
  downloaded, and nobody re-downloads 90 MB for a UI fix.
- **New or retrained checkpoints:** rebuild the bundle, bump `BUNDLE_VERSION` to
  `models-v2` **and** paste the new SHA-256 and byte count into
  `ModelBootstrap.java`, then run through Steps 3–4 again. The version bump is
  what makes existing installs fetch the new weights instead of keeping the old
  ones, since the completeness check only asks whether files are present.

Two tags rather than one, because the artefacts change at different rates.
