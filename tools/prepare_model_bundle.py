#!/usr/bin/env python
"""Build the ``models.zip`` release asset the Fiji plugin downloads on first run.

Takes the trained checkpoint directories from wherever training left them and
writes a bundle holding only what inference needs:

    models.zip
      thresholds.json
      raw_OPC_resnet18_crop200_soft32_ch1_prod/best_model.pt
      raw_B3Tub_resnet18_crop200_soft16_ch3_prod/best_model.pt

Two things are stripped on the way:

* **Optimizer, scheduler and training history.** A training checkpoint is
  134 MB, of which 45 MB is the network. The rest exists to *resume* training
  and is never read by ``load_models``, so shipping it would triple what every
  biologist downloads for no change in any number the plugin produces.
* **Nothing else.** The weights are copied tensor-for-tensor, not re-saved from
  a rebuilt model, so the bundle cannot silently disagree with the private
  checkpoint it came from.

The directory names are not chosen here: Python derives them from ``PROD_CONFIG``
via ``prod_exp_name()``, and this script imports that function rather than
hardcoding the strings, so a production config change cannot produce a bundle
whose layout the pipeline will not find.

Run it, then attach the zip to a GitHub release and paste the printed SHA-256
and byte count into ``ModelBootstrap.java``.

Usage:
    python tools/prepare_model_bundle.py \\
        --src /path/to/classifier_raw/models \\
        --out dist/models.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
# Import the production config from the package being shipped, so "which
# checkpoint backs which marker" has exactly one definition.
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))

from neural_imgs.inference.fixed_model_pipeline import (  # noqa: E402
    MARKERS,
    prod_exp_name,
)

# Kept out of the bundle. Listed rather than inferred so that a new key added to
# a future checkpoint is kept by default and shows up in the printed summary --
# the failure to avoid is silently dropping something inference needs.
TRAINING_ONLY_KEYS = (
    "optimizer_state_dict",
    "scheduler_state_dict",
    "history",
)


def strip_checkpoint(src: Path, dst: Path) -> tuple[int, int]:
    """Copy ``src`` to ``dst`` keeping only what inference reads.

    Returns ``(src_bytes, dst_bytes)``.
    """
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise SystemExit(f"{src}: not a training checkpoint (no model_state_dict)")

    kept = {k: v for k, v in ckpt.items() if k not in TRAINING_ONLY_KEYS}
    dropped = sorted(set(ckpt) - set(kept))
    # best_val_loss / best_val_acc are a few bytes and are what lets someone
    # holding only the bundle say how the shipped model scored, so they stay.
    print(f"    kept:    {sorted(kept)}")
    print(f"    dropped: {dropped}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kept, dst)
    return src.stat().st_size, dst.stat().st_size


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--src", required=True, type=Path,
        help="Directory holding thresholds.json and the per-marker checkpoint dirs",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "dist" / "models.zip",
        help="Zip to write (default: dist/models.zip)",
    )
    args = parser.parse_args()

    src: Path = args.src
    thresholds_path = src / "thresholds.json"
    if not thresholds_path.is_file():
        raise SystemExit(f"No thresholds.json in {src}")

    thresholds = json.loads(thresholds_path.read_text())
    missing = [m for m in MARKERS if m not in thresholds]
    if missing:
        raise SystemExit(f"thresholds.json has no entry for: {missing}")
    print(f"thresholds.json: {thresholds}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        shutil.copy2(thresholds_path, staging / "thresholds.json")

        for marker in MARKERS:
            name = prod_exp_name(marker)
            source = src / name / "best_model.pt"
            if not source.is_file():
                raise SystemExit(f"Missing checkpoint for {marker}: {source}")
            print(f"{marker}: {name}")
            before, after = strip_checkpoint(source, staging / name / "best_model.pt")
            print(f"    {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")

        # ZIP_STORED, not DEFLATE: a .pt file is already packed float32 and
        # compresses by a couple of percent, so deflating costs every installing
        # machine a decompression pass to save almost nothing.
        with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as zf:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(staging).as_posix())

    size = args.out.stat().st_size
    print()
    print(f"wrote {args.out}  ({size / 1e6:.1f} MB)")
    print()
    print("Paste into src/main/java/com/neuralimgs/fiji/ModelBootstrap.java:")
    print(f'    BUNDLE_SHA256 = "{sha256(args.out)}";')
    print(f"    BUNDLE_BYTES  = {size:_}L;")
    print()
    print("Then attach it to a GitHub release tagged to match BUNDLE_VERSION, e.g.")
    print("    gh release create models-v1 dist/models.zip \\")
    print('        --title "Model bundle v1" --notes "OPC + B3-Tub checkpoints"')


if __name__ == "__main__":
    main()
