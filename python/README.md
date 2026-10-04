# neural-cell-classifier (Python side)

The inference pipeline the [Neural Cell Classifier Fiji
plugin](https://github.com/Amirhk-dev/neural-cell-classifier-fiji) drives:
CZI → tile → BaSiC illumination correction → percentile normalisation →
CellPose nucleus detection → per-marker ResNet-18 classification (OPC, B3-Tub)
→ RFP intensity rule.

Installed automatically by the plugin into its own isolated environment. You do
not need to install it by hand to use the plugin — this is here for anyone who
wants to drive the same pipeline from a script.

```bash
pip install "git+https://github.com/Amirhk-dev/neural-cell-classifier-fiji.git#subdirectory=python"
```

```python
from neural_imgs.inference.batch import BatchClassifier, BatchInput
from neural_imgs.inference.fixed_model_pipeline import load_models
from neural_imgs.inference.rfp import RfpConfig

models = load_models("~/.neural-imgs/models")       # the bundle the plugin downloads
result = BatchClassifier(models).run(BatchInput(
    image_paths=["image.czi"],
    out_root="out",
    model_dir="~/.neural-imgs/models",
    patch_grid=4,
    rfp=RfpConfig(),                                 # neighbour rule, r=60, gate 16.5
))
print("\n".join(result.summary_lines()))
```

The import package is `neural_imgs` and its modules are byte-identical to the
private research repository they come from — the preprocessing applied at
inference has to be the preprocessing the shipped checkpoints were trained
under. Only the inference closure is shipped; the training loop, evaluator and
augmentation pipeline are not part of this distribution. See
[`docs/DEVELOPING.md`](../docs/DEVELOPING.md).

MIT licensed.
