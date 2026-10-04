"""Model definition and single-cell preprocessing -- the inference-side subset.

The full neural-image-processor `neural_imgs.training` package also exports the
training loop (`CellClassifierTrainer`), the evaluator, the split manager and
the augmentation pipeline. None of those are reachable from the Fiji plugin, and
each drags in a dependency (albumentations, scikit-learn) that inference never
needs, so this public inference-only distribution ships just the two modules the
pipeline imports: the network itself and the crop -> tensor preprocessing.

What is here is byte-identical to the private repo, which is the point: the
preprocessing applied at inference MUST be the preprocessing the checkpoints
were trained under, so these files are vendored rather than reimplemented.
"""

from .model import CellClassifier, create_resnet18_classifier, get_model_for_channel_mode
from .raw_dataset import RawCropConfig, raw_cell_to_tensor

__all__ = [
    "CellClassifier",
    "RawCropConfig",
    "create_resnet18_classifier",
    "get_model_for_channel_mode",
    "raw_cell_to_tensor",
]
