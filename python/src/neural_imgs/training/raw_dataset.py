"""Dataset for native-resolution single-cell classification.

Unlike :class:`CellClassifierDataset` (which reads 512x512 cells from the large
``*_cells.pkl`` files), this dataset reads the pre-built RAW single-cell store
produced by ``neural_imgs.extraction.raw_cell_dataset`` — one ``.npz`` per cell
holding ``image (5,H,W) uint16`` (DAPI, OPC, RFP, B3-Tub, BF), ``mask (H,W) uint8``
(the DAPI segmentation) and ``bb``. The fold CSVs point at each cell through a
``cell_file`` column and carry a 0/1 ``label``.

Cells keep their NATIVE pixel scale — nothing is resized. A fixed ``crop_size``
window is cut out centered on the DAPI-nucleus centroid; where the window exceeds
the cell it is zero-padded, and where the cell is larger the periphery is cropped.
This makes ``crop_size`` a field-of-view knob (nucleus-only -> whole-cell) at a
constant physical pixel size.

The DAPI mask can be used two ways (independently): appended as an extra binary
input channel (``include_mask``) so the model can tell which pixels belong to
*this* cell's nucleus, and/or used to soft-mask the image channels
(``apply_soft_mask``) so signal from neighbouring cells in a crowded field is
attenuated. The soft mask feathers outward from the nucleus (Gaussian falloff)
so the target cell's own cytoplasm/neurite signal (e.g. B3-Tub) is preserved
near the nucleus rather than hard-cut at the nuclear boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CHANNEL_NAMES = ["DAPI", "OPC", "RFP", "B3-Tub", "BF"]


@dataclass
class RawCropConfig:
    """Configuration for how each raw cell is turned into a model input.

    Parameters
    ----------
    crop_size:
        Side length of the square window (or ``(height, width)``) cut out around
        the DAPI-nucleus centroid, in native pixels. Larger cells are cropped,
        smaller cells zero-padded; never resized.
    include_mask:
        If True, append the binary DAPI mask as a final input channel.
    channels:
        Indices into ``CHANNEL_NAMES`` to keep (default: all five).
    apply_soft_mask:
        If True, multiply the image channels by a *soft* (Gaussian-feathered)
        version of this cell's DAPI mask so signal from neighbouring cells in a
        crowded field is suppressed while the target cell's own signal (nucleus +
        a decaying cytoplasm/neurite ring) is kept. See :func:`soft_mask_weights`.
    mask_sigma:
        Gaussian sigma (native pixels) for the soft-mask falloff. Larger values
        keep a wider ring of surrounding signal before it decays to zero.
    augment:
        If True, apply train-time augmentation (geometric flips/rot90 on all
        channels jointly + light intensity jitter on the image channels only).
    stretch_percentiles:
        Low/high percentiles used for per-channel contrast stretch to [0, 1].
    """

    crop_size: int | tuple[int, int] = 272
    include_mask: bool = True
    channels: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    apply_soft_mask: bool = False
    mask_sigma: float = 8.0
    augment: bool = False
    stretch_percentiles: tuple[float, float] = (1.0, 99.0)

    @property
    def hw(self) -> tuple[int, int]:
        if isinstance(self.crop_size, (tuple, list)):
            return int(self.crop_size[0]), int(self.crop_size[1])
        return int(self.crop_size), int(self.crop_size)

    @property
    def n_channels(self) -> int:
        return len(self.channels) + (1 if self.include_mask else 0)


# ---------------------------------------------------------------------------
# Preprocessing (shared by the Dataset and by inference notebooks/scripts)
# ---------------------------------------------------------------------------

def centroid_of_mask(mask: np.ndarray) -> tuple[int, int]:
    """Center of mass of the DAPI mask; falls back to the image center."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return mask.shape[0] // 2, mask.shape[1] // 2
    return int(round(ys.mean())), int(round(xs.mean()))


def center_window(arr: np.ndarray, cy: int, cx: int, h: int, w: int) -> np.ndarray:
    """Cut an ``h x w`` window from ``arr`` (C, H, W) centered on (cy, cx).

    Out-of-bounds regions are zero-padded; this handles both crop (cell larger
    than the window) and pad (cell smaller) in a single operation.
    """
    c, H, W = arr.shape
    out = np.zeros((c, h, w), dtype=arr.dtype)
    y0 = cy - h // 2
    x0 = cx - w // 2
    # Source region clipped to the array; destination shifted by the same clip.
    sy0, sy1 = max(0, y0), min(H, y0 + h)
    sx0, sx1 = max(0, x0), min(W, x0 + w)
    dy0, dx0 = sy0 - y0, sx0 - x0
    out[:, dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = arr[:, sy0:sy1, sx0:sx1]
    return out


def soft_mask_weights(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Soft (Gaussian-feathered) weights in [0, 1] from a binary DAPI mask.

    The segmented nucleus keeps full weight (1.0) and the weight decays smoothly
    outside it, so a target cell's own cytoplasm/neurite signal survives near the
    nucleus while distant neighbouring cells are attenuated toward zero. Used to
    focus the classifier on *this* segmented cell in crowded fields.

    ``sigma`` is the Gaussian falloff in native pixels; ``sigma <= 0`` reduces to
    the hard binary mask.
    """
    from scipy.ndimage import gaussian_filter

    m = (mask > 0).astype(np.float32)
    if sigma <= 0 or m.max() == 0:
        return m
    soft = gaussian_filter(m, sigma=sigma)
    soft /= soft.max()               # normalize the peak (nucleus center) to 1
    return np.maximum(soft, m)       # guarantee the nucleus itself stays at 1


def _stretch(image: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    """Per-channel percentile contrast stretch to [0, 1]."""
    lo_p, hi_p = percentiles
    out = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        ch = image[c]
        a, b = np.percentile(ch, lo_p), np.percentile(ch, hi_p)
        out[c] = np.clip((ch - a) / (b - a), 0, 1) if b > a else 0.0
    return out


def _augment(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Geometric aug applied jointly; intensity jitter on image channels only."""
    if rng.random() < 0.5:  # horizontal flip
        image = image[:, :, ::-1]
        mask = mask[:, ::-1]
    if rng.random() < 0.5:  # vertical flip
        image = image[:, ::-1, :]
        mask = mask[::-1, :]
    k = int(rng.integers(0, 4))  # 0/90/180/270
    if k:
        image = np.rot90(image, k=k, axes=(1, 2))
        mask = np.rot90(mask, k=k, axes=(0, 1))
    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)
    # Intensity jitter (image channels only; mask must stay binary).
    if rng.random() < 0.5:
        gain = 1.0 + rng.uniform(-0.2, 0.2)      # contrast
        bias = rng.uniform(-0.1, 0.1)            # brightness
        image = np.clip(image * gain + bias, 0, 1)
    return image, mask


def raw_cell_to_tensor(
    image: np.ndarray,
    mask: np.ndarray,
    config: RawCropConfig,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Turn a native ``(C,H,W)`` cell image + ``(H,W)`` DAPI mask into a model input.

    This is the single source of truth for the classifier's preprocessing — the
    training :class:`RawCellClassifierDataset` and any inference code (e.g. the
    production notebook) both call it, so they cannot drift apart. Steps: window a
    ``config.crop_size`` region centered on the DAPI-mask centroid (zero-padded when
    it exceeds the cell), select ``config.channels``, per-channel percentile stretch
    to [0, 1], optionally augment, normalize to ~[-1, 1], and append the binary mask
    as an extra channel when ``config.include_mask``.

    Returns a float32 tensor of shape ``(config.n_channels, H', W')``.
    """
    cy, cx = centroid_of_mask(mask)
    h, w = config.hw
    image = center_window(image, cy, cx, h, w)          # (C, h, w)
    mask = center_window(mask[None], cy, cx, h, w)[0]    # (h, w)

    image = image[config.channels].astype(np.float32)
    image = _stretch(image, config.stretch_percentiles)  # -> [0, 1] per channel

    if config.augment:
        image, mask = _augment(image, mask, rng or np.random.default_rng())

    # Focus on this segmented cell: attenuate signal from neighbouring cells by
    # weighting the image channels with a soft version of the DAPI mask. Done in
    # [0, 1] space so suppressed pixels reach 0 and map to the background (-1)
    # after normalization, matching the zero-padded periphery.
    if config.apply_soft_mask:
        image = image * soft_mask_weights(mask, config.mask_sigma)[None]

    # Normalize image channels to ~[-1, 1]; keep mask as a {0, 1} indicator.
    image = (image - 0.5) / 0.5
    if config.include_mask:
        mask_ch = (mask > 0).astype(np.float32)[None]
        arr = np.concatenate([image, mask_ch], axis=0)
    else:
        arr = image
    return torch.from_numpy(np.ascontiguousarray(arr))


class RawCellClassifierDataset(Dataset):
    """Native-resolution single-cell dataset backed by the raw ``.npz`` store.

    Parameters
    ----------
    csv_path:
        Fold CSV (``{marker}_fold{k}_{split}.csv``) with a ``cell_file`` column
        (absolute path to each cell's ``.npz``) and a ``label`` column.
    label_column:
        Column holding the 0/1 label.
    config:
        :class:`RawCropConfig` controlling crop size, mask channel and augmentation.
    seed:
        RNG seed for augmentation (per-sample randomness stays reproducible run to
        run only in the sense of the initial seed; DataLoader shuffling is separate).
    """

    def __init__(
        self,
        csv_path: str | Path,
        label_column: str = "label",
        config: RawCropConfig | None = None,
        seed: int = 42,
    ):
        self.csv_path = Path(csv_path)
        self.label_column = label_column
        self.config = config or RawCropConfig()
        self._rng = np.random.default_rng(seed)

        df = pd.read_csv(self.csv_path)
        valid = df[label_column].notna() & (df[label_column] != -1.0)
        self.df = df[valid].reset_index(drop=True)

        # Small store (~330 cells); cache arrays in memory on first access.
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # -- torch Dataset API -------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image, mask = self._load(row["cell_file"])
        tensor = raw_cell_to_tensor(image, mask, self.config, rng=self._rng)
        label = int(row[self.label_column])
        return tensor, label

    def _load(self, cell_file: str) -> tuple[np.ndarray, np.ndarray]:
        if cell_file not in self._cache:
            d = np.load(cell_file)
            self._cache[cell_file] = (d["image"], d["mask"])
        return self._cache[cell_file]

    # -- interface expected by Trainer / Evaluator ------------------------
    def get_labels(self) -> np.ndarray:
        return self.df[self.label_column].values.astype(int)

    def get_class_weights(self) -> torch.Tensor:
        labels = self.get_labels()
        counts = np.bincount(labels, minlength=2).astype(float)
        weights = np.where(counts > 0, len(labels) / (2 * counts), 0.0)
        return torch.tensor(weights, dtype=torch.float32)

    def get_class_distribution(self) -> dict[str, int]:
        labels = self.get_labels()
        return {
            "negative (0)": int((labels == 0).sum()),
            "positive (1)": int((labels == 1).sum()),
        }

    def get_sample_info(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "image_filename": row.get("image_filename", ""),
            "cell_file": row["cell_file"],
            "pkl_file": row.get("pkl_file", ""),
            "cell_idx_in_file": int(row.get("cell_idx_in_file", -1)),
            "label": int(row[self.label_column]),
        }

    @property
    def n_channels(self) -> int:
        return self.config.n_channels
