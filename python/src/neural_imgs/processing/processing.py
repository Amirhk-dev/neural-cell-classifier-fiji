import numpy as np
from basicpy import BaSiC
from skimage.transform import resize
from skimage.morphology import binary_dilation, disk

from tqdm import tqdm


def squeeze_img(img):
    return np.squeeze(img)


def patch_image(img, patch_size=4):
    assert img.ndim == 3, "Expected (C, H, W)"
    C, H, W = img.shape
    assert C == 5, f"Expected 5 channels, got {C}"
    n_rows, n_cols = patch_size, patch_size
    H_patch = H // n_rows
    W_patch = W // n_cols
    patches = []
    for i in range(n_rows):
        for j in range(n_cols):
            patch = img[:,
                        i * H_patch:(i + 1) * H_patch,
                        j * W_patch:(j + 1) * W_patch]
            patches.append(patch)
    patches = np.stack(patches, axis=0)  # (16, 5, H_patch, W_patch)
    return patches


def fit_illumination_profiles(patches: np.ndarray, max_iterations: int) -> list:
    """Fit one BaSiC illumination model per channel from a stack of patches.

    Args:
        patches: array of shape (N, C, H, W) — N patches, C channels.
        max_iterations: BaSiC optimisation iterations.

    Returns:
        List of C fitted BaSiC models, one per channel.
    """
    n_channels = patches.shape[1]
    basics = []
    for c in range(n_channels):
        b = BaSiC(max_iterations=max_iterations)
        b.fit(patches[:, c, :, :])  # (N, H, W)
        basics.append(b)
    return basics


def correct_illumination(patch: np.ndarray, basics: list) -> np.ndarray:
    """Apply per-channel illumination correction to a single patch.

    Args:
        patch: array of shape (C, H, W).
        basics: list of C fitted BaSiC models from fit_illumination_profiles.

    Returns:
        Corrected array of shape (C, H, W).
    """
    corrected = np.stack([
        basics[c].transform(patch[c:c + 1], use_tqdm=False)[0]
        for c in range(patch.shape[0])
    ], axis=0)
    return corrected


def percentile_normalization(img, p_low, p_high, eps=1e-8):
    print('assuming dimension 0 is channels')
    for i in range(img.shape[0]):
        lo = np.percentile(img[i], p_low)
        hi = np.percentile(img[i], p_high)
        if hi <= lo:
            hi = img[i].max()
            lo = img[i].min()
        img[i] = np.clip(img[i], lo, hi)
        img[i] = (img[i] - lo) / (hi - lo + eps)
    return img


def extract_cells_512_nodistort(img, masks, pad=20, min_area=0,
                                boundary_margin=200, target=512):
    C, H, W = img.shape
    out = []

    for lab in np.unique(masks):
        if lab == 0:
            continue

        ys, xs = np.where(masks == lab)
        if ys.size < min_area:
            continue

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        if (y_min < boundary_margin or y_max > H - boundary_margin or
                x_min < boundary_margin or x_max > W - boundary_margin):
            continue

        y0 = max(y_min - pad, 0)
        y1 = min(y_max + pad + 1, H)
        x0 = max(x_min - pad, 0)
        x1 = min(x_max + pad + 1, W)
        labels_in_bb = np.unique(masks[y0:y1, x0:x1])
        n_cells_in_region = len(labels_in_bb[labels_in_bb != 0])

        img_crop = img[:, y0:y1, x0:x1].copy()
        mask_crop = (masks[y0:y1, x0:x1] == lab).astype(np.uint8)
        h, w = mask_crop.shape
        s = target / float(min(h, w))
        new_h, new_w = max(1, int(round(h * s))), max(1, int(round(w * s)))
        img_resized = np.stack([
            resize(img_crop[c], (new_h, new_w), order=1,
                   anti_aliasing=True, preserve_range=True)
            for c in range(C)
        ], axis=0).astype(img.dtype)
        mask_resized = resize(mask_crop, (new_h, new_w), order=0,
                              anti_aliasing=False,
                              preserve_range=True).astype(np.uint8)
        y_start = (new_h - target) // 2
        x_start = (new_w - target) // 2
        y_end, x_end = y_start + target, x_start + target
        img_final = img_resized[:, y_start:y_end, x_start:x_end]
        mask_final = mask_resized[y_start:y_end, x_start:x_end]
        img_final[0] *= mask_final
        out.append(dict(label=lab, img=img_final, mask=mask_final,
                        bb_position=[y0, y1, x0, x1],
                        n_cells_in_region=n_cells_in_region))

    return out


def detect_cells_on_DAPI(img, model, batch_size,
                         flow_threshold,
                         cellprob_threshold):
    masks, flows, styles = model.eval(
        img,
        batch_size=batch_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold
    )
    return masks, flows, styles


def zero_out_image_border(img, border=50):
    """
    img: np.ndarray of shape (C, H, W)
    Sets a border of size 'border' to zero on all sides for each channel.
    """
    C, H, W = img.shape
    assert H > 2 * border and W > 2 * border, "Border too large for image size."

    img[:, :border, :] = 0
    img[:, H - border:, :] = 0
    img[:, :, :border] = 0
    img[:, :, W - border:] = 0

    return img


def analyze_image(img,
                  img_mask,
                  channel_names,
                  target_channels,
                  dilation_radius=3,
                  percent=95,
                  eps=0.1):
    assert img.shape[0] == 5, "Expected 5 channels"

    percentile = {}
    for c in target_channels:
        if img[c].min() < -1e-6 or img[c].max() > 1 + 1e-6:
            raise ValueError(
                f"Image intensities must be between 0 and 1, got min={img[c].min()}, max={img[c].max()}"
            )
        percentile[channel_names[c]] = np.percentile(img[c], percent)

    mask = img_mask > 0
    selem = disk(dilation_radius)
    dilated_mask = binary_dilation(mask, selem)

    masked_means = {}
    for c in target_channels:
        ch_name = channel_names[c]
        vals = img[c][dilated_mask]
        masked_means[ch_name] = vals.mean() if vals.size > 0 else np.nan

    positive = {}
    confidence = {}
    score = {}
    for c in target_channels:
        ch_name = channel_names[c]
        ch_mean = masked_means[ch_name]
        ch_percentile = percentile[ch_name]

        # The raw separation between the cell and its local background, kept for
        # every cell whether or not it clears the gate. "confidence(%)" below is
        # this same number censored to positives, which makes it useless for
        # calibration: a threshold can only be moved down if the cells currently
        # below it still carry a score. Signed, so negatives are ordered too.
        score[ch_name] = float("nan") if np.isnan(ch_mean) else (ch_mean - ch_percentile) * 100

        if np.isnan(ch_mean):
            positive[ch_name] = False
            confidence[ch_name] = None
        else:
            positive[ch_name] = ch_mean > (ch_percentile + eps)
            confidence[ch_name] = (ch_mean - ch_percentile) * 100 if positive[ch_name] else None

    return {
        "percentile": percentile,
        "masked_means": masked_means,
        "positive": positive,
        "confidence(%)": confidence,
        "score(%)": score,
        "dilated_mask": dilated_mask,
    }


def process_one_patch(img, basics, model, channel_names, plot=False):
    img = correct_illumination(img, basics)
    p_low = 0.1
    p_high = 99.9
    eps = 0.00000001
    img = percentile_normalization(img, p_low, p_high, eps)
    flow_threshold = 0.09
    cellprob_threshold = 0.6
    batch_size = 1
    masks, flows, styles = detect_cells_on_DAPI(
        img[0], model, batch_size, flow_threshold, cellprob_threshold
    )

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img[0], cmap='gray')
        axes[0].set_title('DAPI channel')
        axes[0].axis('off')
        axes[1].imshow(masks, cmap='nipy_spectral', interpolation='nearest')
        axes[1].set_title(f'Cellpose masks ({masks.max()} cells)')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

    single_cells = extract_cells_512_nodistort(
        img, masks, pad=100, min_area=100, boundary_margin=20, target=512
    )
    percentile = 95
    border = 100
    target_channels = [1, 2, 3]
    for idx in tqdm(range(len(single_cells)), desc='Processing single cells'):
        single_cells[idx]['img'] = zero_out_image_border(
            single_cells[idx]['img'],
            border=border
        )
        single_cells[idx]['result'] = analyze_image(
            single_cells[idx]['img'],
            single_cells[idx]['mask'],
            channel_names,
            target_channels,
            percent=percentile
        )
    return single_cells


def get_reprogramming_quantification(single_cells, confidence_threshold,
                                     channel_names):
    from neural_imgs.visualization.visualization import (
        get_and_show_positive_in_channel,
        get_and_show_negative_cells,
    )
    total_cells_detected = len(single_cells)
    print(f'number of single cells: {total_cells_detected}')
    channel_idx = 1  # OPC
    num_opc_positive, opc_positive_idx = get_and_show_positive_in_channel(
        None, single_cells, channel_names=channel_names,
        channel_idx=channel_idx, confidence_threshold=confidence_threshold,
        n_channels=5, plot=False
    )
    channel_idx = 2  # RFP
    num_rfp_positive, rfp_positive_idx = get_and_show_positive_in_channel(
        None, single_cells, channel_names=channel_names,
        channel_idx=channel_idx, confidence_threshold=confidence_threshold,
        n_channels=5, plot=False
    )
    channel_idx = 3  # B3Tub
    num_b3tub_positive, b3tub_positive_idx = get_and_show_positive_in_channel(
        None, single_cells, channel_names=channel_names,
        channel_idx=channel_idx, confidence_threshold=confidence_threshold,
        n_channels=5, plot=False
    )
    num_dead_cells, dead_positive_idx = get_and_show_negative_cells(
        None, single_cells, channel_names=channel_names, n_channels=5, plot=False
    )
    return (total_cells_detected, num_opc_positive, num_rfp_positive,
            num_b3tub_positive, num_dead_cells,
            opc_positive_idx, rfp_positive_idx, b3tub_positive_idx, dead_positive_idx)


def extract_cells_nodistort_noresize(img, masks, pad=20, min_area=0,
                                     boundary_margin=200):
    C, H, W = img.shape
    out = []

    for lab in np.unique(masks):
        if lab == 0:
            continue

        ys, xs = np.where(masks == lab)
        if ys.size < min_area:
            continue

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        if (y_min < boundary_margin or y_max > H - boundary_margin or
                x_min < boundary_margin or x_max > W - boundary_margin):
            continue

        y0 = max(y_min - pad, 0)
        y1 = min(y_max + pad + 1, H)
        x0 = max(x_min - pad, 0)
        x1 = min(x_max + pad + 1, W)

        img_crop = img[:, y0:y1, x0:x1].copy()
        mask_crop = (masks[y0:y1, x0:x1] == lab).astype(np.uint8)
        img_crop[0] *= mask_crop

        out.append(dict(label=lab,
                        img=img_crop,
                        mask=mask_crop,
                        bb_position=[y0, y1, x0, x1]))

    return out
