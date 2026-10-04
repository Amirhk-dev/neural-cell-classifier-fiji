import os
import torch
from cellpose import models
import yaml
import pandas as pd
import random
import numpy as np


def make_deterministic(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required for deterministic CuBLAS on CUDA >= 10.2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def make_confidence_table(single_cells, channel_names, channel_name):
    """
    Build a pandas DataFrame with cells sorted by confidence for channel_name.
    """
    sorted_cells = _sorted_cells_by_conf(single_cells, channel_names, channel_name)
    rows = []
    for rank, (conf, idx, _) in enumerate(sorted_cells):
        rows.append({
            "rank": rank,
            "cell_idx": idx,
            f"{channel_name} confidence(%)": conf,
        })
    return pd.DataFrame(rows)

def _sorted_cells_by_conf(single_cells, channel_names, channel_name):
    """
    Return list of (confidence, cell_idx, cell_dict) sorted by confidence ascending
    for a given channel_name. Skips None.
    """
    sorted_list = []
    for i, sc in enumerate(single_cells):
        conf_dict = sc["result"]["confidence(%)"]
        conf = conf_dict.get(channel_name, None)
        if conf is None:
            continue
        sorted_list.append((conf, i, sc))

    sorted_list.sort(key=lambda x: x[0])  # low -> high
    return sorted_list

def load_conf(path, project, conf_name):
    # Reads <path>/<project>/configs/<conf_name>, e.g.
    # load_conf(base_dir, 'my-project', 'generate_single_cells.yaml').
    yaml_file = os.path.join(path, project, 'configs', conf_name)
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_img_paths(folder, ext='.czi'):
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(ext)
    ]
    # Sort for deterministic ordering across runs and filesystems
    return sorted(files, key=str.lower)

def best_device() -> str:
    """Fastest torch device available: CUDA > Apple Silicon MPS > CPU.

    Set NEURAL_IMGS_DEVICE=cpu|cuda|mps to override auto-detection, e.g. to
    force CPU for an exact-reproducibility check against a CPU-only reference
    run (MPS/CUDA can differ from CPU by small floating-point amounts).
    """
    forced = os.environ.get("NEURAL_IMGS_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_cellpose_model():
    device = best_device()
    use_gpu = device != "cpu"
    print("Using device:", device)
    return models.CellposeModel(gpu=use_gpu)

def fmt(val, spec):
    return f"{val:{spec}}" if val is not None else "n/a"

def print_tabular(result):
    channels = list(result["masked_means"])
    ch_width = max(len(ch) for ch in channels) + 5

    for ch in channels:
        print(
            f"{ch:<{ch_width-len(ch)}} | "
            f"mean: {fmt(result['masked_means'][ch], '.3f')} | "
            f"percentile: {fmt(result['percentile'][ch], '.2f')} | "
            f"confidence(%): {fmt(result['confidence(%)'][ch], '.2f')} | "
            f"positive: {result['positive'].get(ch, 'n/a')}"
        )

def convert_to_yolo(row0, row1, col0, col1, H, W):
    w = col1 - col0
    h = row1 - row0
    xc = col0 + w/2
    yc = row0 + h/2
    return xc/W, yc/H, w/W, h/H
