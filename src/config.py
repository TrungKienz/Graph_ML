"""Shared project configuration.

Single source of truth for hyperparameters, paths, and reproducibility helpers.
Every module (baselines, models, losses, training, plotting) must import values
from here instead of hardcoding them, so that all experiments are comparable.

Convention: constants are UPPER_CASE. Do not mutate them at runtime; if a script
needs a different value, override it locally and log the override with the run.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducible runs.

    Call this at the very start of every training/evaluation script, before any
    model is built or data is shuffled.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Popularity-group encoding (MUST match src/metrics.py and data preprocessing)
# ---------------------------------------------------------------------------
# item_popularity_group uses these integer codes. This is the authoritative
# encoding: 0 = tail, 1 = middle, 2 = head. metrics.py (tail_recall_at_k,
# exposure_by_group) and the data notebook already follow it. Do NOT invert it.
GROUP_TAIL: int = 0
GROUP_MIDDLE: int = 1
GROUP_HEAD: int = 2

# Degree-percentile cutoffs that define the groups, per the project spec:
# Head = top 20%, Middle = next 30%, Tail = bottom 50%.
# i.e. tail if deg < p50, middle if p50 <= deg < p80, head if deg >= p80.
TAIL_MIDDLE_PERCENTILE: float = 50.0
MIDDLE_HEAD_PERCENTILE: float = 80.0


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
EMBEDDING_DIM: int = 64          # shared by BPR-MF and LightGCN
NUM_LAYERS: int = 3              # LightGCN propagation layers (embeddings aggregated incl. layer 0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
LR: float = 1e-3
BATCH_SIZE: int = 2048
NUM_EPOCHS: int = 100
WEIGHT_DECAY: float = 1e-4       # L2 regularization coefficient (lambda in BPR loss)
NUM_NEGATIVES: int = 1           # negative samples per positive for BPR


# ---------------------------------------------------------------------------
# Proposed method: Item Loss Equalization (ILE)
# ---------------------------------------------------------------------------
LAMBDA_ILE: float = 1.0          # weight of the ILE penalty; swept in the ablation

# Grid used for the lambda_ILE sensitivity ablation (section 2.7).
LAMBDA_ILE_GRID: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)


# ---------------------------------------------------------------------------
# Extension: contrastive learning + degree-aware edge dropout
# ---------------------------------------------------------------------------
LAMBDA_CL: float = 0.1           # weight of the contrastive loss
TAU: float = 0.2                 # temperature in the contrastive loss
DROPOUT_P_MIN: float = 0.1       # min edge-drop probability (degree-aware dropout)
DROPOUT_P_MAX: float = 0.4       # max edge-drop probability (for the most popular item)
UNIFORM_DROPOUT_P: float = 0.1   # baseline uniform edge-drop probability


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
K_LIST: tuple[int, ...] = (10, 20)   # top-K cutoffs; K=20 is the primary reporting cutoff
PRIMARY_K: int = 20


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
RESULTS_CSV: Path = RESULTS_DIR / "results.csv"
DATA_DIR: Path = PROJECT_ROOT / "data"
