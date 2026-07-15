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
        # CRITICAL FIX: Additional reproducibility for CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CRITICAL FIX: Validate device availability
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
        
    return device


# ---------------------------------------------------------------------------
# Popularity-group encoding (MUST match src/metrics.py and data preprocessing)
# ---------------------------------------------------------------------------
# item_popularity_group uses these integer codes. This is the authoritative
# encoding: 0 = tail, 1 = middle, 2 = head. metrics.py (tail_recall_at_k,
# exposure_by_group) and the data notebook already follow it. Do NOT invert it.
GROUP_TAIL: int = 0
GROUP_MIDDLE: int = 1
GROUP_HEAD: int = 2

# CRITICAL FIX: Validate group constants
if not (GROUP_TAIL < GROUP_MIDDLE < GROUP_HEAD):
    raise ValueError(f"Invalid group ordering: {GROUP_TAIL}, {GROUP_MIDDLE}, {GROUP_HEAD}")

# Degree-percentile cutoffs that define the groups, per the project spec:
# Head = top 20%, Middle = next 30%, Tail = bottom 50%.
# i.e. tail if deg < p50, middle if p50 <= deg < p80, head if deg >= p80.
TAIL_MIDDLE_PERCENTILE: float = 50.0
MIDDLE_HEAD_PERCENTILE: float = 80.0

# CRITICAL FIX: Validate percentiles
if not (0 < TAIL_MIDDLE_PERCENTILE < MIDDLE_HEAD_PERCENTILE < 100):
    raise ValueError(f"Invalid percentiles: {TAIL_MIDDLE_PERCENTILE}, {MIDDLE_HEAD_PERCENTILE}")


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
EMBEDDING_DIM: int = 64          # shared by BPR-MF and LightGCN
NUM_LAYERS: int = 3              # LightGCN propagation layers (embeddings aggregated incl. layer 0)

# CRITICAL FIX: Validate architecture parameters
if EMBEDDING_DIM <= 0:
    raise ValueError(f"Invalid EMBEDDING_DIM: {EMBEDDING_DIM}")
if NUM_LAYERS <= 0:
    raise ValueError(f"Invalid NUM_LAYERS: {NUM_LAYERS}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
LR: float = 1e-3
LEARNING_RATE: float = LR  # Alias for compatibility
BATCH_SIZE: int = 4096       # Optimized for A100 GPU (upgraded from 2048)
NUM_EPOCHS: int = 100
WEIGHT_DECAY: float = 1e-4       # L2 regularization coefficient (lambda in BPR loss)
NUM_NEGATIVES: int = 1           # negative samples per positive for BPR

# CRITICAL FIX: Validate training parameters
if LR <= 0:
    raise ValueError(f"Invalid LR: {LR}")
if BATCH_SIZE <= 0:
    raise ValueError(f"Invalid BATCH_SIZE: {BATCH_SIZE}")
if NUM_EPOCHS <= 0:
    raise ValueError(f"Invalid NUM_EPOCHS: {NUM_EPOCHS}")
if WEIGHT_DECAY < 0:
    raise ValueError(f"Invalid WEIGHT_DECAY: {WEIGHT_DECAY}")
if NUM_NEGATIVES <= 0:
    raise ValueError(f"Invalid NUM_NEGATIVES: {NUM_NEGATIVES}")

# A100 Performance Optimizations
GRADIENT_ACCUMULATION_STEPS: int = 1  # Can increase if needed for larger effective batch
MIXED_PRECISION: bool = False         # DISABLED: Sparse ops in LightGCN don't support FP16
PIN_MEMORY: bool = True              # Faster CPU->GPU transfer
NON_BLOCKING: bool = True            # Async tensor transfers

# CRITICAL FIX: Validate optimization parameters
if GRADIENT_ACCUMULATION_STEPS <= 0:
    raise ValueError(f"Invalid GRADIENT_ACCUMULATION_STEPS: {GRADIENT_ACCUMULATION_STEPS}")


# ---------------------------------------------------------------------------
# Proposed method: Item Loss Equalization (ILE)
# ---------------------------------------------------------------------------
LAMBDA_ILE: float = 1.0          # weight of the ILE penalty; swept in the ablation

# Grid used for the lambda_ILE sensitivity ablation (section 2.7).
LAMBDA_ILE_GRID: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)

# CRITICAL FIX: Validate ILE parameters
if LAMBDA_ILE < 0:
    raise ValueError(f"Invalid LAMBDA_ILE: {LAMBDA_ILE}")

# Validate lambda grid
if not LAMBDA_ILE_GRID:
    raise ValueError("LAMBDA_ILE_GRID cannot be empty")

for i, lambda_val in enumerate(LAMBDA_ILE_GRID):
    if not isinstance(lambda_val, (int, float)) or lambda_val < 0:
        raise ValueError(f"Invalid lambda value at index {i}: {lambda_val}")

# Check for duplicates
if len(set(LAMBDA_ILE_GRID)) != len(LAMBDA_ILE_GRID):
    raise ValueError(f"Duplicate values in LAMBDA_ILE_GRID: {LAMBDA_ILE_GRID}")


# ---------------------------------------------------------------------------
# Extension: contrastive learning + degree-aware edge dropout
# ---------------------------------------------------------------------------
LAMBDA_CL: float = 0.1           # weight of the contrastive loss
TAU: float = 0.2                 # temperature in the contrastive loss
DROPOUT_P_MIN: float = 0.1       # min edge-drop probability (degree-aware dropout)
DROPOUT_P_MAX: float = 0.4       # max edge-drop probability (for the most popular item)
UNIFORM_DROPOUT_P: float = 0.1   # baseline uniform edge-drop probability

# CRITICAL FIX: Validate contrastive learning parameters
if LAMBDA_CL < 0:
    raise ValueError(f"Invalid LAMBDA_CL: {LAMBDA_CL}")

if TAU <= 0:
    raise ValueError(f"Invalid TAU (temperature): {TAU}")

# Validate dropout probabilities
if not (0 <= DROPOUT_P_MIN <= 1):
    raise ValueError(f"Invalid DROPOUT_P_MIN: {DROPOUT_P_MIN}")

if not (0 <= DROPOUT_P_MAX <= 1):
    raise ValueError(f"Invalid DROPOUT_P_MAX: {DROPOUT_P_MAX}")

if DROPOUT_P_MIN > DROPOUT_P_MAX:
    raise ValueError(f"DROPOUT_P_MIN ({DROPOUT_P_MIN}) > DROPOUT_P_MAX ({DROPOUT_P_MAX})")

if not (0 <= UNIFORM_DROPOUT_P <= 1):
    raise ValueError(f"Invalid UNIFORM_DROPOUT_P: {UNIFORM_DROPOUT_P}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
K_LIST: tuple[int, ...] = (10, 20)   # top-K cutoffs; K=20 is the primary reporting cutoff
PRIMARY_K: int = 20

# CRITICAL FIX: Validate evaluation parameters
if not K_LIST:
    raise ValueError("K_LIST cannot be empty")

for i, k in enumerate(K_LIST):
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"Invalid K value at index {i}: {k}")

if PRIMARY_K not in K_LIST:
    raise ValueError(f"PRIMARY_K ({PRIMARY_K}) must be in K_LIST {K_LIST}")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
RESULTS_CSV: Path = RESULTS_DIR / "results.csv"
DATA_DIR: Path = PROJECT_ROOT / "data"

# CRITICAL FIX: Validate paths and create directories
if not PROJECT_ROOT.exists():
    raise FileNotFoundError(f"PROJECT_ROOT does not exist: {PROJECT_ROOT}")

# Create directories if they don't exist
for directory in [RESULTS_DIR, FIGURES_DIR, DATA_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directory {directory}: {e}")


# ---------------------------------------------------------------------------
# CRITICAL FIX: Additional validation function
# ---------------------------------------------------------------------------
def validate_config() -> None:
    """Validate all configuration parameters for consistency."""
    
    # Validate that all float values are finite
    float_params = {
        'LR': LR,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'LAMBDA_ILE': LAMBDA_ILE,
        'LAMBDA_CL': LAMBDA_CL,
        'TAU': TAU,
        'DROPOUT_P_MIN': DROPOUT_P_MIN,
        'DROPOUT_P_MAX': DROPOUT_P_MAX,
        'UNIFORM_DROPOUT_P': UNIFORM_DROPOUT_P,
        'TAIL_MIDDLE_PERCENTILE': TAIL_MIDDLE_PERCENTILE,
        'MIDDLE_HEAD_PERCENTILE': MIDDLE_HEAD_PERCENTILE
    }
    
    for param_name, param_value in float_params.items():
        if not isinstance(param_value, (int, float)) or not np.isfinite(param_value):
            raise ValueError(f"Parameter {param_name} is not a finite number: {param_value}")
    
    # Validate lambda grid values are finite
    for i, lambda_val in enumerate(LAMBDA_ILE_GRID):
        if not np.isfinite(lambda_val):
            raise ValueError(f"LAMBDA_ILE_GRID[{i}] is not finite: {lambda_val}")
    
    print("✅ Configuration validation passed")


# Run validation on import
try:
    validate_config()
except Exception as e:
    print(f"❌ Configuration validation failed: {e}")
    raise
