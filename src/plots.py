"""Figure generation for the report.

Figures 1-2 depend only on Member 1's data outputs (``item_degree``,
``item_popularity_group``) and can be produced before any model is trained.
Figures 3-6 read ``results/results.csv`` and become meaningful once the model
runs have been logged via ``src.experiments.save_result``.

Every function saves a PNG under ``config.FIGURES_DIR`` and returns its path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / Colab-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .experiments import load_results


_GROUP_LABELS = {config.GROUP_TAIL: "Tail", config.GROUP_MIDDLE: "Middle", config.GROUP_HEAD: "Head"}
_GROUP_ORDER = [config.GROUP_HEAD, config.GROUP_MIDDLE, config.GROUP_TAIL]
_GROUP_COLORS = {config.GROUP_HEAD: "#d1495b", config.GROUP_MIDDLE: "#edae49", config.GROUP_TAIL: "#00798c"}


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _save(fig, filename: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] saved {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 1: item degree distribution (proves popularity bias exists)
# ---------------------------------------------------------------------------
def plot_item_degree_distribution(item_degree, filename: str = "fig1_degree_distribution.png") -> Path:
    """Log-log rank-frequency plot of item training degree.

    A straight-ish descending line on log-log axes is the signature of a
    long-tailed (power-law-like) catalog: a few movies collect most interactions
    while most movies are rarely interacted with -- i.e. popularity bias is real.
    """
    deg = np.sort(_to_numpy(item_degree).astype(float))[::-1]
    ranks = np.arange(1, deg.size + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ranks, deg, color="#00798c", linewidth=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Item rank (by popularity)")
    ax.set_ylabel("Training degree (# interactions)")
    ax.set_title("Item Degree Distribution (long-tail)")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Figure 2: head/middle/tail group sizes
# ---------------------------------------------------------------------------
def plot_popularity_group_distribution(
    item_popularity_group, filename: str = "fig2_popularity_groups.png"
) -> Path:
    """Bar chart of how many items fall in each popularity group."""
    groups = _to_numpy(item_popularity_group).astype(int)
    counts = {g: int((groups == g).sum()) for g in _GROUP_ORDER}
    total = max(sum(counts.values()), 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [_GROUP_LABELS[g] for g in _GROUP_ORDER]
    values = [counts[g] for g in _GROUP_ORDER]
    bars = ax.bar(labels, values, color=[_GROUP_COLORS[g] for g in _GROUP_ORDER])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v}\n({v / total:.0%})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# items")
    ax.set_title("Item Count per Popularity Group")
    ax.margins(y=0.15)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Figures 3-5: model comparisons from results.csv
# ---------------------------------------------------------------------------
def _latest_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent row per model (results.csv is append-only)."""
    if df.empty:
        return df
    return df.sort_values("timestamp").groupby("model", as_index=False).last()


def plot_recall_vs_tail_recall(filename: str = "fig3_recall_tradeoff.png") -> Path | None:
    """Grouped bars of Recall@20 vs Tail Recall@20 -- the accuracy/diversity trade-off."""
    df = _latest_per_model(load_results())
    if df.empty:
        print("[plots] results.csv is empty; skipping figure 3")
        return None

    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, df["Recall@20"], width, label="Recall@20", color="#00798c")
    ax.bar(x + width / 2, df["TailRecall@20"], width, label="Tail Recall@20", color="#d1495b")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Recall@20 vs Tail Recall@20")
    ax.legend()
    return _save(fig, filename)


def plot_coverage_by_model(filename: str = "fig4_coverage.png") -> Path | None:
    """Bar chart of Catalog Coverage@20 per model."""
    df = _latest_per_model(load_results())
    if df.empty:
        print("[plots] results.csv is empty; skipping figure 4")
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["model"], df["Coverage@20"], color="#edae49")
    ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.set_ylabel("Catalog Coverage@20")
    ax.set_title("Catalog Coverage@20 by Model")
    return _save(fig, filename)


def plot_exposure_by_group(filename: str = "fig5_exposure.png") -> Path | None:
    """Stacked bars of head/middle/tail exposure share per model."""
    df = _latest_per_model(load_results())
    if df.empty:
        print("[plots] results.csv is empty; skipping figure 5")
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(df))
    for col, g in [("HeadExposure@20", config.GROUP_HEAD),
                   ("MiddleExposure@20", config.GROUP_MIDDLE),
                   ("TailExposure@20", config.GROUP_TAIL)]:
        vals = df[col].to_numpy(dtype=float)
        ax.bar(df["model"], vals, bottom=bottom, label=_GROUP_LABELS[g], color=_GROUP_COLORS[g])
        bottom += vals
    ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.set_ylabel("Exposure share of top-20 slots")
    ax.set_title("Head/Middle/Tail Exposure by Model")
    ax.legend()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Figure 6: training loss curve
# ---------------------------------------------------------------------------
def plot_training_loss_from_history(
    history_files: dict[str, str | Path],
    loss_column: str = "bpr",
    filename: str = "fig6_loss_curve.png",
) -> Path | None:
    """Build the loss curve from per-epoch history CSVs written by the trainers.

    Args:
        history_files: mapping of ``label -> path`` to a ``results/popaware/history_*.csv``
            file (columns include ``epoch`` and per-epoch losses ``bpr``/``ile``/``cl``).
        loss_column: which logged loss to plot. ``"bpr"`` is the accuracy objective and
            is comparable across models regardless of ILE/CL weights.
    """
    curves, labels = [], []
    for label, path in history_files.items():
        path = Path(path)
        if not path.exists():
            print(f"[plots] history missing, skipping: {path}")
            continue
        df = pd.read_csv(path)
        if loss_column not in df.columns:
            print(f"[plots] column '{loss_column}' not in {path.name}; skipping")
            continue
        curves.append(df[loss_column].to_numpy(dtype=float))
        labels.append(label)
    if not curves:
        print("[plots] no history data; skipping figure 6")
        return None
    return plot_training_loss(curves, labels=labels, filename=filename)


def plot_training_loss(loss_history, labels=None, filename: str = "fig6_loss_curve.png") -> Path:
    """Training loss vs epoch.

    Args:
        loss_history: a single list of per-epoch losses, or a list of such lists
            (one per model) when ``labels`` is provided.
        labels: optional model names, one per curve.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    if labels is not None:
        for hist, lab in zip(loss_history, labels):
            ax.plot(range(1, len(hist) + 1), hist, linewidth=1.6, label=lab)
        ax.legend()
    else:
        ax.plot(range(1, len(loss_history) + 1), loss_history, color="#00798c", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Training Loss Curve")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    return _save(fig, filename)
