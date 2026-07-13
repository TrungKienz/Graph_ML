"""Experiment result logging.

Fixes the schema of ``results/results.csv`` and provides ``save_result`` so that
every model logs metrics the same way. The metric column names are exactly the
keys returned by ``src.metrics.evaluate_full_ranking`` with ``K_LIST=(10, 20)``.

Contract for other members:
    from src.experiments import save_result
    metrics = evaluate_full_ranking(...)          # dict from src.metrics
    save_result("LightGCN", metrics, hyperparams={"lambda_ile": 0.0})

Do not rename columns in ``RESULT_COLUMNS`` and do not print metrics only to the
console -- always persist them through ``save_result`` so results are comparable.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from . import config


# Identity / bookkeeping columns, written for every run.
ID_COLUMNS: list[str] = [
    "model",          # canonical model name, e.g. "LightGCN", "LightGCN+ILE"
    "seed",
    "embedding_dim",
    "num_layers",
    "lambda_ile",
    "timestamp",
]

# Metric columns -- exactly the keys of evaluate_full_ranking(k_list=(10, 20)).
METRIC_COLUMNS: list[str] = [
    "Recall@10",
    "NDCG@10",
    "Recall@20",
    "NDCG@20",
    "TailRecall@20",
    "Coverage@20",
    "ARP@20",
    "TailExposure@20",
    "MiddleExposure@20",
    "HeadExposure@20",
]

RESULT_COLUMNS: list[str] = ID_COLUMNS + METRIC_COLUMNS

# Canonical model names -- use these exact strings in the ``model`` column so the
# results table does not end up with near-duplicate rows ("LightGCN" vs "lightgcn").
MODEL_NAMES: tuple[str, ...] = (
    "MostPopular",
    "BPR-MF",
    "LightGCN",
    "LightGCN+ILE",
    "LightGCN+UniformDropout",
    "LightGCN+DegreeDropout",
    "FullModel",
)


def save_result(
    model: str,
    metrics: Mapping[str, float],
    hyperparams: Mapping[str, Any] | None = None,
    results_csv: Path | str | None = None,
) -> dict[str, Any]:
    """Append one evaluation result as a row to ``results.csv``.

    Args:
        model: Canonical model name (see ``MODEL_NAMES``).
        metrics: Metric dict as returned by ``evaluate_full_ranking``. Any missing
            metric column is written as an empty cell; unknown keys are ignored
            with a warning so a typo does not silently create a phantom column.
        hyperparams: Optional overrides for the ``ID_COLUMNS`` bookkeeping fields
            (e.g. ``{"seed": 7, "lambda_ile": 0.5}``). Missing fields fall back to
            the values in ``src.config``.
        results_csv: Target CSV path. Defaults to ``config.RESULTS_CSV``.

    Returns:
        The row that was written, as a plain dict.
    """
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    if model not in MODEL_NAMES:
        print(f"[save_result] warning: '{model}' is not in MODEL_NAMES {MODEL_NAMES}; "
              "writing it anyway, but check the spelling for a consistent results table.")

    hp = dict(hyperparams or {})

    unknown = set(metrics) - set(METRIC_COLUMNS)
    if unknown:
        print(f"[save_result] warning: ignoring unexpected metric keys {sorted(unknown)}; "
              f"expected a subset of {METRIC_COLUMNS}")

    row: dict[str, Any] = {
        "model": model,
        "seed": hp.get("seed", config.SEED),
        "embedding_dim": hp.get("embedding_dim", config.EMBEDDING_DIM),
        "num_layers": hp.get("num_layers", config.NUM_LAYERS),
        "lambda_ile": hp.get("lambda_ile", None),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    for col in METRIC_COLUMNS:
        row[col] = metrics.get(col, None)

    path = Path(results_csv) if results_csv is not None else config.RESULTS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    row_df = pd.DataFrame([row], columns=RESULT_COLUMNS)
    header = not path.exists()
    row_df.to_csv(path, mode="a", header=header, index=False)

    print(f"[save_result] appended '{model}' to {path}")
    return row


def load_results(results_csv: Path | str | None = None) -> pd.DataFrame:
    """Load ``results.csv`` as a DataFrame (empty with the right columns if absent)."""
    path = Path(results_csv) if results_csv is not None else config.RESULTS_CSV
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.read_csv(path)
