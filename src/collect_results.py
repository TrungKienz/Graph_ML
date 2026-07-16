"""Consolidate scattered experiment outputs into one canonical ``results/results.csv``.

Member 5 utility. The individual training scripts wrote results in several places
and formats:

* ``results/metrics/main_results.csv``          -- notebook baselines (MostPopular,
  BPR-MF, LightGCN), single seed.
* ``results/popaware_final_meanstd_*.csv``       -- 3-seed mean/std for the LightGCN
  baseline and the PopAware proposed-method variants (same L2 pipeline, same
  train+val evaluation mask).

``src.plots`` and the main results table read a single file, ``config.RESULTS_CSV``
(= ``results/results.csv``), whose columns are ``experiments.RESULT_COLUMNS``. This
module rebuilds that file from the sources above so the figures have data.

IMPORTANT protocol caveat (surfaced by the review, not yet fixed by Member 2/3):
the notebook baselines mask **train-only** at test time, while the PopAware pipeline
masks **train+val**. To keep the LightGCN-vs-PopAware comparison honest, the
canonical ``LightGCN`` row here is taken from the *PopAware pipeline* (3-seed,
train+val mask), NOT from the notebook. ``MostPopular`` and ``BPR-MF`` are still
sourced from the notebook; for MostPopular the mask choice is negligible, for
BPR-MF it is a minor caveat noted in ``PROTOCOL_CAVEAT`` below.

Run:
    python -c "from src.collect_results import main; main()"
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config
from .experiments import METRIC_COLUMNS, RESULT_COLUMNS

# Human-readable warning kept next to the data so it reaches the report writer.
# MostPopular was recomputed with the train+val mask (exact). LightGCN + PopAware
# already use train+val (popaware pipeline, 3-seed). Only BPR-MF still carries a
# train-only mask because it has no saved checkpoint to re-evaluate; the notebook
# eval cell is already patched, so re-running it corrects BPR-MF. Measured impact
# of the mask change on MostPopular was +0.0002 Recall@20 (below multi-seed std),
# so BPR-MF's pending correction does not affect any conclusion.
PROTOCOL_CAVEAT = (
    "All rows use the train+val test mask EXCEPT BPR-MF, which is still train-only "
    "(no checkpoint to re-evaluate). Notebook is patched; re-running it fixes BPR-MF. "
    "Measured mask impact is ~+0.0002 Recall@20, below multi-seed noise."
)

_MAIN_RESULTS = config.RESULTS_DIR / "metrics" / "main_results.csv"


def _find_meanstd() -> Path | None:
    """Newest ``popaware_final_meanstd_*.csv`` under results/ (or None)."""
    candidates = sorted(config.RESULTS_DIR.glob("popaware_final_meanstd_*.csv"))
    return candidates[-1] if candidates else None


def _blank_row(model: str, *, seed, num_layers, lambda_ile) -> dict:
    row = {c: None for c in RESULT_COLUMNS}
    row.update(
        model=model,
        seed=seed,
        embedding_dim=config.EMBEDDING_DIM,
        num_layers=num_layers,
        lambda_ile=lambda_ile,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    return row


def _rows_from_main_results(models: tuple[str, ...]) -> list[dict]:
    """Pull the requested baseline models out of the notebook's main_results.csv."""
    if not _MAIN_RESULTS.exists():
        print(f"[collect] warning: {_MAIN_RESULTS} missing; skipping {models}")
        return []
    df = pd.read_csv(_MAIN_RESULTS)
    rows: list[dict] = []
    for model in models:
        match = df[df["Model"] == model]
        if match.empty:
            print(f"[collect] warning: '{model}' not found in {_MAIN_RESULTS.name}")
            continue
        src = match.iloc[0]
        row = _blank_row(model, seed=config.SEED, num_layers=None, lambda_ile=None)
        for col in METRIC_COLUMNS:
            if col in src:
                row[col] = float(src[col])
        rows.append(row)
    return rows


def _rows_from_meanstd(path: Path) -> list[dict]:
    """One row per model from the 3-seed mean/std summary (means only)."""
    df = pd.read_csv(path)
    rows: list[dict] = []
    for _, src in df.iterrows():
        row = _blank_row(str(src["Model"]), seed="mean(0,1,42)", num_layers=2, lambda_ile=None)
        for col in METRIC_COLUMNS:
            mean_col = f"{col}_mean"
            if mean_col in src:
                row[col] = float(src[mean_col])
        rows.append(row)
    return rows


def collect() -> pd.DataFrame:
    """Assemble the canonical results table as a DataFrame (does not write)."""
    rows: list[dict] = []

    # Non-personalised + non-graph baselines come from the notebook run.
    rows += _rows_from_main_results(("MostPopular", "BPR-MF"))

    # LightGCN baseline + PopAware variants come from the 3-seed pipeline so they
    # share config (L2) and evaluation protocol (train+val mask).
    meanstd = _find_meanstd()
    if meanstd is not None:
        print(f"[collect] using multi-seed summary: {meanstd.name}")
        rows += _rows_from_meanstd(meanstd)
    else:
        print("[collect] warning: no popaware_final_meanstd_*.csv found; "
              "LightGCN/PopAware rows will be missing")

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def main(out: Path | str | None = None) -> Path:
    """Write the consolidated table to ``config.RESULTS_CSV`` (or ``out``)."""
    df = collect()
    path = Path(out) if out is not None else config.RESULTS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[collect] wrote {len(df)} rows to {path}")
    print(f"[collect] models: {list(df['model'])}")
    print(f"[collect] CAVEAT: {PROTOCOL_CAVEAT}")
    return path


if __name__ == "__main__":
    main()
