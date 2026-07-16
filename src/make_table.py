"""Build the main results table (section 4.4) as Markdown + LaTeX.

Member 5 utility. Multi-seed rows (LightGCN baseline + PopAware variants) are shown
as ``mean +/- std`` from ``results/popaware_final_meanstd_*.csv``; the notebook
baselines (MostPopular, BPR-MF) are single-seed and shown as plain values.

Run:
    python -c "from src.make_table import main; main()"
Outputs ``results/main_results_table.md`` and ``results/main_results_table.tex``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

# Report columns and how to render them (label, decimals, higher-is-better arrow).
_COLS = [
    ("Recall@20", 4, "up"),
    ("NDCG@20", 4, "up"),
    ("TailRecall@20", 4, "up"),
    ("Coverage@20", 4, "up"),
    ("ARP@20", 3, "down"),
]

# Row order: baselines first, then proposed variants.
_ORDER = [
    "MostPopular", "BPR-MF", "LightGCN",
    "PopAware-accuracy", "PopAware-high-tail", "PopAware-BEST", "PopAware-fairness",
]


def _load() -> dict[str, dict[str, tuple[float, float | None]]]:
    """model -> {metric: (mean, std_or_None)}."""
    rows: dict[str, dict[str, tuple[float, float | None]]] = {}

    main = config.RESULTS_DIR / "metrics" / "main_results.csv"
    if main.exists():
        df = pd.read_csv(main)
        for _, r in df.iterrows():
            if r["Model"] in ("MostPopular", "BPR-MF"):
                rows[r["Model"]] = {m: (float(r[m]), None) for m, _, _ in _COLS}

    ms = sorted(config.RESULTS_DIR.glob("popaware_final_meanstd_*.csv"))
    if ms:
        df = pd.read_csv(ms[-1])
        for _, r in df.iterrows():
            rows[str(r["Model"])] = {
                m: (float(r[f"{m}_mean"]), float(r[f"{m}_std"])) for m, _, _ in _COLS
            }
    return rows


def _fmt(mean: float, std: float | None, dec: int) -> str:
    return f"{mean:.{dec}f}" + (f" ± {std:.{dec}f}" if std is not None else "")


def build_markdown(rows) -> str:
    head = "| Model | " + " | ".join(f"{m} {'↑' if a=='up' else '↓'}" for m, _, a in _COLS) + " |"
    sep = "|" + "---|" * (len(_COLS) + 1)
    lines = [head, sep]
    for model in _ORDER:
        if model not in rows:
            continue
        cells = [_fmt(*rows[model][m], dec) for m, dec, _ in _COLS]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_latex(rows) -> str:
    spec = "l" + "r" * len(_COLS)
    header = " & ".join(["Model"] + [m.replace("@", "@\\,") for m, _, _ in _COLS]) + r" \\"
    out = [r"\begin{tabular}{" + spec + "}", r"\toprule", header, r"\midrule"]
    for model in _ORDER:
        if model not in rows:
            continue
        cells = [_fmt(*rows[model][m], dec).replace("±", r"$\pm$") for m, dec, _ in _COLS]
        out.append(model.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def main() -> tuple[Path, Path]:
    rows = _load()
    md = build_markdown(rows)
    tex = build_latex(rows)

    md_path = config.RESULTS_DIR / "main_results_table.md"
    tex_path = config.RESULTS_DIR / "main_results_table.tex"
    md_path.write_text(md + "\n", encoding="utf-8")
    tex_path.write_text(tex + "\n", encoding="utf-8")
    print(md)
    print(f"\n[make_table] wrote {md_path}\n[make_table] wrote {tex_path}")
    return md_path, tex_path


if __name__ == "__main__":
    main()
