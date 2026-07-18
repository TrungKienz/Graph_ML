# Popularity-Aware LightGCN for Long-Tail Movie Recommendation

Graph Analytics for Big Data course project on top-K movie recommendation with
popularity-bias analysis, evaluated on **MovieLens-1M**.

Ratings `>= 4` are treated as positive implicit feedback and represented as a
user–movie bipartite graph. Beyond standard accuracy metrics (Recall@K,
NDCG@K), the project measures whether models over-recommend popular movies
and how much exposure long-tail items actually receive, and proposes a method
— **Popularity-Aware LightGCN** — designed to reduce that bias without
destroying accuracy.

## Table of Contents

- [Problem and Goal](#problem-and-goal)
- [Method](#method)
- [Metrics](#metrics)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Environment Setup](#environment-setup)
- [Remote GPU Workflow (A100 cluster)](#remote-gpu-workflow-a100-cluster)
- [How to Run](#how-to-run)
- [Results](#results)
- [Reproducibility Notes](#reproducibility-notes)
- [Documentation](#documentation)
- [Team Workflow](#team-workflow)

## Problem and Goal

Standard collaborative-filtering models (BPR-MF, LightGCN) trained with a
plain BPR loss tend to concentrate almost all recommendation slots on a small
set of already-popular movies ("head" items), starving the long tail of any
exposure. On this dataset, a vanilla `LightGCN` baseline recommends **head**
items in ~96% of its top-20 slots (`HeadExposure@20 ≈ 0.959`) while covering
only ~40% of the catalog.

The goal of this project is to design and validate a method that pushes
recommendations toward the long tail (higher `TailRecall@20`, higher
`Coverage@20`, lower `ARP@20`, lower `HeadExposure@20`) while keeping
top-K accuracy (`Recall@20`, `NDCG@20`) close to the baseline — i.e. a
genuine accuracy/fairness trade-off, not accuracy collapse.

## Method

**Popularity-Aware LightGCN** extends a standard LightGCN backbone with four
independently-toggleable components (see `src/popaware_training.py`):

1. **Item Loss Equalization (ILE)** — items are split into `tail` / `middle` /
   `head` groups by training-graph degree (bottom 50% / next 30% / top 20%,
   see `src/config.py`). An auxiliary penalty
   `λ_ILE * (mean_BPR_loss(head) - mean_BPR_loss(tail))^2` is added to the
   objective so the model is discouraged from fitting head items much better
   than tail items. Implemented in `src/ile_losses.py`.
2. **Degree-Aware Graph Augmentation** — for contrastive view generation,
   edges incident to popular items are dropped with higher probability
   (log-scaled with item degree, bounded in `[DROPOUT_P_MIN, DROPOUT_P_MAX]`).
   Dropout is applied **symmetrically**: the keep/drop decision is made once
   per undirected user–item interaction and mirrored to both the
   user→item and item→user directions, so the propagated graph stays a valid
   undirected graph. Implemented in `src/ile_losses.py`
   (`compute_degree_aware_dropout_probs`) and
   `src/popaware_training.py` (`symmetric_edge_dropout`).
3. **Contrastive Learning (SGL-style)** — two augmented views of the graph are
   propagated through the shared LightGCN encoder, and an InfoNCE loss
   (`src/popaware_training.py: info_nce`) pulls together the two embeddings of
   the same user/item across views while pushing apart other in-batch
   users/items, weighted by `λ_CL` with temperature `τ`.
4. **Popularity-Aware Negative Sampling** — BPR negatives are optionally drawn
   from a `Categorical(degree^β)` distribution instead of uniformly, so
   popular items are sampled as negatives more often, which pushes them down
   in the ranking. Implemented in `src/neg_sampling.py`
   (`build_neg_probs`, `sample_bpr_batch_popaware`); `β = 0` recovers uniform
   sampling.

The four components can be toggled independently for ablation
(`use_ile`, `aug_main`, `use_cl`, `neg_pop_beta` flags on
`train_popaware_lightgcn(...)`), and are jointly optimized as
`L = L_BPR + λ_ILE * L_ILE + λ_CL * L_CL + weight_decay * L2`.

Baselines compared against (implemented outside this method's scope, by other
team members / earlier notebook work):

- `MostPopular` — non-personalized, ranks items by training popularity
  (`src/baselines.py`).
- `BPR-MF` — matrix factorization with BPR loss (`src/models.py`,
  `src/losses.py`, `src/train.py`).
- `LightGCN` — plain graph collaborative filtering, no debiasing
  (`src/models.py`), used as the direct baseline for the proposed method.

## Metrics

All metrics are computed by full-ranking evaluation over the entire item
catalog (masking out each user's train/val positives), implemented in
`src/metrics.py: evaluate_full_ranking`. Ten metrics are reported at `K=10`
and/or `K=20`:

| Metric | Direction | Meaning |
|---|---|---|
| `Recall@K` | ↑ | Fraction of held-out positives retrieved in the top-K |
| `NDCG@K` | ↑ | Rank-weighted retrieval quality |
| `TailRecall@20` | ↑ | Recall restricted to tail-group items |
| `Coverage@20` | ↑ | Fraction of the whole catalog that appears in any user's top-20 |
| `ARP@20` | ↓ | Average Recommendation Popularity — mean training-degree of recommended items |
| `TailExposure@20` | ↑ | Fraction of all top-20 slots (across users) filled by tail items |
| `MiddleExposure@20` | — | Same, for middle-popularity items |
| `HeadExposure@20` | ↓ | Same, for head (popular) items |

`TailExposure + MiddleExposure + HeadExposure = 1` for every user, so these
three always sum to 100% of recommendation slots.

## Repository Structure

```text
TestSSH/
|-- src/                          # reusable project code
|   |-- config.py                 # single source of truth: hyperparameters, paths, seeding
|   |-- data.py                   # original notebook-era data loader (parquet-based)
|   |-- data_loader.py            # DataProcessor: loads cached tensors from preprocess_data/
|   |-- metrics.py                # evaluate_full_ranking + the 10 metrics above
|   |-- baselines.py              # MostPopular
|   |-- models.py                 # BPR-MF, LightGCNRecommender
|   |-- losses.py                 # BPR loss, L2 regularization
|   |-- train.py                  # BPR-MF / LightGCN training loop (notebook path)
|   |-- ile_losses.py             # ILE penalty + degree-aware dropout probabilities
|   |-- ile_training.py           # earlier ILE-only training loop
|   |-- neg_sampling.py           # popularity-aware negative sampling (deg^beta)
|   |-- graph_augmentation.py     # graph augmentation utilities
|   |-- popaware_training.py      # *** main method: train_popaware_lightgcn ***
|   |-- pd_debias.py, pd_training.py   # explored/abandoned causal "Popularity Deconfounding" alternative
|   |-- run_ile_experiments.py, run_augmentation_experiments.py  # earlier ablation runners (superseded)
|   `-- plots.py
|-- preprocess_data/               # cached preprocessed tensors (train/val/test edges, degree, groups)
|-- notebooks/
|   `-- main.ipynb                 # data preprocessing + baseline integration notebook
|-- train_all_popaware.py          # ablation runner: baseline / ile / degreeaug / degreeaug_cl / full
|-- train_sweep_popaware.py        # hyperparameter grid sweep (layers x lambda_ILE x lambda_CL x beta)
|-- train_final_seeds.py           # 3-seed (42, 0, 1) final numbers for 5 operating points
|-- evaluate_test_full.py          # re-evaluate saved models on TEST with the full 10-metric set
|-- run_on_gpu.sh                  # submit/monitor jobs on the remote A100 cluster via Slurm
|-- ssh_config                     # example SSH ProxyJump config to reach the A100 cluster
|-- checkpoints/popaware/          # <run_id>_latest.pt (resume) + <run_id>_best.pt (val-selected)
|-- logs/popaware/                 # per-run training logs (<run_id>.log)
|-- models/                        # final saved models, compatible with evaluate_test_full.py
|-- results/                       # all CSV outputs (per-run, sweep, final mean+-std) + results/metrics/ (baseline reference numbers)
|-- results/popaware/              # per-epoch training history CSVs (history_<run_id>.csv), for loss curves
|-- PopAware_LightGCN_Documentation.md   # full Vietnamese technical report (method, results, analysis)
|-- report_method_section.tex      # English LaTeX method + results section for the paper/report
|-- requirements.txt
`-- README.md
```

Note: the repository also contains a number of one-off debugging/diagnostic
scripts at the top level (`debug_*.py`, `test_*.py`, `fix_data_dtypes.py`,
`comprehensive_fix.py`, etc.) created while chasing specific bugs (glob `**`
crashes on the cluster's Python 3.11, PyTorch≥2.6 `weights_only` default,
tensor dtype mismatches). They are not part of the training pipeline; the
scripts listed above are the ones actually used to produce results.

## Data

Source: **MovieLens-1M**, placed under `data/MovieLens1M/`. Preprocessing
(rating filter `>= 4`, user/item remapping, train/val/test split, training
graph edge index, item degree, and popularity group) happens once in
`notebooks/main.ipynb` and produces cached tensors in `preprocess_data/`
(see `preprocess_data/README.md` for the exact schema):

| File | Contents |
|---|---|
| `train_edges.pt` | `(563204, 2)` int64 — training `(user_id, item_id)` pairs |
| `val_edges.pt` | `(6034, 2)` int64 — one held-out validation interaction per user |
| `test_edges.pt` | `(6034, 2)` int64 — one held-out test interaction per user |
| `edge_index_train.pt` | `(2, 1126408)` int64 — bidirectional PyG-style edge index for message passing |
| `item_degree.pt` | `(3533,)` int64 — training-set degree per item |
| `item_popularity_group.pt` | `(3533,)` int64 — `0=tail, 1=middle, 2=head` per item |
| `metadata.pt` | `num_users`, `num_items`, and related metadata |

`src/data_loader.py: DataProcessor` loads these cached tensors directly (with
`torch.load(..., weights_only=False)`, safe because the files are produced by
this project itself). All training/evaluation scripts use this loader — the
training graph must only ever contain training edges; validation/test edges
must never be used for message passing, and item degree/popularity groups are
computed purely from training interactions.

## Environment Setup

From Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# if execution policy blocks activation for this session:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Key dependencies (`requirements.txt`): `numpy`, `pandas`, `torch>=2.3`,
`torch-geometric>=2.5`, `scikit-learn`, `matplotlib`, `tqdm`,
`ipykernel`/`jupyterlab` for notebook work.

## Remote GPU Workflow (A100 cluster)

**All training and evaluation must run on the remote A100 GPU server, never
on the local laptop.** Code is kept in sync between this Windows machine and
the cluster via Mutagen two-way sync (see `.mutagen`/session named
`testssh`); `run_on_gpu.sh` flushes that sync before every remote command.

```bash
./run_on_gpu.sh <script.py> [args...]     # quick run on the login node (no Slurm) — smoke tests only
./run_on_gpu.sh --train <script.py>       # submit a real training job via Slurm (sbatch) on the A100
./run_on_gpu.sh --status                  # squeue -u $USER
./run_on_gpu.sh --log [jobid]             # tail slurm-<jobid>.out (latest job if jobid omitted)
./run_on_gpu.sh --cancel <jobid>          # scancel a job
```

SSH connectivity uses a `ProxyJump` through an intermediate host to reach the
A100 box (`a100-B`); see `ssh_config` for the connection pattern (copy it
into `~/.ssh/config` with your own credentials — do not commit real
passwords/keys).

Caution: since sync is two-way, an empty local mirror of a directory (e.g.
`checkpoints/`) can cause Mutagen to delete the corresponding files on the
cluster. Do not delete or recreate top-level tracked directories without
checking sync status first.

## How to Run

1. **Preprocess data** (one-time): run the preprocessing cells in
   `notebooks/main.ipynb` to generate the tensors under `preprocess_data/`.
2. **Ablation run** (baseline vs. each method component vs. full method):
   ```bash
   ./run_on_gpu.sh --train train_all_popaware.py
   # or a subset / smoke test:
   ./run_on_gpu.sh train_all_popaware.py --configs baseline,ile --epochs 5
   ```
3. **Hyperparameter sweep** (`num_layers x lambda_ILE x lambda_CL x beta`):
   ```bash
   ./run_on_gpu.sh --train train_sweep_popaware.py
   ```
4. **Final 3-seed numbers** for the reported operating points (baseline +
   4 PopAware configurations, seeds `42, 0, 1`):
   ```bash
   ./run_on_gpu.sh --train train_final_seeds.py
   ```
5. **Re-evaluate saved models** on TEST with the complete 10-metric set
   (does not retrain; reads from `models/final_model_*.pt`):
   ```bash
   ./run_on_gpu.sh evaluate_test_full.py
   ```
   This prints a comparison table to the console and writes
   `results/full_test_metrics_<timestamp>.csv` and
   `results/full_comparison_<timestamp>.csv` — this is the main "show me the
   numbers" step after training.
6. **Qualitative demo** — see real movie-title recommendations for one user
   (does not retrain; reads from `models/final_model_*.pt`):
   ```bash
   python demo_recommend.py                     # random user with a held-out test item
   python demo_recommend.py --user 42 --k 20    # a specific user_idx, top-20
   ./run_on_gpu.sh demo_recommend.py --user 42  # on the cluster (CPU is enough)
   ```
   Prints the movies that user actually watched (with their own ratings), the
   held-out test movie, and the top-K titles the model recommends (each
   tagged 🌳 head / 🌿 middle / 🌱 tail), plus a head/tail count for that
   list — a quick visual read on popularity bias for a single user, to
   complement the aggregate metrics from step 5. `preprocess_data/` only
   stores re-indexed `movie_idx`, not the original MovieLens `MovieID`, so
   this script rebuilds the `movie_idx -> title` mapping straight from
   `data/MovieLens1M/raw/*.dat` and **cross-checks every reconstructed
   `(user, item)` pair against the cached `train/val/test_edges.pt` before
   printing anything** — it refuses to run if the reconstruction doesn't
   match exactly, rather than risk showing a wrong title.

   **⚠️ Naming caveat, confirmed by re-deriving the sweep's frontier rule
   directly from `results/popaware_sweep_20260715_155058.csv`:** the only
   checkpoint currently on disk, `models/final_model_PopAware-BEST_*.pt`, is
   the **automatic winner of the step-3 hyperparameter sweep**
   (`num_layers=3, λ_ILE=0.1, λ_CL=0.5, β=0.5`, single seed, Recall@20=0.1148,
   Coverage@20=0.6148) — it happens to share the display name
   "PopAware-BEST" with, but is a **different model from**, the hand-picked
   `num_layers=2, λ_ILE=1.0, λ_CL=0.1, β=0.5` configuration whose 3-seed
   mean±std (Recall@20=0.1338±0.0030) is what actually appears in the
   [Results](#results) table below. `train_final_seeds.py` only ever saves
   metrics, never model weights, so the exact weights behind the headline
   table don't currently exist on disk. This demo and `evaluate_test_full.py`
   are therefore showing/scoring the sweep-stage model, not the one in the
   results table — to get the latter, rerun that specific configuration and
   save its weights.
7. **Build the final results table** (Markdown + LaTeX, matching the
   [Results](#results) tables below) from `results/metrics/main_results.csv`
   and the newest `results/popaware_final_meanstd_*.csv`:
   ```bash
   python -c "from src.make_table import main; main()"
   ```
   Writes `results/main_results_table.md` and `results/main_results_table.tex`.
8. **Generate report figures** (item degree distribution, popularity group
   sizes, Recall-vs-TailRecall trade-off, Coverage by model, Head/Middle/Tail
   exposure by model, training loss curves) via `src/plots.py`. This module is
   a library of plotting functions (no CLI), so call the ones you need
   directly:
   ```bash
   python - <<'PY'
   from src.data_loader import DataProcessor
   from src.collect_results import main as collect_results
   from src.plots import (
       plot_item_degree_distribution, plot_popularity_group_distribution,
       plot_recall_vs_tail_recall, plot_coverage_by_model, plot_exposure_by_group,
       plot_training_loss_from_history,
   )

   data = DataProcessor()
   plot_item_degree_distribution(data.item_degree)          # fig1 - needs only preprocessed data
   plot_popularity_group_distribution(data.item_popularity_group)  # fig2 - same

   collect_results()  # consolidates results/popaware_final_meanstd_*.csv into results/results.csv
   plot_recall_vs_tail_recall()   # fig3 - needs results/results.csv (previous line)
   plot_coverage_by_model()       # fig4 - same
   plot_exposure_by_group()       # fig5 - same

   plot_training_loss_from_history({  # fig6 - point at any per-epoch history CSV(s) you want to compare
       "PopAware-BEST": "results/popaware/history_<run_id>.csv",
   })
   PY
   ```
   Figures 1-2 only need the preprocessed tensors (`preprocess_data/`) and can
   be produced before any model is trained; figures 3-5 need
   `results/results.csv` populated first (via `collect_results()` above, which
   pulls from the 3-seed mean/std CSV and the notebook baselines); figure 6
   needs the per-epoch `results/popaware/history_<run_id>.csv` written by
   whichever training run(s) you want to plot. All PNGs are saved under
   `figures/`.

Every training run (`train_popaware_lightgcn` in `src/popaware_training.py`)
automatically:

- selects the model checkpoint by **validation** Recall@K only, and touches
  the **test** set exactly once at the end (no leakage);
- writes `<run_id>_latest.pt` (for resume) and `<run_id>_best.pt` (best
  validation score so far) to `checkpoints/popaware/`;
- resumes automatically from `<run_id>_latest.pt` if present (unless
  `--no-resume` is passed);
- logs every epoch to stdout and to `logs/popaware/<run_id>.log`;
- appends per-epoch metrics to `results/popaware/history_<run_id>.csv`.

## Results

*Note: these numbers are the metrics recorded by `train_final_seeds.py`
(mean±std over 3 seeds). No saved checkpoint currently corresponds to the
exact `PopAware-BEST` row below — see the naming caveat under step 6 of
[How to Run](#how-to-run) before using `models/final_model_PopAware-BEST_*.pt`
as if it were this row.*

Reference baselines (`results/metrics/main_results.csv`, K=20):

| Model | Recall@20 ↑ | NDCG@20 ↑ | TailRecall@20 ↑ | Coverage@20 ↑ | ARP@20 ↓ | HeadExposure@20 ↓ |
|---|---|---|---|---|---|---|
| MostPopular | 0.0723 | 0.0269 | 0.0000 | 0.0498 | 7.529 | 1.0000 |
| BPR-MF | 0.1309 | 0.0511 | 0.0109 | 0.6281 | 6.518 | 0.8774 |
| LightGCN (baseline) | 0.1276 | 0.0492 | 0.0036 | 0.3971 | 6.856 | 0.9594 |

Final 3-seed results for Popularity-Aware LightGCN
(`results/popaware_final_meanstd_20260715_174317.csv`, mean ± std over seeds
`{42, 0, 1}`, `num_layers=2`, K=20):

| Model | Recall@20 ↑ | NDCG@20 ↑ | TailRecall@20 ↑ | Coverage@20 ↑ | ARP@20 ↓ | HeadExposure@20 ↓ |
|---|---|---|---|---|---|---|
| LightGCN (baseline) | 0.1287 ± 0.0005 | 0.0497 ± 0.0002 | 0.0050 ± 0.0009 | 0.3978 ± 0.0043 | 6.833 ± 0.009 | 0.9587 ± 0.0006 |
| PopAware-accuracy | 0.1367 ± 0.0035 | 0.0527 ± 0.0006 | 0.0143 ± 0.0023 | 0.4619 ± 0.0289 | 6.500 ± 0.051 | 0.9171 ± 0.0144 |
| PopAware-BEST | 0.1338 ± 0.0030 | 0.0518 ± 0.0004 | 0.0324 ± 0.0049 | 0.5384 ± 0.0293 | 6.400 ± 0.042 | 0.8826 ± 0.0114 |
| PopAware-high-tail | 0.1352 ± 0.0020 | 0.0524 ± 0.0004 | 0.0399 ± 0.0023 | 0.4990 ± 0.0099 | 6.564 ± 0.008 | 0.9117 ± 0.0005 |
| PopAware-fairness | 0.1052 ± 0.0039 | 0.0415 ± 0.0011 | 0.0274 ± 0.0035 | 0.7135 ± 0.0027 | 6.138 ± 0.001 | 0.8333 ± 0.0016 |

The four `PopAware-*` rows are named operating points on an
accuracy/fairness frontier (hyperparameters in `train_final_seeds.py`), all
using `use_ile=True, use_cl=True`, varying `λ_ILE`, `λ_CL`, and the negative
sampling exponent `β`:

- **accuracy** (`λ_ILE=0.1, λ_CL=0.1, β=0.5`): best Recall/NDCG among the
  PopAware configs, still a meaningful tail-recall/coverage gain over
  baseline.
- **BEST** (`λ_ILE=1.0, λ_CL=0.1, β=0.5`): the balanced pick — large bias
  reduction with accuracy close to baseline.
- **high-tail** (`λ_ILE=1.0, λ_CL=0.1, β=0.0`): highest `TailRecall@20` while
  still improving (not hurting) `Recall@20`.
- **fairness** (`λ_ILE=1.0, λ_CL=0.5, β=0.0`): strongest bias reduction
  (`Coverage@20` up to 0.71, `HeadExposure@20` down to 0.83) at the cost of
  a real accuracy drop.

All four PopAware operating points improve every bias metric over the
LightGCN baseline (`TailRecall@20`, `Coverage@20` up; `ARP@20`,
`HeadExposure@20` down), and three of the four (`accuracy`, `BEST`,
`high-tail`) do so while simultaneously *improving* Recall@20/NDCG@20 rather
than trading them away — only the `fairness` point sacrifices accuracy for
the strongest debiasing. See `results/popaware_sweep_20260715_155058.csv` for
the full 24-point hyperparameter grid and
`PopAware_LightGCN_Documentation.md` / `report_method_section.tex` for full
analysis, statistical-significance checks, and a best-seed supplementary
table.

## Reproducibility Notes

- Seeding: `src/config.py: set_seed()` seeds Python/NumPy/PyTorch (CPU+CUDA)
  and enables deterministic cuDNN; call it at the start of every script,
  before building any model or shuffling data.
- Final numbers are reported as mean ± std over 3 seeds (`42, 0, 1`), not a
  single best-seed run, to avoid selection bias ("winner's curse") in the
  headline comparison. A best-seed supplementary table is provided separately
  in the documentation, explicitly labeled as such.
- Model selection strictly uses the **validation** set; the **test** set is
  evaluated exactly once, at the end of training.

## Documentation

This project has several documents, each written for a different reader and a
different depth of understanding. Pick the one that matches what you're
trying to do — reading them in the order below (top to bottom) takes you from
zero knowledge to full technical mastery of the method:

| # | Document | For whom / when to read it |
|---|---|---|
| 1 | [`EXPLANATION.md`](EXPLANATION.md) | **Never studied recommender systems / GNNs before.** A from-scratch teaching document: every one of the model's 13 building blocks explained with intuition, real-world analogies, full LaTeX formulas with every symbol broken down, and **one single worked numeric example threaded through all 13 blocks** (same 4 users/3 items, real hand-computed numbers flowing from block to block) — plus a dedicated Popularity Bias explainer and a comparison table vs. plain LightGCN. Start here if a formula or the overall pipeline doesn't make sense yet. |
| 2 | [`Method_Architecture_Documentation.md`](Method_Architecture_Documentation.md) | **Reading or modifying the code.** Architecture overview (the two-branch design — main path vs. auxiliary contrastive path), folder structure, step-by-step data flow, full loss table, hyperparameter table, the 5 reported configurations, an honest limitations list, and a quick "question → which file" lookup table. |
| 3 | [`PopAware_LightGCN_Documentation.md`](PopAware_LightGCN_Documentation.md) | **Want the full technical report.** Problem/motivation, data, metrics, method formulas, training setup, experimental design, results tables (mean±std, significance, best-seed), architecture/frontier diagrams, analysis vs. expectations, limitations, conclusion — in Vietnamese. |
| 4 | [`report_method_section.tex`](report_method_section.tex) | **Need the English write-up** of the proposed method, implementation details, hyperparameters, and results section, for inclusion in the course report. |
| 5 | [`Slide_Review_and_Defense_QA.md`](Slide_Review_and_Defense_QA.md) | **Preparing to present or defend this project.** 44 anticipated Q&A questions with answers grounded in the actual code/logs, presentation scripts, and a step-by-step pipeline walkthrough tagged with exact file/function references. |

**Reading order recommendation:** if you're new to the project, read `EXPLANATION.md` in full first — it assumes no prior background and builds up every concept before using it. Once the model itself makes sense, use `Method_Architecture_Documentation.md` as your map while reading `src/`. The other three documents are reference material for writing/presenting, not for learning the method.

Everything above reflects the **current, actually-used** method and pipeline
(`train_all_popaware.py` / `train_sweep_popaware.py` / `train_final_seeds.py`
→ `src/popaware_training.py`). An earlier causal-debiasing direction
(`pd_debias.py`, `pd_training.py`) and an earlier ILE-only training loop
(`src/ile_training.py`, `src/run_ile_experiments.py`) were explored and
superseded before this method was finalized — see the "Repository Structure"
note above; they are not documented further because they are not part of the
reported results.

## Team Workflow

- Avoid multiple people editing `notebooks/main.ipynb` simultaneously —
  notebook merge conflicts are hard to resolve.
- Keep reusable logic in `src/`, not in the notebook.
- Save generated outputs to `results/` and plots to `figures/`.
- Training always runs on the remote A100 cluster via `run_on_gpu.sh` — never
  directly on a local machine.
