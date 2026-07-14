# Popularity-Aware LightGCN for Long-Tail Movie Recommendation

Graph Analytics for Big Data course project on top-K movie recommendation with popularity-bias analysis.

## Project Overview

This project studies recommendation on the MovieLens 1M dataset. Ratings greater than or equal to 4 are treated as positive implicit feedback, and the data is represented as a user-movie bipartite graph.

The main goal is to compare standard recommendation accuracy with long-tail behavior. In addition to top-K metrics such as Recall@K and NDCG@K, the project evaluates whether models over-recommend popular movies and how much exposure tail items receive.

## Method Summary

Planned model variants:

- `MostPopular`: non-personalized baseline that recommends movies by training-set popularity.
- `BPR-MF`: matrix factorization trained with Bayesian Personalized Ranking loss.
- `LightGCN`: graph collaborative filtering model using user-item message passing.
- `LightGCN + Item Loss Equalization (ILE)`: LightGCN variant that reweights item contributions to reduce popularity bias.
- Optional degree-aware graph augmentation: graph modification strategy based on item degree/popularity.

Current implementation status:

- `MostPopular` is implemented in `src/baselines.py`.
- Shared full-ranking evaluation metrics are implemented in `src/metrics.py`.
- `BPR-MF` and `LightGCN` are implemented in `src/models.py`, with the BPR loss in `src/losses.py` and training/negative-sampling loops in `src/train.py`. Both are wired into `notebooks/main.ipynb` and evaluated through `evaluate_full_ranking`.
- ILE (Item Loss Equalization) is the remaining next step and builds on the LightGCN training loop.

## Repository Structure

```text
graph-recommender-project/
|-- notebooks/
|   `-- main.ipynb
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data.py
|   |-- metrics.py
|   |-- baselines.py
|   |-- models.py
|   |-- losses.py
|   |-- train.py
|   `-- plots.py
|-- results/
|-- figures/
|-- requirements.txt
|-- README.md
`-- .gitignore
```

- `notebooks/`: final integration and experiment notebooks. Start with `notebooks/main.ipynb`.
- `src/`: reusable project code for data processing, metrics, baselines, models, losses, training, and plotting.
- `results/`: saved metric tables, experiment outputs, and intermediate result files.
- `figures/`: plots and visualizations used for analysis or presentation.

## Environment Setup

From Windows PowerShell, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation for the current session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional notebook kernel setup:

```powershell
python -m ipykernel install --user --name graph-ml --display-name "Python (Graph ML)"
```

Google Colab can also be used. If using Colab, upload or mount the repository and install the dependencies inside the notebook runtime.

## How To Run The Project

1. Open `notebooks/main.ipynb` in Jupyter, VS Code, Kaggle, or Colab.
2. Select the project virtual environment kernel, for example `Python (Graph ML)`.
3. Run the setup/import cells.
4. Run the preprocessing cells to load MovieLens 1M, keep ratings `>= 4`, remap users/items, and create train/validation/test splits.
5. Run the `MostPopular` evaluation pipeline.
6. Later notebook sections will run BPR-MF, LightGCN, and LightGCN + ILE after those implementations are added.

## Current Progress

- Repository structure initialized.
- Main notebook created.
- Preprocessing pipeline prepared.
- Evaluation metrics implemented in `src/metrics.py`.
- `MostPopular` baseline implemented in `src/baselines.py`.
- Next steps: implement BPR-MF, LightGCN, and LightGCN + ILE.

## Team Workflow

- Avoid having multiple people edit `notebooks/main.ipynb` at the same time because notebook merge conflicts are difficult to resolve.
- Team members should mainly work in separate files under `src/`.
- Keep reusable logic out of the notebook when possible.
- Use `notebooks/main.ipynb` as the final integration and experiment file.
- Save generated outputs in `results/` and plots in `figures/`.

## Notes

- The training graph must use only training edges.
- Validation and test edges must not be used for graph message passing.
- Item popularity, item groups, and item degree should be computed from training interactions only.
- All models should output a score matrix with shape `[num_users, num_items]`.
- All models should be evaluated using the shared `evaluate_full_ranking` function from `src/metrics.py`.
