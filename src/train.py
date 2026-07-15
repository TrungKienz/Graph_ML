"""Training loops and BPR negative sampling for BPR-MF and LightGCN.

Public entry points:
    * ``sample_bpr_batch`` -- draw a batch of ``(user, pos_item, neg_item)``
      triplets with negatives that the user has *not* interacted with in train.
    * ``train_bpr_mf``     -- train a :class:`~src.models.BPRMF` model.
    * ``train_lightgcn``   -- train a :class:`~src.models.LightGCNRecommender`.

Both trainers return the fitted model plus a list of per-epoch mean losses. Get
the score matrix afterwards with ``model.full_sort_scores(...)`` and evaluate it
with ``src.metrics.evaluate_full_ranking``.

All hyperparameters default to the values in ``src.config`` so every model in the
project is trained comparably.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

import config
from config import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    LR,
    NUM_EPOCHS,
    NUM_LAYERS,
    NUM_NEGATIVES,
    SEED,
    WEIGHT_DECAY,
)
from losses import bpr_loss, l2_regularization
from models import BPRMF, LightGCNRecommender


def sample_bpr_batch(
    train_edges: torch.Tensor,
    num_items: int,
    train_user_pos_items: Sequence[set[int]],
    batch_size: int = BATCH_SIZE,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a batch of BPR triplets ``(user, pos_item, neg_item)``.

    Positive pairs are drawn uniformly (with replacement) from ``train_edges``.
    For each, a negative item is drawn uniformly from the catalog and rejected /
    resampled until it is an item the user has **not** interacted with in the
    training split, using ``train_user_pos_items[u]``.

    Args:
        train_edges: Training interactions, shape ``[num_train_edges, 2]`` with
            rows ``(user_id, item_id)``. Kept on CPU for sampling.
        num_items: Catalog size; negatives are sampled from ``[0, num_items)``.
        train_user_pos_items: ``train_user_pos_items[u]`` is the set of item IDs
            user ``u`` interacted with in train (used to filter negatives).
        batch_size: Number of triplets to sample.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        ``(users, pos_items, neg_items)`` as 1D CPU ``LongTensor``s of length
        ``batch_size``. Move them to the model's device in the training loop.
    """
    edges_cpu = train_edges.cpu()
    num_edges = edges_cpu.size(0)

    idx = torch.randint(0, num_edges, (batch_size,), generator=generator)
    users = edges_cpu[idx, 0].clone()
    pos_items = edges_cpu[idx, 1].clone()

    neg_items = torch.randint(0, num_items, (batch_size,), generator=generator)

    users_list = users.tolist()
    neg_list = neg_items.tolist()
    for b in range(batch_size):
        seen = train_user_pos_items[users_list[b]]
        j = neg_list[b]
        while j in seen:
            # CRITICAL FIX: Ensure we get int, not tensor to avoid dtype issues
            j = torch.randint(0, num_items, (1,), generator=generator).item()
        neg_list[b] = j
    # CRITICAL FIX: Explicitly create tensor with correct dtype
    neg_items = torch.tensor(neg_list, dtype=torch.long)

    return users, pos_items, neg_items


def _steps_per_epoch(num_edges: int, batch_size: int) -> int:
    """One pass' worth of batches (at least one)."""
    return max(1, num_edges // batch_size)


def train_bpr_mf(
    train_edges: torch.Tensor,
    num_users: int,
    num_items: int,
    train_user_pos_items: Sequence[set[int]],
    *,
    embedding_dim: int = EMBEDDING_DIM,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    weight_decay: float = WEIGHT_DECAY,
    num_negatives: int = NUM_NEGATIVES,
    device: torch.device | str | None = None,
    seed: int = SEED,
    verbose: bool = True,
) -> tuple[BPRMF, list[float]]:
    """Train a BPR-MF model.

    Args:
        train_edges: ``[num_train_edges, 2]`` training interactions.
        num_users, num_items: Catalog sizes.
        train_user_pos_items: Per-user train item sets (for negative sampling).
        embedding_dim, num_epochs, lr, batch_size, weight_decay, num_negatives:
            Hyperparameters; default to ``src.config``.
        device: Torch device; defaults to CUDA if available.
        seed: Seed for reproducible init + sampling.
        verbose: Print per-epoch mean loss.

    Returns:
        ``(model, loss_history)``. Call ``model.full_sort_scores()`` for the
        ``[num_users, num_items]`` score matrix.
    """
    device = torch.device(device) if device is not None else config.get_device()
    config.set_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    model = BPRMF(num_users, num_items, embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    edges_cpu = train_edges.cpu()
    steps = _steps_per_epoch(edges_cpu.size(0), batch_size)
    loss_history: list[float] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(steps):
            users, pos_items, neg_items = sample_bpr_batch(
                edges_cpu, num_items, train_user_pos_items, batch_size, generator
            )
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            pos_scores, neg_scores, (u, p, n) = model.bpr_forward(users, pos_items, neg_items)
            loss = bpr_loss(pos_scores, neg_scores)
            reg = l2_regularization(u, p, n, batch_size=users.size(0))
            loss = loss + weight_decay * reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        mean_loss = epoch_loss / steps
        loss_history.append(mean_loss)
        if verbose:
            print(f"[BPR-MF] epoch {epoch:3d}/{num_epochs}  loss={mean_loss:.4f}")

    return model, loss_history


def train_lightgcn(
    train_edges: torch.Tensor,
    edge_index_train: torch.Tensor,
    num_users: int,
    num_items: int,
    train_user_pos_items: Sequence[set[int]],
    *,
    embedding_dim: int = EMBEDDING_DIM,
    num_layers: int = NUM_LAYERS,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    weight_decay: float = WEIGHT_DECAY,
    num_negatives: int = NUM_NEGATIVES,
    device: torch.device | str | None = None,
    seed: int = SEED,
    verbose: bool = True,
) -> tuple[LightGCNRecommender, list[float]]:
    """Train a LightGCN model.

    Same BPR objective and sampling as :func:`train_bpr_mf`, but embeddings are
    produced by message passing over ``edge_index_train`` (recomputed every
    step, as embeddings change). Regularization is applied to the layer-0 ego
    embeddings only.

    Args:
        train_edges: ``[num_train_edges, 2]`` training interactions (for sampling).
        edge_index_train: Bidirectional graph ``[2, num_edges]`` with items offset
            by ``num_users`` (message-passing graph).
        num_users, num_items: Catalog sizes.
        train_user_pos_items: Per-user train item sets (for negative sampling).
        embedding_dim, num_layers, num_epochs, lr, batch_size, weight_decay,
            num_negatives: Hyperparameters; default to ``src.config``.
        device: Torch device; defaults to CUDA if available.
        seed: Seed for reproducible init + sampling.
        verbose: Print per-epoch mean loss.

    Returns:
        ``(model, loss_history)``. Call ``model.full_sort_scores(edge_index_train)``
        for the ``[num_users, num_items]`` score matrix.
    """
    device = torch.device(device) if device is not None else config.get_device()
    config.set_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    model = LightGCNRecommender(num_users, num_items, embedding_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    edge_index = edge_index_train.to(device)
    edges_cpu = train_edges.cpu()
    steps = _steps_per_epoch(edges_cpu.size(0), batch_size)
    loss_history: list[float] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(steps):
            users, pos_items, neg_items = sample_bpr_batch(
                edges_cpu, num_items, train_user_pos_items, batch_size, generator
            )
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            pos_scores, neg_scores, (u0, p0, n0) = model.bpr_forward(
                edge_index, users, pos_items, neg_items
            )
            loss = bpr_loss(pos_scores, neg_scores)
            reg = l2_regularization(u0, p0, n0, batch_size=users.size(0))
            loss = loss + weight_decay * reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        mean_loss = epoch_loss / steps
        loss_history.append(mean_loss)
        if verbose:
            print(f"[LightGCN] epoch {epoch:3d}/{num_epochs}  loss={mean_loss:.4f}")

    return model, loss_history
