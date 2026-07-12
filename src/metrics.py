"""Full-ranking top-K recommendation metrics.

These utilities assume an implicit-feedback setup where every user has one
held-out test item. Training interactions must be masked before ranking so the
model is evaluated only on unseen recommendations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch


TrainUserPosItems = Mapping[int, Iterable[int] | torch.Tensor] | Sequence[
    Iterable[int] | torch.Tensor
]
TestItems = torch.Tensor | Mapping[int, int] | Sequence[int]
ItemVector = torch.Tensor | Sequence[int]


def _validate_scores(scores: torch.Tensor) -> tuple[int, int]:
    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a torch.Tensor")
    if scores.dim() != 2:
        raise ValueError(f"scores must have shape [num_users, num_items], got {tuple(scores.shape)}")
    if scores.size(0) == 0 or scores.size(1) == 0:
        raise ValueError("scores must contain at least one user and one item")
    if not torch.is_floating_point(scores):
        raise TypeError("scores must be a floating-point tensor so items can be masked with -inf")
    return scores.size(0), scores.size(1)


def _validate_k(k: int, max_items: int) -> None:
    if not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be positive")
    if k > max_items:
        raise ValueError(f"k={k} cannot exceed the available item count/top-k width ({max_items})")


def _validate_item_indices(item_ids: torch.Tensor, num_items: int, name: str) -> None:
    if item_ids.numel() == 0:
        return
    min_item = int(item_ids.min().item())
    max_item = int(item_ids.max().item())
    if min_item < 0 or max_item >= num_items:
        raise IndexError(f"{name} contains item IDs outside [0, {num_items - 1}]")


def _topk_at(topk_items: torch.Tensor, k: int) -> torch.Tensor:
    if not isinstance(topk_items, torch.Tensor):
        raise TypeError("topk_items must be a torch.Tensor")
    if topk_items.dim() != 2:
        raise ValueError(f"topk_items must have shape [num_users, k], got {tuple(topk_items.shape)}")
    if topk_items.size(0) == 0 or topk_items.size(1) == 0:
        raise ValueError("topk_items must contain at least one user and one recommendation")
    if torch.is_floating_point(topk_items):
        raise TypeError("topk_items must contain integer item indices")
    _validate_k(k, topk_items.size(1))
    return topk_items[:, :k].to(dtype=torch.long)


def _as_1d_tensor(values: ItemVector, *, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.to(device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(list(values), dtype=dtype, device=device)

    if tensor.dim() != 1:
        raise ValueError(f"{name} must be a 1D tensor or sequence, got shape {tuple(tensor.shape)}")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    return tensor


def _as_test_items(
    test_items: TestItems,
    *,
    num_users: int,
    device: torch.device,
    num_items: int | None = None,
) -> torch.Tensor:
    if isinstance(test_items, torch.Tensor):
        test_tensor = test_items.to(device=device, dtype=torch.long)
        if test_tensor.dim() != 1:
            raise ValueError(f"test_items must be 1D, got shape {tuple(test_tensor.shape)}")
        if test_tensor.numel() != num_users:
            raise ValueError(f"test_items must contain {num_users} entries, got {test_tensor.numel()}")
    elif isinstance(test_items, Mapping):
        test_tensor = torch.full((num_users,), -1, dtype=torch.long, device=device)
        for user_id, item_id in test_items.items():
            user_idx = int(user_id)
            if user_idx < 0 or user_idx >= num_users:
                raise IndexError(f"test_items contains user ID {user_idx} outside [0, {num_users - 1}]")
            test_tensor[user_idx] = int(item_id)
    else:
        test_tensor = torch.as_tensor(list(test_items), dtype=torch.long, device=device)
        if test_tensor.dim() != 1:
            raise ValueError(f"test_items must be 1D, got shape {tuple(test_tensor.shape)}")
        if test_tensor.numel() != num_users:
            raise ValueError(f"test_items must contain {num_users} entries, got {test_tensor.numel()}")

    if (test_tensor < 0).any().item():
        raise ValueError("test_items must contain one non-negative test item for every user")
    if num_items is not None:
        _validate_item_indices(test_tensor, num_items, "test_items")
    return test_tensor


def _as_item_id_tensor(items: Iterable[int] | torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if isinstance(items, torch.Tensor):
        if items.dim() != 1:
            raise ValueError(f"training item lists must be 1D, got shape {tuple(items.shape)}")
        return items.to(device=device, dtype=torch.long)
    return torch.as_tensor(list(items), dtype=torch.long, device=device)


def _validate_item_group(item_group: torch.Tensor) -> None:
    if ((item_group < 0) | (item_group > 2)).any().item():
        raise ValueError("item_group must use 0=tail, 1=middle, and 2=head encodings")


def mask_train_items(scores: torch.Tensor, train_user_pos_items: TrainUserPosItems) -> torch.Tensor:
    """Return a copy of ``scores`` with each user's training items set to ``-inf``.

    Args:
        scores: Full score matrix with shape ``[num_users, num_items]``.
        train_user_pos_items: Either a sequence where ``train_user_pos_items[u]``
            is an iterable of remapped training item IDs, or a dict mapping user
            IDs to iterables of remapped training item IDs.

    Returns:
        A cloned score matrix with training interactions masked. The input
        ``scores`` tensor is not modified in-place.
    """
    num_users, num_items = _validate_scores(scores)
    masked_scores = scores.clone()

    if isinstance(train_user_pos_items, Mapping):
        user_items_iter = train_user_pos_items.items()
    else:
        if len(train_user_pos_items) != num_users:
            raise ValueError(
                f"train_user_pos_items must contain {num_users} user entries, "
                f"got {len(train_user_pos_items)}"
            )
        user_items_iter = enumerate(train_user_pos_items)

    for user_id, item_ids in user_items_iter:
        user_idx = int(user_id)
        if user_idx < 0 or user_idx >= num_users:
            raise IndexError(f"train_user_pos_items contains user ID {user_idx} outside [0, {num_users - 1}]")

        item_tensor = _as_item_id_tensor(item_ids, device=masked_scores.device)
        if item_tensor.numel() == 0:
            continue

        _validate_item_indices(item_tensor, num_items, "train_user_pos_items")
        masked_scores[user_idx, item_tensor] = float("-inf")

    return masked_scores


def get_topk_items(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return the top-k item IDs for each user from a full score matrix.

    Args:
        scores: Full score matrix with shape ``[num_users, num_items]``.
        k: Number of highest-scoring items to return per user.

    Returns:
        Tensor of item indices with shape ``[num_users, k]``.
    """
    _, num_items = _validate_scores(scores)
    _validate_k(k, num_items)
    return torch.topk(scores, k=k, dim=1).indices


def recall_at_k(topk_items: torch.Tensor, test_items: TestItems, k: int) -> float:
    """Compute Recall@K when each user has exactly one held-out test item."""
    topk = _topk_at(topk_items, k)
    test_tensor = _as_test_items(test_items, num_users=topk.size(0), device=topk.device)

    hits = (topk == test_tensor.unsqueeze(1)).any(dim=1)
    return float(hits.float().mean().item())


def ndcg_at_k(topk_items: torch.Tensor, test_items: TestItems, k: int) -> float:
    """Compute NDCG@K when each user has exactly one held-out test item.

    A hit at 1-based rank ``r`` receives gain ``1 / log2(r + 1)``. Users whose
    test item is absent from the top-k list receive zero gain.
    """
    topk = _topk_at(topk_items, k)
    test_tensor = _as_test_items(test_items, num_users=topk.size(0), device=topk.device)

    matches = topk == test_tensor.unsqueeze(1)
    hits = matches.any(dim=1)
    ranks = matches.float().argmax(dim=1).float() + 1.0

    ndcg = torch.zeros(topk.size(0), dtype=torch.float32, device=topk.device)
    ndcg[hits] = 1.0 / torch.log2(ranks[hits] + 1.0)
    return float(ndcg.mean().item())


def tail_recall_at_k(topk_items: torch.Tensor, test_items: TestItems, item_group: ItemVector, k: int) -> float:
    """Compute Recall@K only over users whose held-out test item is a tail item.

    ``item_group[item]`` must use ``0=tail``, ``1=middle``, and ``2=head``.
    Returns ``0.0`` when there are no users with tail test items.
    """
    topk = _topk_at(topk_items, k)
    group_tensor = _as_1d_tensor(item_group, name="item_group", device=topk.device, dtype=torch.long)
    _validate_item_group(group_tensor)

    test_tensor = _as_test_items(
        test_items,
        num_users=topk.size(0),
        device=topk.device,
        num_items=group_tensor.numel(),
    )
    tail_users = group_tensor[test_tensor] == 0
    if tail_users.sum().item() == 0:
        return 0.0

    hits = (topk == test_tensor.unsqueeze(1)).any(dim=1)
    return float(hits[tail_users].float().mean().item())


def catalog_coverage_at_k(topk_items: torch.Tensor, num_items: int, k: int) -> float:
    """Compute the fraction of catalog items that appear in any top-k list."""
    if not isinstance(num_items, int):
        raise TypeError("num_items must be an integer")
    if num_items <= 0:
        raise ValueError("num_items must be positive")

    topk = _topk_at(topk_items, k)
    _validate_item_indices(topk, num_items, "topk_items")
    return float(torch.unique(topk).numel() / num_items)


def average_recommendation_popularity(topk_items: torch.Tensor, item_degree: ItemVector, k: int) -> float:
    """Compute average log-popularity over all recommendation slots.

    Popularity for an item is ``log(1 + item_degree[item])``. ``item_degree`` is
    expected to be computed from training interactions only.
    """
    topk = _topk_at(topk_items, k)
    degree_tensor = _as_1d_tensor(item_degree, name="item_degree", device=topk.device, dtype=torch.float32)
    if (degree_tensor < 0).any().item():
        raise ValueError("item_degree must be non-negative")

    _validate_item_indices(topk, degree_tensor.numel(), "topk_items")
    return float(torch.log1p(degree_tensor[topk]).mean().item())


def exposure_by_group(topk_items: torch.Tensor, item_group: ItemVector, k: int) -> dict[str, float]:
    """Compute recommendation-slot exposure fractions for tail, middle, and head items.

    ``item_group[item]`` must use ``0=tail``, ``1=middle``, and ``2=head``.
    Returned keys include the numeric cutoff, for example ``TailExposure@20``.
    """
    topk = _topk_at(topk_items, k)
    group_tensor = _as_1d_tensor(item_group, name="item_group", device=topk.device, dtype=torch.long)
    _validate_item_group(group_tensor)

    _validate_item_indices(topk, group_tensor.numel(), "topk_items")
    recommended_groups = group_tensor[topk]
    total_slots = recommended_groups.numel()

    return {
        f"TailExposure@{k}": float((recommended_groups == 0).sum().item() / total_slots),
        f"MiddleExposure@{k}": float((recommended_groups == 1).sum().item() / total_slots),
        f"HeadExposure@{k}": float((recommended_groups == 2).sum().item() / total_slots),
    }


def evaluate_full_ranking(
    scores: torch.Tensor,
    train_user_pos_items: TrainUserPosItems,
    test_items: TestItems,
    item_group: ItemVector,
    item_degree: ItemVector,
    k_list: Iterable[int] = (10, 20),
) -> dict[str, float]:
    """Evaluate full-ranking top-K recommendation metrics.

    This function ranks over the full item catalog after masking training items.
    It does not use sampled negative evaluation.

    Args:
        scores: Full score matrix with shape ``[num_users, num_items]``.
        train_user_pos_items: Training positives to mask before ranking.
        test_items: One held-out test item per user, as a tensor/list or dict.
        item_group: Per-item popularity group encoded as ``0=tail``,
            ``1=middle``, and ``2=head``.
        item_degree: Per-item training degree used for ARP.
        k_list: Cutoffs for Recall and NDCG. The largest cutoff is also used
            for tail recall, coverage, ARP, and exposure metrics.

    Returns:
        Dictionary containing metrics such as ``Recall@10``, ``NDCG@20``,
        ``TailRecall@20``, ``Coverage@20``, ``ARP@20``, and group exposures.
    """
    _, num_items = _validate_scores(scores)
    cutoffs = list(k_list)
    if len(cutoffs) == 0:
        raise ValueError("k_list must contain at least one cutoff")

    for k in cutoffs:
        _validate_k(k, num_items)

    max_k = max(cutoffs)

    masked_scores = mask_train_items(scores, train_user_pos_items)
    topk_items = get_topk_items(masked_scores, max_k)

    results: dict[str, float] = {}
    for k in cutoffs:
        _validate_k(k, max_k)
        results[f"Recall@{k}"] = recall_at_k(topk_items, test_items, k)
        results[f"NDCG@{k}"] = ndcg_at_k(topk_items, test_items, k)

    results[f"TailRecall@{max_k}"] = tail_recall_at_k(topk_items, test_items, item_group, max_k)
    results[f"Coverage@{max_k}"] = catalog_coverage_at_k(topk_items, num_items, max_k)
    results[f"ARP@{max_k}"] = average_recommendation_popularity(topk_items, item_degree, max_k)
    results.update(exposure_by_group(topk_items, item_group, max_k))

    return results
