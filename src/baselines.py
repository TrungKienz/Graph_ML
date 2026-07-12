"""Simple recommendation baselines."""

from __future__ import annotations

import torch


class MostPopular:
    """Recommend items by training-set popularity.

    The same popularity score vector is assigned to every user. Training-item
    masking is intentionally not handled here; use evaluation utilities such as
    ``evaluate_full_ranking`` to mask already-seen items before ranking.
    """

    def __init__(self) -> None:
        self.item_popularity: torch.Tensor | None = None
        self.num_items: int | None = None

    def fit(self, train_edges: torch.Tensor, num_items: int) -> "MostPopular":
        """Compute item popularity from training interactions only.

        Args:
            train_edges: Tensor with shape ``[num_train_edges, 2]`` where each
                row is ``(user_id, item_id)`` using remapped IDs.
            num_items: Total number of remapped items in the catalog.

        Returns:
            The fitted ``MostPopular`` instance.
        """
        if not isinstance(train_edges, torch.Tensor):
            raise TypeError("train_edges must be a torch.Tensor")
        if train_edges.dim() != 2 or train_edges.size(1) != 2:
            raise ValueError(f"train_edges must have shape [num_train_edges, 2], got {tuple(train_edges.shape)}")
        if torch.is_floating_point(train_edges) or torch.is_complex(train_edges):
            raise TypeError("train_edges must contain integer user and item IDs")
        if not isinstance(num_items, int):
            raise TypeError("num_items must be an integer")
        if num_items <= 0:
            raise ValueError("num_items must be positive")

        item_ids = train_edges[:, 1].to(dtype=torch.long)
        if item_ids.numel() > 0:
            min_item = int(item_ids.min().item())
            max_item = int(item_ids.max().item())
            if min_item < 0 or max_item >= num_items:
                raise IndexError(f"train_edges contains item IDs outside [0, {num_items - 1}]")

        self.item_popularity = torch.bincount(item_ids, minlength=num_items).to(dtype=torch.float32)
        self.num_items = num_items
        return self

    def predict_all(self, num_users: int) -> torch.Tensor:
        """Return unmasked MostPopular scores for every user-item pair.

        Args:
            num_users: Number of remapped users to score.

        Returns:
            Score matrix with shape ``[num_users, num_items]``. Every row is the
            same item-popularity vector.
        """
        if self.item_popularity is None or self.num_items is None:
            raise RuntimeError("MostPopular must be fitted before calling predict_all")
        if not isinstance(num_users, int):
            raise TypeError("num_users must be an integer")
        if num_users <= 0:
            raise ValueError("num_users must be positive")

        return self.item_popularity.unsqueeze(0).expand(num_users, self.num_items).clone()
