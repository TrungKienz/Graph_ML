#!/usr/bin/env python3
"""Popularity-aware negative sampling cho BPR.

neg ∝ deg^β: item phổ biến được chọn làm negative NHIỀU hơn → bị đẩy điểm số xuống
mạnh hơn trong quá trình học → giảm popularity bias, kỳ vọng tăng TailRecall.
β = 0  → uniform (giống sampler gốc).
β > 0  → càng lớn càng phạt item phổ biến mạnh (thường 0.3–1.0).
"""

from __future__ import annotations

import torch


def build_neg_probs(item_degree: torch.Tensor, beta: float) -> torch.Tensor | None:
    """Phân phối xác suất [num_items] ∝ deg^β (trên CPU). Trả None nếu β=0 (dùng uniform)."""
    if beta is None or float(beta) == 0.0:
        return None
    w = item_degree.detach().cpu().float().clamp(min=0.0) ** float(beta)
    total = w.sum()
    if total <= 0:
        return None
    return w / total


def sample_bpr_batch_popaware(
    train_edges: torch.Tensor,
    num_items: int,
    train_user_pos_items,
    batch_size: int,
    neg_probs: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    max_reject_iters: int = 20,
):
    """Sinh batch BPR (users, pos_items, neg_items) trên CPU.

    - Positive: lấy uniform từ ``train_edges`` (giống sampler gốc).
    - Negative: lấy theo ``neg_probs`` (nếu None → uniform), có rejection để negative
      không nằm trong positive train của user.
    """
    n_edges = train_edges.size(0)
    idx = torch.randint(0, n_edges, (batch_size,), generator=generator)
    users = train_edges[idx, 0].long()
    pos_items = train_edges[idx, 1].long()

    def draw(n: int) -> torch.Tensor:
        if neg_probs is None:
            return torch.randint(0, num_items, (n,), generator=generator)
        return torch.multinomial(neg_probs, n, replacement=True, generator=generator)

    neg_items = draw(batch_size)
    users_list = users.tolist()
    for _ in range(max_reject_iters):
        neg_list = neg_items.tolist()
        bad = [i for i in range(batch_size)
               if neg_list[i] in train_user_pos_items[users_list[i]]]
        if not bad:
            break
        repl = draw(len(bad))
        for j, i in enumerate(bad):
            neg_items[i] = repl[j]
    return users, pos_items, neg_items.long()
