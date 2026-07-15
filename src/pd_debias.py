#!/usr/bin/env python3
"""Popularity Deconfounding (PD/PDA) — core transforms.

Phương pháp chính để giảm bias vào item quá phổ biến.

Tham chiếu: Zhang et al., "Causal Intervention for Leveraging Popularity Bias
in Recommendation", SIGIR 2021 (PD / PDA).

Ý tưởng nhân quả
----------------
Độ phổ biến Z là **confounder**: nó vừa làm item dễ được hiển thị, vừa làm tăng
xác suất click. Model thường học nhầm confounding này thành "chất lượng" → thiên
lệch về item phổ biến.

- **Training (backdoor adjustment):** dùng điểm số có gắn phổ biến
      ŷ = f(s) · m^γ
  với s = <e_u, e_i> là điểm matching, m = độ phổ biến chuẩn hoá, f(·) = ELU'(·)
  (luôn dương). Vì item phổ biến đã được cộng sẵn hệ số m^γ khi train, phần s
  KHÔNG còn phải mã hoá độ phổ biến để khớp dữ liệu → s trở thành "sở thích thuần".

- **Inference (do-operation / PDA):** BỎ thừa số m^γ, xếp hạng theo f(s). Vì f đơn
  điệu tăng nên thứ tự theo f(s) trùng thứ tự theo s — tức chỉ cần rank theo
  ``full_sort_scores`` thô của model đã train PD. Đây là chỗ khử ảnh hưởng phổ biến.

γ ≥ 0 điều khiển mức khử phổ biến: γ=0 ≈ không khử, γ lớn = khử mạnh.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def elu_plus(x: torch.Tensor) -> torch.Tensor:
    """f(x) = ELU(x) + 1: luôn > 0 và đơn điệu tăng theo x.

    Positivity cần để tích f(s)·m^γ có nghĩa (mô hình P(click) ∝ interest·popularity).
    Đơn điệu tăng đảm bảo rank theo f(s) ≡ rank theo s ở inference.
    """
    return F.elu(x) + 1.0


def item_popularity(item_degree: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Độ phổ biến chuẩn hoá m_i ∈ (0, 1] từ degree TRAIN.

    m_i = deg_i / max_deg. Clamp min=eps để item degree 0 không cho m^γ = 0
    (tránh tail item bị triệt tiêu hoàn toàn trong huấn luyện).
    """
    m = item_degree.float()
    max_deg = m.max()
    if max_deg <= 0:
        return torch.full_like(m, eps)
    return (m / (max_deg + eps)).clamp(min=eps)


def pd_train_score(raw_scores: torch.Tensor,
                   item_pop: torch.Tensor,
                   gamma: float) -> torch.Tensor:
    """Điểm số confounded lúc TRAIN: ŷ = f(s) · m^γ.

    Args:
        raw_scores: s = <e_u, e_i> cho batch, shape [B].
        item_pop:   m của đúng các item tương ứng trong batch, shape [B].
        gamma:      cường độ khử phổ biến (γ ≥ 0).
    """
    return elu_plus(raw_scores) * item_pop.pow(gamma)
