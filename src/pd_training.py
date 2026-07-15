#!/usr/bin/env python3
"""Training PD-LightGCN (Popularity Deconfounding).

Tự chứa: chỉ TÁI DÙNG (đọc) các thành phần có sẵn — LightGCNRecommender, bpr_loss,
l2_regularization, sample_bpr_batch, evaluate_full_ranking — và KHÔNG sửa file chung
để tránh xung đột với phần baseline do người khác làm.

Khác biệt duy nhất so với training thường: điểm số đưa vào BPR lúc train là
``ŷ = f(s)·m^γ`` (xem src/pd_debias.py). Inference vẫn rank theo ``full_sort_scores``
thô (đã khử phổ biến) nên model lưu ra tương thích thẳng với ``evaluate_test_full.py``.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
import config
from config import get_device, PROJECT_ROOT
from models import LightGCNRecommender
from losses import bpr_loss, l2_regularization
from train import sample_bpr_batch
from metrics import evaluate_full_ranking
from pd_debias import item_popularity, pd_train_score


@torch.no_grad()
def _evaluate(model: LightGCNRecommender,
              edge_index: torch.Tensor,
              mask_pos_items,
              test_items: torch.Tensor,
              item_group: torch.Tensor,
              item_degree: torch.Tensor,
              device: torch.device) -> Dict[str, float]:
    """Rank theo score thô (đã khử phổ biến với model PD) rồi tính full metrics."""
    model.eval()
    scores = model.full_sort_scores(edge_index.to(device)).float()

    valid = test_items >= 0
    if bool(valid.all()):
        s, t, m = scores, test_items.to(device), mask_pos_items
    else:
        idx = valid.nonzero(as_tuple=True)[0]            # trên CPU (test_items ở CPU)
        s = scores[idx.to(scores.device)]                # index phải cùng device với scores
        t = test_items[idx].to(device)
        m = [mask_pos_items[int(i)] for i in idx.tolist()]

    return evaluate_full_ranking(
        scores=s,
        train_user_pos_items=m,
        test_items=t,
        item_group=item_group.to(device),
        item_degree=item_degree.to(device),
        k_list=list(config.K_LIST),
    )


def train_pd_lightgcn(
    data,
    gamma: float,
    num_layers: int = config.NUM_LAYERS,
    num_epochs: int = config.NUM_EPOCHS,
    device: torch.device | None = None,
    eval_every: int = 5,
    patience: int = 10,
    verbose: bool = True,
) -> Tuple[LightGCNRecommender, Dict[str, float]]:
    """Huấn luyện PD-LightGCN với hệ số khử phổ biến ``gamma``.

    Args:
        data: DataProcessor đã nạp.
        gamma: cường độ khử phổ biến (γ ≥ 0). γ=0 ≈ không khử.
        num_layers: số lớp LightGCN (2 hoặc 3).
        num_epochs: số epoch.
        eval_every: đánh giá trên val mỗi bao nhiêu epoch.
        patience: early stopping theo Recall@PRIMARY_K trên val.

    Returns:
        (model tốt nhất theo val, dict metrics test đầy đủ + siêu tham số).
    """
    device = device or get_device()
    if gamma < 0:
        raise ValueError(f"gamma phải >= 0, nhận {gamma}")

    model = LightGCNRecommender(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=config.EMBEDDING_DIM,
        num_layers=num_layers,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    edge_index_train = data.edge_index_train.to(device)
    edges_cpu = data.train_edges.cpu()
    item_pop = item_popularity(data.item_degree).to(device)          # m_i ∈ (0,1]
    test_items = data.get_test_items_tensor()
    val_mask = data.val_user_pos_items                               # mask = train+val

    steps_per_epoch = max(1, len(edges_cpu) // config.BATCH_SIZE)
    primary = config.PRIMARY_K

    best_recall = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    if verbose:
        print(f"🚀 PD-LightGCN | γ={gamma} | layers={num_layers} | "
              f"epochs={num_epochs} | steps/epoch={steps_per_epoch} | device={device}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0
        pbar = range(steps_per_epoch)
        if verbose:
            # Tắt thanh tiến trình khi không phải terminal (tránh ngập log Slurm)
            pbar = tqdm(pbar, desc=f"Epoch {epoch:3d}/{num_epochs}",
                        leave=False, disable=not sys.stdout.isatty())

        for _ in pbar:
            users, pos_items, neg_items = sample_bpr_batch(
                edges_cpu, data.num_items, data.train_user_pos_items, config.BATCH_SIZE
            )
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            pos_s, neg_s, (u0, p0, n0) = model.bpr_forward(
                edge_index_train, users, pos_items, neg_items
            )

            # PD: điểm số confounded lúc train = f(s)·m^γ
            pos_hat = pd_train_score(pos_s, item_pop[pos_items], gamma)
            neg_hat = pd_train_score(neg_s, item_pop[neg_items], gamma)

            loss = bpr_loss(pos_hat, neg_hat)
            reg = l2_regularization(u0, p0, n0, batch_size=users.size(0))
            total = loss + config.WEIGHT_DECAY * reg

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += float(total.item())

        if epoch % eval_every == 0 or epoch == num_epochs:
            val_metrics = _evaluate(
                model, edge_index_train, val_mask, test_items,
                data.item_popularity_group, data.item_degree, device,
            )
            recall = val_metrics[f"Recall@{primary}"]
            if recall > best_recall:
                best_recall, best_epoch = recall, epoch
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if verbose:
                print(f"   epoch {epoch:3d}: loss={running/steps_per_epoch:.4f} "
                      f"val Recall@{primary}={recall:.4f} "
                      f"Coverage@{primary}={val_metrics[f'Coverage@{primary}']:.4f} "
                      f"ARP@{primary}={val_metrics[f'ARP@{primary}']:.4f} "
                      f"(best {best_recall:.4f}@{best_epoch})")
            if no_improve >= patience and best_recall > -1:
                if verbose:
                    print(f"   ⏰ early stopping @ epoch {epoch}")
                break

    # Khôi phục model tốt nhất theo val, đánh giá cuối trên TEST
    model.load_state_dict(best_state)
    test_metrics = _evaluate(
        model, edge_index_train, val_mask, test_items,
        data.item_popularity_group, data.item_degree, device,
    )

    result = {
        "method": "PD-LightGCN",
        "gamma": float(gamma),
        "num_layers": int(num_layers),
        "best_epoch": int(best_epoch),
        "best_val_recall": float(best_recall),
        **{k: float(v) for k, v in test_metrics.items()},
    }
    return model, result


def save_pd_model(model: LightGCNRecommender, model_name: str, result: Dict,
                  models_dir: Path | None = None) -> Path:
    """Lưu model theo đúng format ``evaluate_test_full.py`` đọc được (rank theo full_sort_scores)."""
    models_dir = models_dir or (PROJECT_ROOT / "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pkg = {
        "model_state_dict": model.state_dict(),
        "config": {"embedding_dim": model.embedding_dim, "num_layers": model.num_layers},
        "metadata": {
            "model_name": model_name,
            "lambda_ile": "",
            "dropout_type": "",
            "gamma": result.get("gamma"),
            **{k: v for k, v in result.items() if k.startswith("test_") or k in
               ("Recall@20", "NDCG@20", "Coverage@20", "ARP@20")},
        },
    }
    path = models_dir / f"final_model_{model_name}_{ts}.pt"
    torch.save(pkg, path)
    return path
