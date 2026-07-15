#!/usr/bin/env python3
"""Popularity-Aware LightGCN — training hợp nhất, đầy đủ các bước của 1 dự án AI.

Thành phần method (bật/tắt độc lập để ablation):
  1) ILE (Item Loss Equalization)      — đổi objective (không đổi đồ thị)
  2) Degree-Aware Graph Augmentation   — drop cạnh item phổ biến nhiều hơn (ĐỐI XỨNG)
  3) Contrastive Learning (SGL-style)  — InfoNCE cross-view giữa 2 view augment
  4) Popularity-aware negative sampling (neg ∝ deg^β) — tùy chọn, đẩy TailRecall

Hạ tầng training:
  - VAL để chọn model / early-stopping; TEST chỉ chấm 1 lần ở cuối (KHÔNG peek test).
  - Checkpoint ra disk: <run_id>_latest.pt (resume) + <run_id>_best.pt (best theo val).
  - Resume tự động từ latest nếu có.
  - Logging đầy đủ ra stdout + file logs/popaware/<run_id>.log.
  - Lưu history mỗi epoch ra results/popaware/history_<run_id>.csv (vẽ loss curve).
  - Đánh giá TEST đủ 10 metric qua evaluate_full_ranking.
  - Model cuối lưu format tương thích evaluate_test_full.py.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
import config
from config import get_device, PROJECT_ROOT
from models import LightGCNRecommender
from losses import bpr_loss, l2_regularization
from metrics import evaluate_full_ranking
from ile_losses import ile_loss, compute_degree_aware_dropout_probs
from neg_sampling import build_neg_probs, sample_bpr_batch_popaware


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Augmentation + Contrastive helpers
# --------------------------------------------------------------------------- #
def make_drop_probs(data, dropout_type: str, device: torch.device) -> torch.Tensor:
    if dropout_type == "uniform":
        return torch.full((data.num_items,), float(config.UNIFORM_DROPOUT_P), device=device)
    if dropout_type == "degree_aware":
        return compute_degree_aware_dropout_probs(
            data.item_degree.to(device), config.DROPOUT_P_MIN, config.DROPOUT_P_MAX
        ).to(device)
    raise ValueError(f"dropout_type không hợp lệ: {dropout_type}")


def symmetric_edge_dropout(edge_index: torch.Tensor, drop_probs: torch.Tensor,
                           num_users: int) -> torch.Tensor:
    """Drop cạnh ĐỐI XỨNG: xét cạnh user->item, drop theo drop_probs[item] rồi mirror lại."""
    src, dst = edge_index[0], edge_index[1]
    ui = (src < num_users) & (dst >= num_users)
    fwd = edge_index[:, ui]
    if fwd.size(1) == 0:
        return edge_index
    items = fwd[1] - num_users
    keep = torch.rand(items.size(0), device=edge_index.device) >= drop_probs[items]
    kept = fwd[:, keep]
    if kept.size(1) == 0:
        return edge_index
    rev = torch.stack([kept[1], kept[0]])
    return torch.cat([kept, rev], dim=1)


def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = (z1 @ z2.t()) / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def fmt_metrics(m: dict) -> str:
    """In gọn ĐỦ 10 metric trên một dòng log."""
    return (f"R@10={m['Recall@10']:.4f} R@20={m['Recall@20']:.4f} "
            f"N@10={m['NDCG@10']:.4f} N@20={m['NDCG@20']:.4f} "
            f"TailR@20={m['TailRecall@20']:.4f} Cov@20={m['Coverage@20']:.4f} "
            f"ARP@20={m['ARP@20']:.4f} HeadExp={m['HeadExposure@20']:.4f} "
            f"MidExp={m['MiddleExposure@20']:.4f} TailExp={m['TailExposure@20']:.4f}")


# --------------------------------------------------------------------------- #
# Evaluation (dùng chung cho VAL và TEST — khác nhau ở target items + mask)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _evaluate(model, edge_index, mask_pos_items, target_items, item_group, item_degree, device):
    model.eval()
    scores = model.full_sort_scores(edge_index.to(device)).float()
    valid = target_items >= 0
    if bool(valid.all()):
        s, t, m = scores, target_items.to(device), mask_pos_items
    else:
        idx = valid.nonzero(as_tuple=True)[0]
        s = scores[idx.to(scores.device)]
        t = target_items[idx].to(device)
        m = [mask_pos_items[int(i)] for i in idx.tolist()]
    return evaluate_full_ranking(
        scores=s, train_user_pos_items=m, test_items=t,
        item_group=item_group.to(device), item_degree=item_degree.to(device),
        k_list=list(config.K_LIST),
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_popaware_lightgcn(
    data,
    *,
    use_ile: bool = False,
    lambda_ile: float = config.LAMBDA_ILE,
    aug_main: bool = False,
    use_cl: bool = False,
    lambda_cl: float = config.LAMBDA_CL,
    tau: float = config.TAU,
    dropout_type: str = "degree_aware",
    neg_pop_beta: float = 0.0,
    num_layers: int = config.NUM_LAYERS,
    num_epochs: int = config.NUM_EPOCHS,
    seed: int = config.SEED,
    device: torch.device | None = None,
    eval_every: int = 5,
    patience: int = 10,
    run_id: Optional[str] = None,
    ckpt_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    history_dir: Optional[Path] = None,
    resume: bool = True,
    save_every: int = 5,
    verbose: bool = True,
) -> Tuple[LightGCNRecommender, Dict]:
    """Huấn luyện Popularity-Aware LightGCN.

    total_loss = BPR + wd·L2  [+ λ_ILE·(head−tail)²]  [+ λ_CL·InfoNCE(view1,view2)]
    Negative sampling: uniform (β=0) hoặc ∝ deg^β.

    Chọn model tốt nhất theo VAL Recall@K; đánh giá cuối trên TEST (đủ 10 metric).
    """
    device = device or get_device()
    primary = config.PRIMARY_K
    run_id = run_id or f"popaware_L{num_layers}"

    log_file = (Path(log_dir) / f"{run_id}.log") if log_dir else None
    log = get_logger(f"pa.{run_id}", log_file)

    hparams = dict(use_ile=use_ile, lambda_ile=lambda_ile, aug_main=aug_main, use_cl=use_cl,
                   lambda_cl=lambda_cl, tau=tau, dropout_type=dropout_type,
                   neg_pop_beta=neg_pop_beta, num_layers=num_layers, num_epochs=num_epochs,
                   seed=seed, embedding_dim=config.EMBEDDING_DIM, lr=config.LR,
                   weight_decay=config.WEIGHT_DECAY, batch_size=config.BATCH_SIZE)

    # ----- model / optimizer / RNG -----
    torch.manual_seed(seed)
    model = LightGCNRecommender(data.num_users, data.num_items,
                                config.EMBEDDING_DIM, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    gen = torch.Generator()                       # RNG riêng cho sampling (CPU), để resume được
    gen.manual_seed(seed)

    # ----- data tensors -----
    edge_index = data.edge_index_train.to(device)
    edges_cpu = data.train_edges.cpu()
    item_group = data.item_popularity_group.to(device)
    num_users = data.num_users
    # VAL: target = val items, mask = train positives (KHÔNG peek test)
    val_items = data.get_val_items_tensor()
    train_mask = data.train_user_pos_items
    # TEST: target = test items, mask = train+val
    test_items = data.get_test_items_tensor()
    trainval_mask = data.val_user_pos_items

    neg_probs = build_neg_probs(data.item_degree, neg_pop_beta)
    drop_probs = make_drop_probs(data, dropout_type, device) if (aug_main or use_cl) else None
    steps_per_epoch = max(1, len(edges_cpu) // config.BATCH_SIZE)

    # ----- state (có thể resume) -----
    start_epoch = 1
    best_recall, best_epoch, no_improve = -1.0, 0, 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []

    latest_path = best_path = None
    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
        latest_path = ckpt_dir / f"{run_id}_latest.pt"
        best_path = ckpt_dir / f"{run_id}_best.pt"
        if resume and latest_path.exists():
            ck = torch.load(latest_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optim"])
            gen.set_state(ck["gen_state"])
            best_recall, best_epoch, no_improve = ck["best_recall"], ck["best_epoch"], ck["no_improve"]
            history = ck.get("history", [])
            start_epoch = ck["epoch"] + 1
            if best_path.exists():
                best_state = torch.load(best_path, map_location=device, weights_only=False)["model"]
            log.info(f"↩️  RESUME từ epoch {start_epoch} (best {best_recall:.4f}@{best_epoch})")

    log.info(f"🚀 [{run_id}] ILE={use_ile}(λ={lambda_ile}) aug_main={aug_main} "
             f"CL={use_cl}(λ={lambda_cl},τ={tau}) negβ={neg_pop_beta} drop={dropout_type} "
             f"layers={num_layers} epochs={num_epochs} steps/epoch={steps_per_epoch} device={device}")

    # ----- training loop -----
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        t0 = time.time()
        run_bpr = run_ile = run_cl = 0.0
        it = range(steps_per_epoch)
        if verbose:
            it = tqdm(it, desc=f"[{run_id}] Ep {epoch:3d}/{num_epochs}",
                      leave=False, disable=not sys.stdout.isatty())

        for _ in it:
            users, pos_items, neg_items = sample_bpr_batch_popaware(
                edges_cpu, data.num_items, data.train_user_pos_items,
                config.BATCH_SIZE, neg_probs=neg_probs, generator=gen,
            )
            users = users.to(device); pos_items = pos_items.to(device); neg_items = neg_items.to(device)

            main_edge = symmetric_edge_dropout(edge_index, drop_probs, num_users) if aug_main else edge_index
            pos_s, neg_s, (u0, p0, n0) = model.bpr_forward(main_edge, users, pos_items, neg_items)

            loss = bpr_loss(pos_s, neg_s)
            reg = l2_regularization(u0, p0, n0, batch_size=users.size(0))
            total = loss + config.WEIGHT_DECAY * reg
            run_bpr += float(loss.item())

            if use_ile:
                ile = ile_loss(pos_s, neg_s, pos_items, item_group, device)
                total = total + lambda_ile * ile
                run_ile += float(ile.item())

            if use_cl:
                v1 = symmetric_edge_dropout(edge_index, drop_probs, num_users)
                v2 = symmetric_edge_dropout(edge_index, drop_probs, num_users)
                u1, i1 = model.propagate(v1)
                u2, i2 = model.propagate(v2)
                cl = info_nce(u1[users.unique()], u2[users.unique()], tau) + \
                     info_nce(i1[pos_items.unique()], i2[pos_items.unique()], tau)
                total = total + lambda_cl * cl
                run_cl += float(cl.item())

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        dt = time.time() - t0
        ep_row = {"epoch": epoch, "bpr": run_bpr / steps_per_epoch,
                  "ile": run_ile / steps_per_epoch, "cl": run_cl / steps_per_epoch, "sec": dt}

        # ----- validation (chọn model) -----
        if epoch % eval_every == 0 or epoch == num_epochs:
            vm = _evaluate(model, edge_index, train_mask, val_items,
                           data.item_popularity_group, data.item_degree, device)
            recall = vm[f"Recall@{primary}"]
            ep_row.update({f"val_{k}": v for k, v in vm.items()})
            improved = recall > best_recall
            if improved:
                best_recall, best_epoch, no_improve = recall, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
                if best_path is not None:
                    torch.save({"model": best_state, "epoch": epoch, "val_metrics": vm,
                                "hparams": hparams}, best_path)
            else:
                no_improve += 1
            log.info(f"ep {epoch:3d} | {dt:4.1f}s | BPR={ep_row['bpr']:.4f} ILE={ep_row['ile']:.4f} "
                     f"CL={ep_row['cl']:.4f} | VAL {fmt_metrics(vm)} "
                     f"{'⭐best' if improved else f'(best {best_recall:.4f}@{best_epoch})'}")
        else:
            log.info(f"ep {epoch:3d} | {dt:4.1f}s | BPR={ep_row['bpr']:.4f} "
                     f"ILE={ep_row['ile']:.4f} CL={ep_row['cl']:.4f}")

        history.append(ep_row)

        # ----- checkpoint latest (resume) -----
        if latest_path is not None and (epoch % save_every == 0 or epoch == num_epochs):
            torch.save({"epoch": epoch, "model": model.state_dict(), "optim": optimizer.state_dict(),
                        "gen_state": gen.get_state(), "best_recall": best_recall,
                        "best_epoch": best_epoch, "no_improve": no_improve,
                        "history": history, "hparams": hparams}, latest_path)

        if no_improve >= patience and best_recall > -1:
            log.info(f"⏰ early stopping @ epoch {epoch}")
            break

    # ----- lưu history -----
    if history_dir is not None:
        history_dir = Path(history_dir); history_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(history_dir / f"history_{run_id}.csv", index=False)

    # ----- TEST (1 lần duy nhất, trên model tốt nhất theo val) -----
    model.load_state_dict(best_state)
    test_metrics = _evaluate(model, edge_index, trainval_mask, test_items,
                             data.item_popularity_group, data.item_degree, device)
    log.info(f"✅ [{run_id}] TEST | {fmt_metrics(test_metrics)}")

    result = {"run_id": run_id, **hparams, "best_epoch": best_epoch,
              "best_val_recall": float(best_recall),
              **{k: float(v) for k, v in test_metrics.items()}}
    return model, result


def save_model(model, model_name: str, result: Dict, models_dir: Path | None = None) -> Path:
    """Lưu model cuối theo format evaluate_test_full.py đọc được."""
    import datetime
    models_dir = models_dir or (PROJECT_ROOT / "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pkg = {
        "model_state_dict": model.state_dict(),
        "config": {"embedding_dim": model.embedding_dim, "num_layers": model.num_layers},
        "metadata": {"model_name": model_name, "lambda_ile": result.get("lambda_ile", ""),
                     "dropout_type": result.get("dropout_type", ""),
                     **{k: v for k, v in result.items()
                        if k in ("Recall@20", "NDCG@20", "Coverage@20", "ARP@20", "TailRecall@20")}},
    }
    path = models_dir / f"final_model_{model_name}_{ts}.pt"
    torch.save(pkg, path)
    return path
