#!/usr/bin/env python3
"""train_all_popaware.py — Pipeline PHƯƠNG PHÁP: Popularity-Aware LightGCN.

Huấn luyện ablation của method (ILE + Degree-Aware Graph Augmentation + Contrastive
Learning), đo đủ 10 metric trên TEST, và so sánh delta với baseline LightGCN theo đúng
kỳ vọng giảm popularity bias: TailRecall↑, Coverage↑, ARP↓, HeadExp↓, Recall/NDCG không tụt mạnh.

Chạy TRÊN SERVER (không chạy laptop):
    bash run_on_gpu.sh --train train_all_popaware.py                 # full ablation qua Slurm/A100 (~1h)
    bash run_on_gpu.sh train_all_popaware.py --epochs 5 --configs baseline,ile   # smoke (login node, chậm)

Sinh ra:
    results/popaware_results_<ts>.csv     — mọi cấu hình, đủ 10 metric
    models/final_model_*.pt               — tương thích evaluate_test_full.py

Baseline ngoài (MostPopular/BPR-MF/logit-adjust/IPS...) do người khác lo; file này chỉ lo method.
"""

from __future__ import annotations

import argparse
import datetime
import sys
import traceback
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import config
from src.config import get_device, set_seed, PROJECT_ROOT
from src.data_loader import DataProcessor
from src.popaware_training import train_popaware_lightgcn, save_model

MAIN_COLS = [
    "Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "TailRecall@20",
    "Coverage@20", "ARP@20", "TailExposure@20", "MiddleExposure@20", "HeadExposure@20",
]

# Ablation của Popularity-Aware LightGCN. Baseline PHẢI đứng đầu để tính delta.
def build_configs(args):
    return {
        "baseline":        dict(name="LightGCN",                use_ile=False, aug_main=False, use_cl=False),
        "ile":             dict(name="LightGCN+ILE",            use_ile=True,  aug_main=False, use_cl=False),
        "degreeaug":       dict(name="LightGCN+DegreeAug",      use_ile=False, aug_main=True,  use_cl=False),
        "degreeaug_cl":    dict(name="LightGCN+DegreeAug+CL",   use_ile=False, aug_main=False, use_cl=True),
        "full":            dict(name="PopAware-Full(ILE+Aug+CL)", use_ile=True, aug_main=False, use_cl=True),
    }


def banner(title, ch="="):
    w = 88
    print(f"\n{ch*w}\n{title.center(w)}\n{ch*w}")


def main() -> bool:
    p = argparse.ArgumentParser(description="Popularity-Aware LightGCN training")
    p.add_argument("--configs", default="baseline,ile,degreeaug,degreeaug_cl,full",
                   help="danh sách cấu hình, phân tách bằng phẩy")
    p.add_argument("--lambda-ile", type=float, default=config.LAMBDA_ILE)
    p.add_argument("--lambda-cl", type=float, default=config.LAMBDA_CL)
    p.add_argument("--tau", type=float, default=config.TAU)
    p.add_argument("--dropout-type", default="degree_aware", choices=["degree_aware", "uniform"])
    p.add_argument("--neg-pop-beta", type=float, default=0.0,
                   help="popularity-aware negative sampling: neg ∝ deg^β (0=uniform)")
    p.add_argument("--layers", type=int, default=config.NUM_LAYERS)
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--no-save-model", action="store_true")
    p.add_argument("--no-resume", action="store_true", help="bỏ qua checkpoint, train lại từ đầu")
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    device = get_device()
    all_cfgs = build_configs(args)
    chosen = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in chosen:
        if c not in all_cfgs:
            print(f"❌ Cấu hình không hợp lệ: {c}. Hợp lệ: {list(all_cfgs)}")
            return False

    banner("🎬 POPULARITY-AWARE LIGHTGCN — TRAINING PIPELINE")
    print(f"Start: {datetime.datetime.now():%Y-%m-%d %H:%M:%S} | Device: {device} | "
          f"PyTorch {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"configs={chosen} | λ_ILE={args.lambda_ile} | λ_CL={args.lambda_cl} | τ={args.tau} | "
          f"drop={args.dropout_type} | layers={args.layers} | epochs={args.epochs} | seed={args.seed}")

    banner("📁 LOADING DATA (once)")
    try:
        data = DataProcessor()
        assert data.num_users > 0 and data.num_items > 0 and len(data.train_edges) > 0
    except Exception as e:
        print(f"❌ Lỗi nạp dữ liệu: {e}\n{traceback.format_exc()}")
        return False
    print(f"✅ users={data.num_users} items={data.num_items} train={len(data.train_edges)} "
          f"val={len(data.val_edges)} test={len(data.test_edges)}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []

    for key in chosen:
        cfg = all_cfgs[key]
        banner(f"🚀 {cfg['name']}  [{key}]", "-")
        try:
            set_seed(args.seed)
            run_id = f"{key}_L{args.layers}_seed{args.seed}"
            model, result = train_popaware_lightgcn(
                data=data,
                use_ile=cfg["use_ile"], lambda_ile=args.lambda_ile,
                aug_main=cfg["aug_main"],
                use_cl=cfg["use_cl"], lambda_cl=args.lambda_cl, tau=args.tau,
                dropout_type=args.dropout_type, neg_pop_beta=args.neg_pop_beta,
                num_layers=args.layers, num_epochs=args.epochs, seed=args.seed,
                device=device, verbose=True,
                run_id=run_id,
                ckpt_dir=PROJECT_ROOT / "checkpoints" / "popaware",
                log_dir=PROJECT_ROOT / "logs" / "popaware",
                history_dir=PROJECT_ROOT / "results" / "popaware",
                resume=not args.no_resume,
            )
            row = {"config": key, "Model": cfg["name"],
                   **{c: result.get(c) for c in MAIN_COLS if c in result}}
            rows.append(row)
            print(f"✅ {cfg['name']}: Recall@20={row['Recall@20']:.4f} NDCG@20={row['NDCG@20']:.4f} "
                  f"TailRec@20={row['TailRecall@20']:.4f} Cov@20={row['Coverage@20']:.4f} "
                  f"ARP@20={row['ARP@20']:.4f} HeadExp@20={row['HeadExposure@20']:.4f}")
            if not args.no_save_model:
                path = save_model(model, cfg["name"], result)
                print(f"   💾 {path.name}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ FAIL {cfg['name']}: {e}\n{traceback.format_exc()}")
            failures.append({"name": cfg["name"], "error": str(e)})

    if not rows:
        print("❌ Không có cấu hình nào thành công.")
        return False

    df = pd.DataFrame(rows)
    df = df[["config", "Model"] + [c for c in MAIN_COLS if c in df.columns]]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"popaware_results_{ts}.csv"
    df.to_csv(out_path, index=False)

    banner("📊 KẾT QUẢ (test set)")
    with pd.option_context("display.max_columns", None, "display.width", 240,
                           "display.float_format", lambda v: f"{v:.4f}"):
        show = ["Model", "Recall@20", "NDCG@20", "TailRecall@20", "Coverage@20",
                "ARP@20", "HeadExposure@20", "TailExposure@20"]
        print(df[[c for c in show if c in df.columns]].to_string(index=False))

    # So sánh delta với baseline theo kỳ vọng giảm bias
    if "baseline" in df["config"].values:
        banner("🎯 DELTA vs BASELINE (kỳ vọng: TailRecall↑ Coverage↑ ARP↓ HeadExp↓, Recall/NDCG ~giữ)", "-")
        b = df[df["config"] == "baseline"].iloc[0]
        for _, r in df.iterrows():
            if r["config"] == "baseline":
                continue
            def d(col, worse_if_up=False):
                delta = r[col] - b[col]
                pct = 100 * delta / b[col] if b[col] else float("nan")
                good = (delta < 0) if worse_if_up else (delta > 0)
                mark = "✅" if good else "⚠️"
                return f"{col} {delta:+.4f} ({pct:+.1f}%){mark}"
            print(f"\n  {r['Model']}:")
            print(f"    bias↓ : {d('TailRecall@20')} | {d('Coverage@20')} | "
                  f"{d('ARP@20', worse_if_up=True)} | {d('HeadExposure@20', worse_if_up=True)}")
            print(f"    acc   : {d('Recall@20')} | {d('NDCG@20')}   (mong muốn không tụt >5%)")

    banner("🎉 DONE")
    print(f"Thành công: {len(rows)} | Thất bại: {len(failures)}")
    for f in failures:
        print(f"   ❌ {f['name']}: {f['error']}")
    print(f"💾 {out_path}")
    print(f"💾 models/ (dùng evaluate_test_full.py để gộp so sánh với baseline nhóm)")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
