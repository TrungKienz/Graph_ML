#!/usr/bin/env python3
"""train_all_pd.py — Pipeline huấn luyện PHƯƠNG PHÁP CHÍNH: PD-LightGCN.

Mục tiêu: giảm bias vào item quá phổ biến bằng Popularity Deconfounding (PD/PDA,
Zhang et al. SIGIR'21). Đây là file "train_all" cho riêng phương pháp — baseline so
sánh (MostPopular/BPR-MF/LightGCN/logit-adjust/IPS...) do người khác lo.

Chạy TRÊN SERVER (không chạy laptop):
    bash run_on_gpu.sh --train train_all_pd.py        # full sweep qua Slurm/A100
    bash run_on_gpu.sh train_all_pd.py --epochs 5 --gammas 0,0.2 --layers 2   # smoke (login node)

Sinh ra:
    results/pd_results_<ts>.csv   — mọi cấu hình, đủ 10 metric (+ seed/gamma/layers)
    models/final_model_PD-LightGCN_*.pt — tương thích evaluate_test_full.py để gộp so sánh
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
from src.pd_training import train_pd_lightgcn, save_pd_model

MAIN_COLS = [
    "Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "TailRecall@20",
    "Coverage@20", "ARP@20", "TailExposure@20", "MiddleExposure@20", "HeadExposure@20",
]


def banner(title: str, ch: str = "="):
    w = 80
    print(f"\n{ch * w}\n{title.center(w)}\n{ch * w}")


def parse_list(s, cast):
    return [cast(x) for x in str(s).split(",") if str(x).strip() != ""]


def log_system_info(device):
    banner("🖥️  SYSTEM INFO")
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB) | "
              f"CUDA {torch.version.cuda}")
    print(f"Python: {sys.version.split()[0]} | Project: {PROJECT_ROOT}")
    print(f"embedding_dim={config.EMBEDDING_DIM} | batch={config.BATCH_SIZE} | "
          f"lr={config.LR} | weight_decay={config.WEIGHT_DECAY} | K_LIST={config.K_LIST}")


def recommend_gamma(df: pd.DataFrame, drop_tol: float):
    """Chọn γ tốt nhất theo quy tắc frontier: max Coverage@20 với Recall@20 giảm ≤ drop_tol
    so với γ=0 (cùng số lớp). Trung bình trên các seed."""
    banner("🎯 KHUYẾN NGHỊ γ (max Coverage@20 với Recall giảm ≤ "
           f"{drop_tol*100:.0f}%)", "-")
    agg = df.groupby(["num_layers", "gamma"])[["Recall@20", "Coverage@20", "ARP@20",
                                               "HeadExposure@20"]].mean().reset_index()
    for L in sorted(agg["num_layers"].unique()):
        sub = agg[agg["num_layers"] == L].sort_values("gamma")
        base = sub[sub["gamma"] == 0.0]
        if base.empty:
            base = sub.iloc[[0]]
        r0 = float(base["Recall@20"].iloc[0])
        thresh = r0 * (1.0 - drop_tol)
        cand = sub[(sub["gamma"] > 0) & (sub["Recall@20"] >= thresh)]
        print(f"\n[layers={L}] baseline(γ=0) Recall@20={r0:.4f} → ngưỡng Recall≥{thresh:.4f}")
        if cand.empty:
            print("   ⚠️  Không có γ>0 nào giữ được Recall trong ngưỡng → cần γ nhỏ hơn "
                  "hoặc PD chưa phù hợp ở mức layers này.")
            continue
        best = cand.loc[cand["Coverage@20"].idxmax()]
        print(f"   ✅ γ*={best['gamma']}: Recall@20={best['Recall@20']:.4f} "
              f"(Δ={100*(best['Recall@20']-r0)/r0:+.1f}%), "
              f"Coverage@20={best['Coverage@20']:.4f} "
              f"(Δ={100*(best['Coverage@20']-float(base['Coverage@20'].iloc[0]))/float(base['Coverage@20'].iloc[0]):+.1f}%), "
              f"ARP@20={best['ARP@20']:.4f} (baseline {float(base['ARP@20'].iloc[0]):.4f})")


def main() -> bool:
    p = argparse.ArgumentParser(description="PD-LightGCN training pipeline")
    p.add_argument("--gammas", default="0,0.02,0.05,0.1,0.2,0.5")
    p.add_argument("--layers", default="2,3")
    # Mặc định 1 seed (retrain nhanh ~1h qua --train). Bảng headline: đổi thành "42,0,1"
    # (hoặc chạy quick mode: bash run_on_gpu.sh train_all_pd.py --seeds 42,0,1 ... nhưng chậm trên CPU login node).
    p.add_argument("--seeds", default=str(config.SEED))
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--drop-tol", type=float, default=0.05, help="ngưỡng cho phép Recall giảm")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--no-save-model", action="store_true")
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)   # log Slurm hiện tiến trình trực tiếp
    except Exception:
        pass

    gammas = parse_list(args.gammas, float)
    layers = parse_list(args.layers, int)
    seeds = parse_list(args.seeds, int)
    device = get_device()

    banner("🎬 PD-LightGCN TRAINING PIPELINE (Popularity Deconfounding)")
    print(f"Start: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    log_system_info(device)
    print(f"\nγ = {gammas}\nlayers = {layers}\nseeds = {seeds}\nepochs = {args.epochs}")
    print(f"Tổng số run: {len(gammas) * len(layers) * len(seeds)}")

    banner("📁 LOADING DATA (once)")
    try:
        data = DataProcessor()
        assert data.num_users > 0 and data.num_items > 0
        assert len(data.train_edges) > 0
    except Exception as e:
        print(f"❌ Lỗi nạp dữ liệu: {e}\n{traceback.format_exc()}")
        return False
    print(f"✅ users={data.num_users} items={data.num_items} "
          f"train={len(data.train_edges)} val={len(data.val_edges)} test={len(data.test_edges)}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []

    for seed in seeds:
        for L in layers:
            for g in gammas:
                tag = f"seed={seed} layers={L} γ={g}"
                banner(f"🚀 RUN: {tag}", "-")
                try:
                    set_seed(seed)   # reset RNG trước mỗi run để công bằng & tái lập
                    model, result = train_pd_lightgcn(
                        data=data, gamma=g, num_layers=L,
                        num_epochs=args.epochs, device=device, verbose=True,
                    )
                    row = {
                        "Model": f"PD-LightGCN_g{g}_L{L}",
                        "seed": seed, "gamma": g, "num_layers": L,
                        "best_epoch": result["best_epoch"],
                        **{c: result.get(c) for c in MAIN_COLS if c in result},
                    }
                    rows.append(row)
                    print(f"✅ {tag} → Recall@20={row['Recall@20']:.4f} "
                          f"NDCG@20={row['NDCG@20']:.4f} Coverage@20={row['Coverage@20']:.4f} "
                          f"ARP@20={row['ARP@20']:.4f} HeadExp@20={row['HeadExposure@20']:.4f}")
                    if not args.no_save_model:
                        # chỉ lưu model cho seed đầu để đỡ tốn đĩa; đổi nếu cần mọi seed
                        if seed == seeds[0]:
                            path = save_pd_model(model, row["Model"], result)
                            print(f"   💾 {path.name}")
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"❌ FAIL {tag}: {e}\n{traceback.format_exc()}")
                    failures.append({"tag": tag, "error": str(e)})
                    continue

    if not rows:
        print("❌ Không có run nào thành công.")
        for f in failures:
            print(f"   {f['tag']}: {f['error']}")
        return False

    df = pd.DataFrame(rows)
    ordered = ["Model", "seed", "gamma", "num_layers", "best_epoch"] + \
              [c for c in MAIN_COLS if c in df.columns]
    df = df[ordered]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"pd_results_{ts}.csv"
    df.to_csv(out_path, index=False)

    banner("📊 KẾT QUẢ PD-LightGCN (test set, trung bình theo seed)")
    agg = df.groupby(["num_layers", "gamma"])[[c for c in MAIN_COLS if c in df.columns]].mean()
    with pd.option_context("display.max_columns", None, "display.width", 220,
                           "display.float_format", lambda v: f"{v:.4f}"):
        show = ["Recall@20", "NDCG@20", "Coverage@20", "ARP@20",
                "HeadExposure@20", "TailExposure@20"]
        print(agg[[c for c in show if c in agg.columns]].to_string())

    recommend_gamma(df, args.drop_tol)

    banner("🎉 DONE")
    print(f"Thành công: {len(rows)} run | Thất bại: {len(failures)}")
    for f in failures:
        print(f"   ❌ {f['tag']}: {f['error']}")
    print(f"💾 Kết quả: {out_path}")
    print(f"💾 Models: {PROJECT_ROOT / 'models'} (dùng evaluate_test_full.py để gộp so sánh với baseline)")
    print(f"End: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
