#!/usr/bin/env python3
"""Chạy thí nghiệm PD-LightGCN (Popularity Deconfounding) — phương pháp chính.

Quét γ (cường độ khử phổ biến) × số lớp × seed, huấn luyện, đánh giá TEST với đầy
đủ 10 metric, lưu CSV và (tuỳ chọn) model tương thích evaluate_test_full.py.

Baseline so sánh (MostPopular/BPR-MF/LightGCN/logit-adjust/IPS...) do người khác lo —
script này CHỈ tạo ra các dòng của phương pháp PD.

Cách chạy:
    # đầy đủ (bảng chính E3):
    ./run_on_gpu.sh --train run_pd_experiments.py
    # smoke test nhanh (kiểm tra code + xu hướng):
    python run_pd_experiments.py --epochs 5 --gammas 0,0.2 --layers 2 --seeds 42 --no-save-model
"""

from __future__ import annotations

import argparse
import datetime
import sys
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


def parse_list(s: str, cast):
    return [cast(x) for x in str(s).split(",") if str(x).strip() != ""]


def main() -> bool:
    p = argparse.ArgumentParser(description="PD-LightGCN experiments")
    p.add_argument("--gammas", default="0,0.02,0.05,0.1,0.2,0.5",
                   help="danh sách γ, phân tách bằng dấu phẩy")
    p.add_argument("--layers", default="2,3", help="danh sách số lớp LightGCN")
    p.add_argument("--seeds", default=str(config.SEED), help="danh sách seed")
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--no-save-model", action="store_true",
                   help="không lưu file model (dùng cho smoke test)")
    args = p.parse_args()

    # Line-buffer stdout để log Slurm hiện tiến trình trực tiếp (không phải chờ buffer đầy)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    gammas = parse_list(args.gammas, float)
    layers = parse_list(args.layers, int)
    seeds = parse_list(args.seeds, int)
    device = get_device()

    print("=" * 80)
    print("🎯 PD-LightGCN — POPULARITY DECONFOUNDING")
    print(f"   γ = {gammas} | layers = {layers} | seeds = {seeds} | "
          f"epochs = {args.epochs} | device = {device}")
    print("=" * 80)

    print("\n📁 Nạp dữ liệu (một lần)...")
    data = DataProcessor()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for seed in seeds:
        for num_layers in layers:
            for gamma in gammas:
                set_seed(seed)  # reset trước mỗi run để công bằng
                print(f"\n--- seed={seed} | layers={num_layers} | γ={gamma} ---")
                model, result = train_pd_lightgcn(
                    data=data, gamma=gamma, num_layers=num_layers,
                    num_epochs=args.epochs, device=device, verbose=True,
                )
                row = {
                    "Model": f"PD-LightGCN_g{gamma}_L{num_layers}",
                    "seed": seed, "gamma": gamma, "num_layers": num_layers,
                    "best_epoch": result["best_epoch"],
                    **{c: result.get(c) for c in MAIN_COLS if c in result},
                }
                rows.append(row)
                print(f"   ✅ TEST Recall@20={row['Recall@20']:.4f}  "
                      f"NDCG@20={row['NDCG@20']:.4f}  Coverage@20={row['Coverage@20']:.4f}  "
                      f"ARP@20={row['ARP@20']:.4f}  HeadExp@20={row['HeadExposure@20']:.4f}")

                if not args.no_save_model:
                    path = save_pd_model(model, row["Model"], result)
                    print(f"   💾 saved {path.name}")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    ordered = ["Model", "seed", "gamma", "num_layers", "best_epoch"] + \
              [c for c in MAIN_COLS if c in df.columns]
    df = df[ordered]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"pd_results_{ts}.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 120)
    print("📊 KẾT QUẢ PD-LightGCN (test set)")
    print("=" * 120)
    with pd.option_context("display.max_columns", None, "display.width", 220,
                           "display.float_format", lambda v: f"{v:.4f}"):
        show = ["Model", "gamma", "num_layers", "Recall@20", "NDCG@20",
                "Coverage@20", "ARP@20", "HeadExposure@20", "TailExposure@20"]
        print(df[[c for c in show if c in df.columns]].to_string(index=False))

    # Gợi ý xu hướng: γ tăng thì ARP nên giảm, Coverage nên tăng
    print("\n🔎 Kiểm tra xu hướng theo γ (kỳ vọng: ARP↓, Coverage↑ khi γ↑):")
    for num_layers in layers:
        sub = df[df["num_layers"] == num_layers].groupby("gamma")[["Recall@20", "Coverage@20", "ARP@20"]].mean()
        if len(sub) > 1:
            print(f"\n   [layers={num_layers}]")
            print(sub.to_string())

    print(f"\n💾 Kết quả: {out_path}")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
