#!/usr/bin/env python3
"""train_final_seeds.py — Chạy các cấu hình CHỐT qua nhiều seed để lấy số liệu cuối (mean±std).

Huấn luyện: baseline (LightGCN) + 4 điểm vận hành PopAware (accuracy / BEST / high-tail / fairness),
tất cả ở 2 lớp, trên các seed {42, 0, 1}. Gộp mean±std trên 10 metric để đưa vào báo cáo (mục 7).

Chạy TRÊN SERVER:
    bash run_on_gpu.sh --train train_final_seeds.py
    bash run_on_gpu.sh train_final_seeds.py --seeds 42 --epochs 5   # smoke

Sinh ra:
    results/popaware_final_runs_<ts>.csv      — từng (config,seed)
    results/popaware_final_meanstd_<ts>.csv   — mean & std theo config (số liệu cuối)
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import traceback
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import config
from src.config import get_device, set_seed, PROJECT_ROOT
from src.data_loader import DataProcessor
from src.popaware_training import train_popaware_lightgcn

MAIN_COLS = ["Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "TailRecall@20",
             "Coverage@20", "ARP@20", "TailExposure@20", "MiddleExposure@20", "HeadExposure@20"]

# Các cấu hình CHỐT (đều num_layers=2). name = tên hiển thị.
CONFIGS = [
    dict(key="baseline", name="LightGCN",           use_ile=False, lambda_ile=0.0, use_cl=False, lambda_cl=0.0, beta=0.0),
    dict(key="accuracy", name="PopAware-accuracy",  use_ile=True,  lambda_ile=0.1, use_cl=True,  lambda_cl=0.1, beta=0.5),
    dict(key="best",     name="PopAware-BEST",      use_ile=True,  lambda_ile=1.0, use_cl=True,  lambda_cl=0.1, beta=0.5),
    dict(key="hightail", name="PopAware-high-tail", use_ile=True,  lambda_ile=1.0, use_cl=True,  lambda_cl=0.1, beta=0.0),
    dict(key="fairness", name="PopAware-fairness",  use_ile=True,  lambda_ile=1.0, use_cl=True,  lambda_cl=0.5, beta=0.0),
]


def banner(t, ch="="):
    w = 92
    print(f"\n{ch*w}\n{t.center(w)}\n{ch*w}", flush=True)


def main() -> bool:
    p = argparse.ArgumentParser(description="PopAware final multi-seed runs")
    p.add_argument("--seeds", default="42,0,1")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--tau", type=float, default=config.TAU)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    device = get_device()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "popaware" / "final_runs"; runs_dir.mkdir(parents=True, exist_ok=True)
    dirs = dict(ckpt=PROJECT_ROOT / "checkpoints" / "popaware",
                log=PROJECT_ROOT / "logs" / "popaware",
                hist=PROJECT_ROOT / "results" / "popaware")

    banner("🎲 POPAWARE — MULTI-SEED FINAL RUNS")
    print(f"Start: {datetime.datetime.now():%Y-%m-%d %H:%M:%S} | device={device} | "
          f"seeds={seeds} | layers={args.layers} | epochs={args.epochs}")
    print(f"→ {len(CONFIGS)} cấu hình × {len(seeds)} seed = {len(CONFIGS)*len(seeds)} run")

    banner("📁 LOADING DATA (once)")
    data = DataProcessor()
    print(f"✅ users={data.num_users} items={data.num_items} train={len(data.train_edges)}", flush=True)

    rows = []
    for cfg in CONFIGS:
        for seed in seeds:
            run_id = f"final_{cfg['key']}_L{args.layers}_s{seed}"
            jp = runs_dir / f"{run_id}.json"
            if (not args.no_resume) and jp.exists():
                print(f"⏭️  SKIP {run_id} (đã xong)", flush=True)
                rows.append(json.load(open(jp))); continue
            banner(f"🚀 {cfg['name']} | seed={seed}", "-")
            try:
                set_seed(seed)
                _, res = train_popaware_lightgcn(
                    data=data,
                    use_ile=cfg["use_ile"], lambda_ile=cfg["lambda_ile"],
                    aug_main=False,
                    use_cl=cfg["use_cl"], lambda_cl=cfg["lambda_cl"], tau=args.tau,
                    dropout_type="degree_aware", neg_pop_beta=cfg["beta"],
                    num_layers=args.layers, num_epochs=args.epochs, seed=seed, device=device,
                    verbose=True, run_id=run_id, ckpt_dir=dirs["ckpt"], log_dir=dirs["log"],
                    history_dir=dirs["hist"], resume=not args.no_resume,
                )
                row = {"key": cfg["key"], "Model": cfg["name"], "seed": seed,
                       **{c: res.get(c) for c in MAIN_COLS if c in res}}
                json.dump(row, open(jp, "w"), indent=2, default=float)
                rows.append(row)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ FAIL {run_id}: {e}\n{traceback.format_exc()}", flush=True)

    if not rows:
        print("❌ Không có run nào thành công."); return False

    df = pd.DataFrame(rows)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(out_dir / f"popaware_final_runs_{ts}.csv", index=False)

    # ----- gộp mean ± std theo config -----
    order = [c["key"] for c in CONFIGS]
    name_of = {c["key"]: c["name"] for c in CONFIGS}
    agg_rows = []
    for k in order:
        sub = df[df["key"] == k]
        if sub.empty:
            continue
        r = {"key": k, "Model": name_of[k], "n_seed": len(sub)}
        for c in MAIN_COLS:
            r[f"{c}_mean"] = sub[c].mean()
            r[f"{c}_std"] = sub[c].std(ddof=0)
        agg_rows.append(r)
    agg = pd.DataFrame(agg_rows)
    meanstd_path = out_dir / f"popaware_final_meanstd_{ts}.csv"
    agg.to_csv(meanstd_path, index=False)

    banner("📊 SỐ LIỆU CUỐI (mean ± std trên các seed)")
    key_cols = ["Recall@20", "NDCG@20", "TailRecall@20", "Coverage@20", "ARP@20", "HeadExposure@20"]
    print(f"{'Model':<22}" + "".join(f"{c:>22}" for c in key_cols))
    for _, r in agg.iterrows():
        line = f"{r['Model']:<22}"
        for c in key_cols:
            line += f"{r[f'{c}_mean']:.4f}±{r[f'{c}_std']:.4f}".rjust(22)
        print(line)

    print(f"\n💾 Per-run:  {out_dir / f'popaware_final_runs_{ts}.csv'}")
    print(f"💾 Mean±std: {meanstd_path}")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
