#!/usr/bin/env python3
"""train_sweep_popaware.py — Quét siêu tham số cho Popularity-Aware LightGCN + chọn theo frontier.

Quét (Cartesian) λ_ILE × λ_CL × τ × layers × neg_pop_beta cho một cấu hình gốc (mặc định
"full" = ILE+Aug+CL), huấn luyện đầy đủ (VAL chọn best, TEST 10 metric), rồi CHỌN cấu hình
tốt nhất theo quy tắc frontier: max Coverage@20 với ràng buộc Recall@20 giảm ≤ drop-tol so
với baseline (cùng số lớp).

Đặc điểm hạ tầng:
  - Resume 2 mức: (a) run-level — run đã xong (có results/popaware/runs/<run_id>.json) thì BỎ QUA;
    (b) epoch-level — run dở resume từ checkpoint latest.
  - Logging đầy đủ ra stdout + file/run; history/checkpoint/best-weight lưu đĩa.

Chạy TRÊN SERVER:
    bash run_on_gpu.sh --train train_sweep_popaware.py           # full sweep qua Slurm/A100
    bash run_on_gpu.sh train_sweep_popaware.py --epochs 5 --lambda-ile-grid 0.5 --layers-grid 2 --neg-pop-beta-grid 0  # smoke
"""

from __future__ import annotations

import argparse
import datetime
import itertools
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
from src.models import LightGCNRecommender
from src.popaware_training import train_popaware_lightgcn, save_model

MAIN_COLS = ["Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "TailRecall@20",
             "Coverage@20", "ARP@20", "TailExposure@20", "MiddleExposure@20", "HeadExposure@20"]

BASE = {
    "baseline":     dict(use_ile=False, aug_main=False, use_cl=False),
    "ile":          dict(use_ile=True,  aug_main=False, use_cl=False),
    "degreeaug":    dict(use_ile=False, aug_main=True,  use_cl=False),
    "degreeaug_cl": dict(use_ile=False, aug_main=False, use_cl=True),
    "full":         dict(use_ile=True,  aug_main=False, use_cl=True),
}


def flist(s, cast):
    return [cast(x) for x in str(s).split(",") if str(x).strip() != ""]


def banner(t, ch="="):
    w = 92
    print(f"\n{ch*w}\n{t.center(w)}\n{ch*w}", flush=True)


def rid(base, li, lc, t, L, b, seed):
    return f"{base}_L{L}_ile{li}_cl{lc}_tau{t}_beta{b}_s{seed}".replace(".", "p")


def train_one(data, base_cfg, *, li, lc, tau, layers, beta, seed, epochs, device,
              run_id, dirs, resume):
    _, result = train_popaware_lightgcn(
        data=data,
        use_ile=base_cfg["use_ile"], lambda_ile=li,
        aug_main=base_cfg["aug_main"],
        use_cl=base_cfg["use_cl"], lambda_cl=lc, tau=tau,
        dropout_type="degree_aware", neg_pop_beta=beta,
        num_layers=layers, num_epochs=epochs, seed=seed, device=device, verbose=True,
        run_id=run_id, ckpt_dir=dirs["ckpt"], log_dir=dirs["log"],
        history_dir=dirs["hist"], resume=resume,
    )
    return result


def main() -> bool:
    p = argparse.ArgumentParser(description="PopAware-LightGCN hyperparameter sweep")
    p.add_argument("--base-config", default="full", choices=list(BASE))
    p.add_argument("--lambda-ile-grid", default="0.1,0.5,1.0")
    p.add_argument("--lambda-cl-grid", default="0.1,0.5")
    p.add_argument("--tau-grid", default="0.2")
    p.add_argument("--layers-grid", default="2,3")
    p.add_argument("--neg-pop-beta-grid", default="0,0.5")
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--drop-tol", type=float, default=0.05)
    p.add_argument("--no-baseline", action="store_true", help="không train baseline tham chiếu")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    device = get_device()
    base_cfg = BASE[args.base_config]

    # Collapse các chiều không liên quan để tránh run trùng
    li_grid = flist(args.lambda_ile_grid, float) if base_cfg["use_ile"] else [config.LAMBDA_ILE]
    lc_grid = flist(args.lambda_cl_grid, float) if base_cfg["use_cl"] else [config.LAMBDA_CL]
    tau_grid = flist(args.tau_grid, float) if base_cfg["use_cl"] else [config.TAU]
    beta_grid = flist(args.neg_pop_beta_grid, float)
    layers_grid = flist(args.layers_grid, int)

    combos = list(itertools.product(layers_grid, li_grid, lc_grid, tau_grid, beta_grid))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "popaware" / "runs"; runs_dir.mkdir(parents=True, exist_ok=True)
    dirs = {"ckpt": PROJECT_ROOT / "checkpoints" / "popaware",
            "log": PROJECT_ROOT / "logs" / "popaware",
            "hist": PROJECT_ROOT / "results" / "popaware"}

    banner("🔬 POPAWARE-LIGHTGCN — HYPERPARAMETER SWEEP")
    print(f"Start: {datetime.datetime.now():%Y-%m-%d %H:%M:%S} | device={device} | "
          f"base={args.base_config} | epochs={args.epochs} | seed={args.seed}")
    print(f"Grid: layers={layers_grid} λ_ILE={li_grid} λ_CL={lc_grid} τ={tau_grid} β={beta_grid}")
    print(f"→ {len(combos)} run sweep" + ("" if args.no_baseline else f" + {len(layers_grid)} baseline"))

    banner("📁 LOADING DATA (once)")
    data = DataProcessor()
    print(f"✅ users={data.num_users} items={data.num_items} train={len(data.train_edges)}", flush=True)

    def run_cached(base_name, base_c, li, lc, tau, L, beta):
        run_id = rid(base_name, li, lc, tau, L, beta, args.seed)
        jp = runs_dir / f"{run_id}.json"
        if (not args.no_resume) and jp.exists():
            print(f"⏭️  SKIP {run_id} (đã xong)", flush=True)
            return json.load(open(jp)), run_id
        try:
            set_seed(args.seed)
            res = train_one(data, base_c, li=li, lc=lc, tau=tau, layers=L, beta=beta,
                            seed=args.seed, epochs=args.epochs, device=device,
                            run_id=run_id, dirs=dirs, resume=not args.no_resume)
            res["base_config"] = base_name
            json.dump(res, open(jp, "w"), indent=2, default=float)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return res, run_id
        except Exception as e:
            print(f"❌ FAIL {run_id}: {e}\n{traceback.format_exc()}", flush=True)
            return None, run_id

    # ----- baseline tham chiếu (mỗi số lớp) -----
    baseline_recall = {}
    baseline_rows = []
    if not args.no_baseline:
        for L in layers_grid:
            banner(f"🚀 baseline (layers={L})", "-")
            res, _ = run_cached("baseline", BASE["baseline"], config.LAMBDA_ILE,
                                config.LAMBDA_CL, config.TAU, L, 0.0)
            if res:
                baseline_recall[L] = res["Recall@20"]
                baseline_rows.append({"Model": f"baseline_L{L}", **res})

    # ----- sweep -----
    rows = []
    for i, (L, li, lc, tau, beta) in enumerate(combos, 1):
        banner(f"🚀 [{i}/{len(combos)}] {args.base_config} L={L} λ_ILE={li} λ_CL={lc} τ={tau} β={beta}", "-")
        res, run_id = run_cached(args.base_config, base_cfg, li, lc, tau, L, beta)
        if res:
            rows.append({"Model": run_id, **res})

    if not rows:
        print("❌ Không có run sweep nào thành công.")
        return False

    df = pd.DataFrame(baseline_rows + rows)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"popaware_sweep_{ts}.csv"
    keep = ["Model", "base_config", "num_layers", "lambda_ile", "lambda_cl", "tau",
            "neg_pop_beta", "best_epoch"] + MAIN_COLS
    df[[c for c in keep if c in df.columns]].to_csv(out_path, index=False)

    banner("📊 SWEEP RESULTS (test)")
    with pd.option_context("display.max_columns", None, "display.width", 260,
                           "display.float_format", lambda v: f"{v:.4f}"):
        show = ["Model", "Recall@20", "NDCG@20", "TailRecall@20", "Coverage@20",
                "ARP@20", "HeadExposure@20"]
        print(df[[c for c in show if c in df.columns]].to_string(index=False))

    # ----- chọn theo frontier -----
    banner(f"🎯 CHỌN CẤU HÌNH (max Coverage@20 | Recall@20 giảm ≤ {args.drop_tol*100:.0f}% vs baseline)", "-")
    sweep_df = pd.DataFrame(rows)
    winner = None
    for L in sorted(sweep_df["num_layers"].unique()):
        subset = sweep_df[sweep_df["num_layers"] == L]
        base_r = baseline_recall.get(L)
        if base_r is None:
            base_r = subset["Recall@20"].max()  # fallback nếu không có baseline
            note = "(không có baseline, dùng max Recall trong nhóm)"
        else:
            note = f"(baseline Recall@20={base_r:.4f})"
        thr = base_r * (1 - args.drop_tol)
        cand = subset[subset["Recall@20"] >= thr]
        print(f"\n[layers={L}] ngưỡng Recall≥{thr:.4f} {note} — {len(cand)}/{len(subset)} ứng viên đạt")
        if cand.empty:
            print("   ⚠️  Không cấu hình nào giữ được Recall trong ngưỡng.")
            continue
        best = cand.loc[cand["Coverage@20"].idxmax()]
        print(f"   ✅ {best['Model']}: Recall@20={best['Recall@20']:.4f} "
              f"NDCG@20={best['NDCG@20']:.4f} TailRec@20={best['TailRecall@20']:.4f} "
              f"Coverage@20={best['Coverage@20']:.4f} ARP@20={best['ARP@20']:.4f} "
              f"HeadExp@20={best['HeadExposure@20']:.4f}")
        if winner is None or best["Coverage@20"] > winner["Coverage@20"]:
            winner = best

    # ----- lưu model của cấu hình thắng (từ best checkpoint đã có trên đĩa) -----
    if winner is not None:
        try:
            best_ckpt = dirs["ckpt"] / f"{winner['Model']}_best.pt"
            if best_ckpt.exists():
                ck = torch.load(best_ckpt, map_location=device, weights_only=False)
                m = LightGCNRecommender(data.num_users, data.num_items,
                                        config.EMBEDDING_DIM, int(winner["num_layers"])).to(device)
                m.load_state_dict(ck["model"])
                path = save_model(m, "PopAware-BEST", dict(winner))
                print(f"\n💾 Model cấu hình thắng lưu: {path.name}")
        except Exception as e:
            print(f"⚠️  Không lưu được model thắng: {e}")

    banner("🎉 DONE")
    print(f"💾 Sweep CSV: {out_path}")
    print(f"💾 Per-run JSON: {runs_dir}  |  logs: {dirs['log']}  |  history: {dirs['hist']}")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
