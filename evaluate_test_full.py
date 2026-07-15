#!/usr/bin/env python3
"""Re-evaluate trained LightGCN models on the TEST set with the FULL metric set.

Vì sao có script này
--------------------
Pipeline training ĐÃ tính đủ mọi metric qua ``src.metrics.evaluate_full_ranking``
(Recall@10/20, NDCG@10/20, TailRecall@20, Coverage@20, ARP@20 và 3 exposure),
nhưng các runner chỉ ghi 4 metric ra CSV rồi bỏ phần còn lại. Script này nạp lại
các model đã lưu trong ``models/final_model_*.pt`` và tính lại đầy đủ — KHÔNG train
lại — rồi xuất CSV có đúng bộ cột như ``Graph_ML/results/metrics/main_results.csv``:

    Recall@10, NDCG@10, Recall@20, NDCG@20, TailRecall@20,
    Coverage@20, ARP@20, TailExposure@20, MiddleExposure@20, HeadExposure@20

Cách chạy
---------
    python evaluate_test_full.py                 # mask = train+val (mặc định)
    python evaluate_test_full.py --mask train    # mask chỉ train (khớp chữ notebook)
    ./run_on_gpu.sh evaluate_test_full.py         # chạy trên cluster (login node/CPU là đủ)

Kết quả lưu vào ``results/``:
    * full_test_metrics_<timestamp>.csv    -> các model đã train (kèm lambda/dropout)
    * full_comparison_<timestamp>.csv      -> gộp thêm baseline từ main_results.csv
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd
import torch

# Cho phép import package src
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import config
from src.config import get_device, set_seed, PROJECT_ROOT
from src.data_loader import DataProcessor
from src.models import LightGCNRecommender
from src.metrics import evaluate_full_ranking


# Thứ tự cột khớp Graph_ML/results/metrics/main_results.csv
MAIN_COLS = [
    "Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "TailRecall@20",
    "Coverage@20", "ARP@20", "TailExposure@20", "MiddleExposure@20", "HeadExposure@20",
]

# main_results.csv của Graph_ML (baseline MostPopular / BPR-MF không có model lưu lại,
# nên lấy trực tiếp từ file tham chiếu này để đưa vào bảng so sánh).
REFERENCE_CSV = PROJECT_ROOT / "Graph_ML" / "results" / "metrics" / "main_results.csv"


def find_final_models(models_dir: Path) -> list[Path]:
    """Liệt kê file final_model_*.pt bằng iterdir (tránh lỗi glob '**' trên Python <3.13)."""
    if not models_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục models: {models_dir}")

    files = [
        f for f in models_dir.iterdir()
        if f.is_file() and f.name.startswith("final_model_") and f.name.endswith(".pt")
    ]
    if not files:
        raise FileNotFoundError(f"Không có file final_model_*.pt nào trong {models_dir}")

    # Nếu một model_name có nhiều bản (train nhiều lần), giữ bản mới nhất theo mtime.
    latest: dict[str, Path] = {}
    for f in files:
        pkg = torch.load(f, map_location="cpu")
        name = pkg.get("metadata", {}).get("model_name") or f.stem
        if name not in latest or f.stat().st_mtime > latest[name].stat().st_mtime:
            latest[name] = f
    return sorted(latest.values(), key=lambda p: p.name)


def load_model(path: Path, data: DataProcessor, device: torch.device):
    """Nạp LightGCN từ package đã lưu, dùng đúng embedding_dim/num_layers lúc train."""
    pkg = torch.load(path, map_location=device)
    cfg = pkg.get("config", {})
    meta = pkg.get("metadata", {})

    model = LightGCNRecommender(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=int(cfg.get("embedding_dim", config.EMBEDDING_DIM)),
        num_layers=int(cfg.get("num_layers", config.NUM_LAYERS)),
    ).to(device)
    model.load_state_dict(pkg["model_state_dict"])
    model.eval()

    info = {
        "model_name": meta.get("model_name", path.stem),
        "lambda_ile": meta.get("lambda_ile", ""),
        "dropout_type": meta.get("dropout_type", ""),
    }
    return model, info


@torch.no_grad()
def evaluate_on_test(model, data: DataProcessor, mask_pos_items, test_items: torch.Tensor,
                     device: torch.device) -> dict[str, float]:
    """Score toàn bộ user rồi chạy full-ranking metrics trên tập test."""
    scores = model.full_sort_scores(data.edge_index_train.to(device)).float()

    # evaluate_full_ranking yêu cầu mỗi user có đúng 1 test item >= 0.
    # Nếu có user không có test item (-1), chỉ eval trên các user hợp lệ.
    valid = test_items >= 0
    if bool(valid.all()):
        s, t, m = scores, test_items.to(device), mask_pos_items
    else:
        idx = valid.nonzero(as_tuple=True)[0]
        s = scores[idx]
        t = test_items[idx].to(device)
        m = [mask_pos_items[i] for i in idx.tolist()]
        print(f"   ⚠️  {int((~valid).sum())} user không có test item -> eval trên {len(idx)} user")

    return evaluate_full_ranking(
        scores=s,
        train_user_pos_items=m,
        test_items=t,
        item_group=data.item_popularity_group.to(device),
        item_degree=data.item_degree.to(device),
        k_list=list(config.K_LIST),
    )


def build_reference_rows() -> pd.DataFrame | None:
    """Lấy các dòng baseline (MostPopular, BPR-MF, LightGCN) từ main_results.csv nếu có."""
    if not REFERENCE_CSV.exists():
        print(f"⚠️  Không thấy file tham chiếu {REFERENCE_CSV}, bỏ qua phần gộp baseline")
        return None
    ref = pd.read_csv(REFERENCE_CSV)
    keep = ["Model"] + [c for c in MAIN_COLS if c in ref.columns]
    ref = ref[keep].copy()
    ref["Source"] = "reference (main_results.csv)"
    return ref


def main() -> bool:
    parser = argparse.ArgumentParser(description="Eval lại tập test với đầy đủ metric.")
    parser.add_argument(
        "--mask", choices=["trainval", "train"], default="trainval",
        help="Item đã-thấy bị mask trước khi xếp hạng. "
             "trainval = train+val (mặc định, khớp protocol test của pipeline); "
             "train = chỉ train (khớp chữ notebook main_results.csv).",
    )
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    args = parser.parse_args()

    set_seed()
    device = get_device()
    print(f"🖥️  Device: {device} | mask = {args.mask} | K_LIST = {config.K_LIST}")

    print("\n📁 Nạp dữ liệu...")
    data = DataProcessor()
    test_items = data.get_test_items_tensor()
    mask_pos_items = data.val_user_pos_items if args.mask == "trainval" else data.train_user_pos_items
    print(f"   Users={data.num_users}, Items={data.num_items}, "
          f"test users có item={int((test_items >= 0).sum())}")

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_files = find_final_models(models_dir)
    print(f"\n🔎 Tìm thấy {len(model_files)} model để eval:")
    for f in model_files:
        print(f"   - {f.name}")

    rows: list[dict] = []
    for f in model_files:
        model, info = load_model(f, data, device)
        print(f"\n📈 Eval: {info['model_name']}")
        metrics = evaluate_on_test(model, data, mask_pos_items, test_items, device)
        row = {
            "Model": info["model_name"],
            "lambda_ile": info["lambda_ile"],
            "dropout_type": info["dropout_type"],
            **{c: float(metrics[c]) for c in MAIN_COLS if c in metrics},
        }
        rows.append(row)
        print(f"   Recall@20={row['Recall@20']:.4f}  NDCG@20={row['NDCG@20']:.4f}  "
              f"TailRecall@20={row['TailRecall@20']:.4f}  Coverage@20={row['Coverage@20']:.4f}  "
              f"ARP@20={row['ARP@20']:.4f}")

        # Giải phóng bộ nhớ giữa các model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) File metric đầy đủ cho các model đã train
    df = pd.DataFrame(rows)
    ordered_cols = ["Model", "lambda_ile", "dropout_type"] + [c for c in MAIN_COLS if c in df.columns]
    df = df[ordered_cols]
    full_path = out_dir / f"full_test_metrics_{timestamp}.csv"
    df.to_csv(full_path, index=False)

    # 2) Bảng so sánh gộp thêm baseline từ main_results.csv
    my_rows = df[["Model"] + [c for c in MAIN_COLS if c in df.columns]].copy()
    my_rows["Source"] = f"recomputed (mask={args.mask})"
    ref = build_reference_rows()
    comparison = pd.concat([ref, my_rows], ignore_index=True) if ref is not None else my_rows
    front = ["Model", "Source"] + [c for c in MAIN_COLS if c in comparison.columns]
    comparison = comparison[[c for c in front if c in comparison.columns]]
    comp_path = out_dir / f"full_comparison_{timestamp}.csv"
    comparison.to_csv(comp_path, index=False)

    # In bảng ra màn hình
    print("\n" + "=" * 120)
    print("📊 BẢNG SO SÁNH (test set, đầy đủ metric)")
    print("=" * 120)
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(comparison.to_string(index=False))

    print(f"\n💾 Metric đầy đủ:     {full_path}")
    print(f"💾 Bảng so sánh:      {comp_path}")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
