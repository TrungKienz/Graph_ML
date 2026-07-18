#!/usr/bin/env python3
"""Demo: show real movie-title recommendations for a MovieLens-1M user.

Loads a trained checkpoint (``models/final_model_*.pt``), scores the full
catalog for one user with the same "Branch A" inference path documented in
``EXPLANATION.md`` (Khối 13 — clean graph, no dropout/augmentation), masks
items already interacted with, and prints the top-K recommended movie titles
next to what that user actually watched. This is a qualitative complement to
the purely numeric evaluation in ``evaluate_test_full.py``.

Cách chạy
---------
    python demo_recommend.py                    # user ngẫu nhiên (seed cố định)
    python demo_recommend.py --user 42           # đúng user_idx nội bộ = 42
    python demo_recommend.py --user 42 --k 20    # top-20 thay vì top-10
    ./run_on_gpu.sh demo_recommend.py --user 42  # chạy trên cluster (CPU đủ dùng)

Vì sao phải TỰ dựng lại movie_idx -> title (không dùng torch_geometric)
------------------------------------------------------------------------
``preprocess_data/`` chỉ lưu ``movie_idx`` (re-index từ 0), KHÔNG lưu lại
MovieID gốc của MovieLens-1M. ``notebooks/main.ipynb`` tạo ``movie_idx`` qua
2 bước:
  1) ``torch_geometric.datasets.MovieLens1M`` tự đánh số node "movie" theo
     đúng thứ tự dòng trong ``movies.dat`` (đã kiểm tra trực tiếp trên file
     thật: ``movies.dat``/``users.dat`` đều đã sắp xếp tăng dần theo
     MovieID/UserID gốc, không có dòng nào bị đảo thứ tự).
  2) Notebook lọc ``rating >= 4``, lọc user có ``>= 5`` tương tác, rồi
     re-index lại lần 2 (``np.sort(...unique()...)`` + ``enumerate``) để ra
     đúng ``movie_idx``/``user_idx`` cuối cùng mà toàn bộ pipeline dùng.

Script này đọc thẳng ``movies.dat``/``users.dat``/``ratings.dat`` và lặp lại
đúng 2 bước trên bằng pandas (không cần cài ``torch_geometric``), rồi
**đối chiếu kết quả với ``train_edges.pt``/``val_edges.pt``/``test_edges.pt``
đã cache** — mọi cặp ``(user_idx, movie_idx)`` phải khớp 100%. Nếu không khớp,
script dừng lại và báo lỗi thay vì âm thầm hiển thị sai tên phim.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import get_device, set_seed, PROJECT_ROOT  # noqa: E402
from src.data_loader import DataProcessor  # noqa: E402

# evaluate_test_full.py lives next to this script at the repo root -- reuse its
# checkpoint-loading logic instead of duplicating it.
from evaluate_test_full import find_final_models, load_model  # noqa: E402


RAW_DIR = PROJECT_ROOT / "data" / "MovieLens1M" / "raw"
POS_THRESHOLD = 4.0   # must match notebooks/main.ipynb, cell "Convert Explicit -> Implicit Feedback"
MIN_INTERACTIONS = 5  # must match notebooks/main.ipynb, cell "Remove users with fewer than 5 positive interactions"

GROUP_LABELS = {0: "tail  ", 1: "middle", 2: "head  "}
GROUP_STARS = {0: "🌱", 1: "🌿", 2: "🌳"}  # tail / middle / head, purely cosmetic


# ---------------------------------------------------------------------------
# Step 1: rebuild movie_idx -> title (and rating lookup) from raw .dat files
# ---------------------------------------------------------------------------
def _read_dat_ids(path: Path, n_fields: int) -> list[list[str]]:
    rows = []
    with io.open(path, encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("::", n_fields - 1)
            rows.append(parts)
    return rows


def rebuild_catalog(data: DataProcessor) -> tuple[dict[int, str], dict[tuple[int, int], float]]:
    """Return (movie_idx -> title, (user_idx, movie_idx) -> rating), verified
    against the cached train/val/test edges before being trusted."""
    print("🎬 Rebuilding movie_idx -> title map from raw MovieLens-1M files...")

    movies_rows = _read_dat_ids(RAW_DIR / "movies.dat", 3)
    users_rows = _read_dat_ids(RAW_DIR / "users.dat", 5)
    ratings_rows = _read_dat_ids(RAW_DIR / "ratings.dat", 4)

    movie_pyg_order = [int(r[0]) for r in movies_rows]   # pyg movie-node i -> raw MovieID
    id_to_title = {int(r[0]): r[1] for r in movies_rows}
    user_pyg_order = [int(r[0]) for r in users_rows]     # pyg user-node i -> raw UserID

    movie_raw_to_pyg = {raw: i for i, raw in enumerate(movie_pyg_order)}
    user_raw_to_pyg = {raw: i for i, raw in enumerate(user_pyg_order)}

    df = pd.DataFrame(
        {
            "user": [user_raw_to_pyg[int(r[0])] for r in ratings_rows],
            "movie": [movie_raw_to_pyg[int(r[1])] for r in ratings_rows],
            "rating": [float(r[2]) for r in ratings_rows],
        }
    )

    # Same two filters as notebooks/main.ipynb, in the same order.
    df = df[df["rating"] >= POS_THRESHOLD].copy()
    user_counts = df.groupby("user").size()
    valid_users = user_counts[user_counts >= MIN_INTERACTIONS].index
    df = df[df["user"].isin(valid_users)].copy()

    unique_users = np.sort(df["user"].unique())
    unique_movies = np.sort(df["movie"].unique())

    if len(unique_users) != data.num_users or len(unique_movies) != data.num_items:
        raise RuntimeError(
            f"Reconstruction mismatch: got {len(unique_users)} users / {len(unique_movies)} "
            f"items, expected {data.num_users} / {data.num_items} from preprocess_data/. "
            f"Refusing to show possibly-wrong movie titles -- check that POS_THRESHOLD/"
            f"MIN_INTERACTIONS above still match notebooks/main.ipynb."
        )

    user2idx = {u: i for i, u in enumerate(unique_users)}
    movie2idx = {m: i for i, m in enumerate(unique_movies)}
    df["user_idx"] = df["user"].map(user2idx)
    df["movie_idx"] = df["movie"].map(movie2idx)

    # --- Cross-validate against the cached tensors before trusting anything ---
    reconstructed_pairs = set(zip(df["user_idx"].tolist(), df["movie_idx"].tolist()))
    cached_pairs = set()
    for edges in (data.train_edges, data.val_edges, data.test_edges):
        cached_pairs.update(map(tuple, edges.tolist()))

    if reconstructed_pairs != cached_pairs:
        missing = len(cached_pairs - reconstructed_pairs)
        extra = len(reconstructed_pairs - cached_pairs)
        raise RuntimeError(
            f"Reconstruction does not exactly match cached edges "
            f"({missing} missing, {extra} extra out of {len(cached_pairs)} pairs). "
            f"Refusing to show possibly-wrong movie titles."
        )
    print(f"   ✅ Verified: all {len(cached_pairs)} cached (user,item) pairs match the "
          f"reconstruction exactly -- movie titles below are trustworthy.")

    movie_idx_to_title = {
        final_idx: id_to_title[movie_pyg_order[pyg_idx]]
        for final_idx, pyg_idx in enumerate(unique_movies)
    }
    rating_lookup = {
        (u, m): r for u, m, r in zip(df["user_idx"], df["movie_idx"], df["rating"])
    }
    return movie_idx_to_title, rating_lookup


# ---------------------------------------------------------------------------
# Step 2: pick a model checkpoint (reusing evaluate_test_full.py's loader)
# ---------------------------------------------------------------------------
def pick_model_path(models_dir: Path, model_name: str | None) -> Path:
    candidates = find_final_models(models_dir)
    if model_name is not None:
        matches = [p for p in candidates if model_name in p.name]
        if not matches:
            names = "\n".join(f"  - {p.name}" for p in candidates)
            raise FileNotFoundError(f"No checkpoint matching '{model_name}' in {models_dir}. Available:\n{names}")
        return matches[0]
    if len(candidates) > 1:
        names = "\n".join(f"  - {p.name}" for p in candidates)
        print(f"⚠️  Multiple checkpoints found, using the first one alphabetically. "
              f"Pass --model-name to pick a specific one:\n{names}")
    return candidates[0]


# ---------------------------------------------------------------------------
# Step 3: score + mask + top-K for one user
# ---------------------------------------------------------------------------
@torch.no_grad()
def recommend_for_user(model, data: DataProcessor, user_idx: int, k: int, device):
    """Branch A inference (clean graph, no augmentation) -- see EXPLANATION.md Khối 13."""
    scores = model.full_sort_scores(data.edge_index_train.to(device)).float()
    user_scores = scores[user_idx].clone()

    seen_trainval = data.val_user_pos_items[user_idx]  # train+val, matches evaluate_test_full.py's default mask
    for item in seen_trainval:
        user_scores[item] = float("-inf")

    k = min(k, int((user_scores > float("-inf")).sum().item()))
    topk_scores, topk_items = torch.topk(user_scores, k)
    return topk_items.tolist(), topk_scores.tolist()


def format_movie(item_idx: int, titles: dict, groups: torch.Tensor, extra: str = "") -> str:
    """Just the '<star> [group] Title extra' part, no leading indent."""
    title = titles.get(item_idx, f"<unknown item {item_idx}>")
    group = int(groups[item_idx].item())
    return f"{GROUP_STARS[group]} [{GROUP_LABELS[group]}] {title}{extra}"


def main() -> bool:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", type=int, default=None,
                         help="Internal user_idx to demo (0-based, matches all other scripts). "
                              "Default: a random user that has a held-out test item.")
    parser.add_argument("--k", type=int, default=10, help="Number of recommendations to show (default: 10).")
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--model-name", default=None,
                         help="Substring to pick a specific checkpoint if models/ has more than one "
                              "(e.g. --model-name PopAware-BEST).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random user selection (default: 42).")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"🖥️  Device: {device}")

    print("\n📁 Loading preprocessed data...")
    data = DataProcessor()

    movie_idx_to_title, rating_lookup = rebuild_catalog(data)

    model_path = pick_model_path(Path(args.models_dir), args.model_name)
    model, info = load_model(model_path, data, device)
    print(f"\n📦 Using checkpoint: {model_path.name} (model_name={info['model_name']})")
    if info["model_name"] == "PopAware-BEST":
        print("   ⚠️  Naming caveat (confirmed against results/popaware_sweep_*.csv): this file is "
              "the automatic winner of the hyperparameter SWEEP (layers=3, λ_ILE=0.1, λ_CL=0.5, "
              "β=0.5, single seed). It shares its display name with, but is NOT the same model as, "
              "the hand-picked layers=2/λ_ILE=1.0/λ_CL=0.1/β=0.5 config whose 3-seed mean±std is "
              "reported as 'PopAware-BEST' in README.md/results tables. See README.md 'How to Run' "
              "step 6 for details.")

    test_items = data.get_test_items_tensor()
    if args.user is not None:
        user_idx = args.user
        if not (0 <= user_idx < data.num_users):
            raise ValueError(f"--user must be in [0, {data.num_users - 1}], got {user_idx}")
    else:
        candidates = (test_items >= 0).nonzero(as_tuple=True)[0]
        user_idx = int(candidates[torch.randint(0, len(candidates), (1,))].item())

    print(f"\n{'=' * 78}\n👤 USER user_idx={user_idx}\n{'=' * 78}")

    train_items = sorted(data.train_user_pos_items[user_idx])
    print(f"\n📼 Đã xem (train, {len(train_items)} phim):")
    for item in train_items:
        rating = rating_lookup.get((user_idx, item))
        extra = f"  ({rating:.0f}★)" if rating is not None else ""
        print(f"   {format_movie(item, movie_idx_to_title, data.item_popularity_group, extra)}")

    test_item = int(test_items[user_idx].item())
    if test_item >= 0:
        rating = rating_lookup.get((user_idx, test_item))
        extra = f"  ({rating:.0f}★, giữ lại để đánh giá)" if rating is not None else "  (giữ lại để đánh giá)"
        print(f"\n🎯 Test item (mô hình KHÔNG được thấy lúc suy luận, chỉ dùng để so sánh):")
        print(f"   {format_movie(test_item, movie_idx_to_title, data.item_popularity_group, extra)}")
    else:
        print("\n🎯 User này không có test item (bị loại khi leave-one-out split).")

    topk_items, topk_scores = recommend_for_user(model, data, user_idx, args.k, device)
    print(f"\n🏆 Top-{len(topk_items)} gợi ý (Branch A, đồ thị sạch, không dropout -- xem EXPLANATION.md Khối 13):")
    for rank, (item, score) in enumerate(zip(topk_items, topk_scores), start=1):
        hit = "  ⭐ ĐÚNG TEST ITEM!" if item == test_item else ""
        movie = format_movie(item, movie_idx_to_title, data.item_popularity_group, hit)
        print(f"  #{rank:<2d} score={score:+.4f}  {movie}")

    n_head = sum(1 for i in topk_items if data.item_popularity_group[i].item() == 2)
    n_tail = sum(1 for i in topk_items if data.item_popularity_group[i].item() == 0)
    print(f"\n📊 Trong top-{len(topk_items)}: {n_head} phim head (🌳 phổ biến), {n_tail} phim tail (🌱 ít phổ biến) "
          f"-- so sánh giữa các checkpoint (--model-name) để thấy popularity bias thay đổi thế nào.")

    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
