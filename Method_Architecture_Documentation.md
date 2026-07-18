# Popularity-Aware LightGCN — Tài liệu Kiến trúc & Thiết kế Kỹ thuật

*Mục đích: giúp bất kỳ ai (kể cả người chưa từng đọc code) hiểu nhanh phương pháp đề xuất và tra cứu đúng file/hàm khi cần đọc/sửa code. Mọi mô tả trong tài liệu này đã được đối chiếu trực tiếp với code thật (`src/`), không suy đoán.*

---

## 1. Tổng quan kiến trúc (Big Picture)

**Bài toán:** Gợi ý phim trên MovieLens-1M, mục tiêu kép: (1) độ chính xác top-K cao, (2) giảm popularity bias (item phổ biến lấn át item long-tail).

**Ý tưởng cốt lõi:** Kiến trúc gồm **2 nhánh lan truyền song song**, chạy đồng thời mỗi bước huấn luyện:

```
                    ┌── NHÁNH A (chính) ──────────────────────────────┐
                    │  Đồ thị GỐC (không dropout)                      │
Đồ thị gốc ─────────┤  → LightGCN Backbone (1 lần propagate)           │
                    │  → embedding gốc → Scoring s(u,i)                │
                    │  → BPR Loss + ILE Loss                           │
                    │  (embedding này CŨNG dùng cho Inference)         │
                    └───────────────────────────────────────────────────┘

                    ┌── NHÁNH B (phụ, chỉ phục vụ Contrastive Loss) ──┐
                    │  Đồ thị gốc → Degree-Aware Augmentation          │
                    │  → 2 view dropout ĐỘC LẬP, lấy mẫu lại mỗi bước  │
                    │  → LightGCN Backbone (2 lần propagate riêng)     │
                    │  → 2 bộ embedding (view 1, view 2)               │
                    │  → thẳng vào Contrastive Loss (không qua Scoring)│
                    └───────────────────────────────────────────────────┘
```

**4 thành phần đề xuất** (bật/tắt độc lập qua cờ, phục vụ ablation):

| # | Tên | Tác động vào đâu | Cờ bật/tắt |
|---|---|---|---|
| 1 | Item Loss Equalization (ILE) | Hàm mục tiêu (loss) | `use_ile` |
| 2 | Degree-Aware Graph Augmentation | Tạo 2 view cho Nhánh B | `aug_main` (áp lên đồ thị chính, mặc định **False**), dropout luôn bật ngầm khi `use_cl=True` |
| 3 | Contrastive Learning (InfoNCE) | Không gian embedding | `use_cl` |
| 4 | Popularity-aware Negative Sampling | Tín hiệu học đối nghịch (negative) | `neg_pop_beta` (β) |

**Điểm quan trọng nhất cần nhớ:** `aug_main=False` trong **mọi cấu hình đã báo cáo kết quả**. Nghĩa là Degree-Aware Augmentation **không bao giờ** làm thay đổi đồ thị dùng để tính điểm số/suy luận thật — nó chỉ tồn tại để tạo 2 view phụ trợ cho Contrastive Loss.

---

## 2. Cấu trúc thư mục & vai trò từng file

```text
TestSSH/
├── src/
│   ├── config.py              # hằng số & siêu tham số duy nhất (single source of truth)
│   ├── data.py                 # tiền xử lý gốc: lọc rating≥4, split_per_user() (leave-one-out theo thời gian)
│   ├── data_loader.py           # DataProcessor: load tensor cache từ preprocess_data/
│   ├── metrics.py               # 10 metric + evaluate_full_ranking()
│   ├── baselines.py             # MostPopular
│   ├── models.py                # BPRMF, LightGCNRecommender (propagate, scoring)
│   ├── losses.py                 # bpr_loss, l2_regularization
│   ├── train.py                  # vòng lặp train BPR-MF/LightGCN gốc (notebook path)
│   ├── ile_losses.py              # ile_loss(), compute_degree_aware_dropout_probs()
│   ├── neg_sampling.py            # build_neg_probs(), sample_bpr_batch_popaware()
│   └── popaware_training.py       # ★ FILE TRUNG TÂM: train_popaware_lightgcn()
│
├── preprocess_data/              # tensor cache (train/val/test edges, item_degree, item_popularity_group)
├── train_all_popaware.py          # ablation tách từng thành phần (1 seed)
├── train_sweep_popaware.py         # quét 24 điểm hyperparameter + chọn operating point
├── train_final_seeds.py             # 5 cấu hình chốt × 3 seed → số liệu báo cáo chính thức
├── evaluate_test_full.py             # nạp lại model đã lưu, tính đủ 10 metric (không train lại)
└── run_on_gpu.sh                      # gửi job lên cụm GPU qua Slurm
```

*(Bảng đầy đủ hơn — gồm cả `results/`, `checkpoints/`, `logs/`, và danh sách file KHÔNG cần quan tâm (script debug tạm) — xem `Slide_Review_and_Defense_QA.md` mục 7.)*

---

## 3. Luồng dữ liệu chi tiết — Step by Step

### Bước 0 — Tiền xử lý (1 lần, trước khi train)
`src/data.py: split_per_user()` — lọc rating≥4, sắp xếp tương tác mỗi user theo thời gian, cắt tương tác gần nhất làm test, kế đó làm val, còn lại làm train. Tính `item_degree` (số lượt xuất hiện trong train) và `item_popularity_group` (tail=0/middle=1/head=2, theo percentile P50/P80). Lưu ra `preprocess_data/*.pt`.

### Bước 1 — Load dữ liệu
`src/data_loader.py: DataProcessor` — đọc lại tensor cache, dựng `train_user_pos_items` (mỗi user → tập item đã tương tác trong train, dùng để loại trừ khi lấy negative).

### Bước 2 — Khởi tạo Embedding
`src/models.py: LightGCNRecommender.__init__` — MỘT bảng embedding duy nhất `nn.Embedding(num_users+num_items, 64)` cho cả user và item (item thứ *i* có index `num_users+i`), khởi tạo ngẫu nhiên `std=0.1`.

### Bước 3 — Lan truyền (Nhánh A, mỗi bước)
`src/models.py: propagate()` — `E^(k+1)=Ã·E^(k)`, `Ã=D^(-1/2)AD^(-1/2)`, lặp K lớp, lấy trung bình cộng các lớp `E=1/(K+1)Σ E^(k)`. Dùng đồ thị `main_edge = edge_index` (gốc, vì `aug_main=False`).

### Bước 4 — Tính điểm số
`src/models.py: bpr_forward()` → `s(u,i)=⟨e_u,e_i⟩` (dot product).

### Bước 5 — Lấy mẫu batch
`src/neg_sampling.py: sample_bpr_batch_popaware()` — positive lấy đều từ train edges; negative lấy đều (β=0) hoặc thiên vị `∝deg^β` (β>0), có rejection-sampling (tối đa 20 lần) tránh trùng positive.

### Bước 6 — Tính Loss (4 thành phần, `src/popaware_training.py` dòng 244-272)
- `L_BPR` (`src/losses.py: bpr_loss`) — luôn tính.
- `L_ILE` (`src/ile_losses.py: ile_loss`) — chỉ tính khi `use_ile=True`.
- `L_CL` (`info_nce()` trong `popaware_training.py`) — chỉ tính khi `use_cl=True`, **dùng embedding riêng từ Nhánh B**, không qua bước 4.
- `L_reg` (`src/losses.py: l2_regularization`) — luôn tính, trên ego-embedding batch.

### Bước 7 — Cập nhật tham số
`optimizer.step()` (Adam, lr=1e-3) sau `loss.backward()` và `clip_grad_norm_(max_norm=1.0)`.

### Bước 8 — Lặp qua epoch, kiểm tra định kỳ
Mỗi `eval_every=5` epoch, gọi `_evaluate()` trên VAL (Recall@K).

### Bước 9 — Early stopping + checkpoint
Cải thiện → lưu `_best.pt`; không cải thiện `patience=10` lần liên tiếp → dừng sớm. Luôn lưu `_latest.pt` để resume.

### Bước 10 — Đánh giá TEST (1 lần duy nhất)
Nạp lại `_best.pt`, gọi `evaluate_full_ranking()` (`src/metrics.py`) → đủ 10 metric.

---

## 4. Chi tiết kiến trúc mô hình

### 4.1 Embedding
- Một bảng `nn.Embedding` chung cho user+item, `embedding_dim=64` (`config.EMBEDDING_DIM`).
- Không có feature transform, không có non-linearity (đúng thiết kế LightGCN gốc — He et al. 2020: các thành phần này không đóng góp cho collaborative filtering, thậm chí gây overfitting).

### 4.2 LightGCN Backbone (dùng chung cho cả 2 nhánh, cùng 1 hàm `propagate()`)
```
Ã = D^(-1/2) A D^(-1/2)                     # chuẩn hoá đối xứng
E^(k+1) = Ã · E^(k)                          # lan truyền tuyến tính, không activation
E = 1/(K+1) · Σ_{k=0}^{K} E^(k)              # layer-wise mean pooling
```
`num_layers` mặc định K=2 cho các cấu hình chốt (sweep có thử K=3).

### 4.3 Degree-Aware Graph Augmentation (chỉ Nhánh B)
```
p_drop(i) = p_min + (p_max - p_min) · log(1+deg_i) / log(1+deg_max)     # p ∈ [0.1, 0.4]
```
`src/ile_losses.py: compute_degree_aware_dropout_probs()`. Áp dụng **đối xứng**: quyết định giữ/drop chỉ thực hiện 1 lần trên mỗi cạnh vô hướng, rồi mirror sang cả 2 chiều (`symmetric_edge_dropout()` trong `popaware_training.py`) — giữ tính chất đồ thị vô hướng cho phép chuẩn hoá `D^(-1/2)AD^(-1/2)` hợp lệ. Hai view (v1, v2) được lấy mẫu **độc lập, lại từ đầu mỗi bước huấn luyện** — không cố định.

### 4.4 Scoring
`s(u,i) = ⟨e_u, e_i⟩` — chỉ dùng embedding từ Nhánh A (đồ thị gốc).

---

## 5. Hàm mục tiêu đầy đủ

| Thành phần | Công thức | File / hàm |
|---|---|---|
| BPR Loss | `L_BPR = -1/|B| Σ log σ(s⁺_ui - s⁻_uj)` | `src/losses.py: bpr_loss()` |
| ILE | `L_ILE = (l̄_head - l̄_tail)²`, `l̄_g = mean_{i∈g}[-log σ(s⁺_ui-s⁻_uj)]` | `src/ile_losses.py: ile_loss()` |
| Contrastive Loss | `L_CL = L_NCE^user + L_NCE^item` (InfoNCE **một chiều**: view1=query, view2=key) | `info_nce()` trong `popaware_training.py` (dòng 96-101, 269-270) |
| L2 Regularization | `L_reg = (‖u₀‖²+‖p₀‖²+‖n₀‖²)/|batch|` (u₀,p₀,n₀ = ego embedding batch, KHÔNG phải toàn bộ tham số) | `src/losses.py: l2_regularization()` |
| **Total Loss** | `L = L_BPR + λ_ILE·L_ILE + λ_CL·L_CL + wd·L_reg` | dòng 254-272 |

**Lưu ý quan trọng:** `L_CL` không đi qua bước Scoring (mục 4.4) — nó dùng embedding thô, chuẩn hoá rồi nhân ma trận trực tiếp (`F.normalize` + `z1@z2.T`), không phải dot-product `s(u,i)`.

---

## 6. Popularity-Aware Negative Sampling

```
P(neg=i) ∝ deg_i^β        # β=0: uniform (mặc định/nguyên bản) | β>0: thiên vị item phổ biến
```
`src/neg_sampling.py: build_neg_probs()`, `sample_bpr_batch_popaware()`. Rejection-sampling tối đa 20 lần nếu trùng positive của user (xác suất trùng cực thấp với catalog ~3.500 item, không đảm bảo tuyệt đối 100% nhưng không đáng kể).

---

## 7. Hạ tầng huấn luyện (`train_popaware_lightgcn`, `src/popaware_training.py`)

| Cơ chế | Chi tiết |
|---|---|
| Model selection | VAL Recall@K, mỗi `eval_every=5` epoch |
| Early stopping | `patience=10` lần đánh giá không cải thiện |
| Checkpoint | `<run_id>_best.pt` (theo VAL) + `<run_id>_latest.pt` (resume) |
| Resume | Tự động nếu `_latest.pt` tồn tại và `resume=True` |
| Logging | stdout + file `logs/popaware/<run_id>.log`, mỗi epoch |
| History | CSV `results/popaware/history_<run_id>.csv` mỗi epoch (vẽ loss curve) |
| Gradient clipping | `clip_grad_norm_(max_norm=1.0)` trước mỗi `optimizer.step()` |
| Seed | `torch.manual_seed(seed)` + `gen=torch.Generator()` riêng cho sampling (để resume đúng trạng thái random) |
| Optimizer | Adam, `lr=1e-3` (`config.LR`) |
| Batch size | 4096 (`config.BATCH_SIZE`) |

**Chống rò rỉ dữ liệu (leakage-free):**
- VAL: `target=val_items`, `mask=train_user_pos_items` (không đụng test).
- TEST: `target=test_items`, `mask=train+val` (đúng comment trong code, dòng 199-201), chấm **đúng 1 lần** ở cuối, sau khi nạp lại `_best.pt`.

---

## 8. Metrics (đối chiếu `src/metrics.py`)

| Metric | Công thức / định nghĩa | Ghi chú |
|---|---|---|
| `Recall@K` | Tỉ lệ user có test item nằm trong top-K (= Hit Rate@K vì mỗi user chỉ có đúng 1 test item) | `recall_at_k()` |
| `NDCG@K` | `1/log2(rank+1)` nếu trúng, 0 nếu không | `ndcg_at_k()` |
| `TailRecall@K` | Recall@K chỉ tính trên user có test item thuộc nhóm **tail** | `tail_recall_at_k()` |
| `Coverage@K` | Tỉ lệ item xuất hiện trong **ít nhất 1** top-K của **bất kỳ** user nào / tổng catalog (không phải trung bình per-user) | `catalog_coverage_at_k()` |
| `ARP@K` | Trung bình `log(1+degree)` của các item được gợi ý (thang **log**, không phải degree thô) | `average_recommendation_popularity()` |
| `TailExposure/MiddleExposure/HeadExposure@K` | Tỉ lệ slot top-K (trên toàn hệ thống) thuộc mỗi nhóm popularity, tổng = 1 | `exposure_by_group()` |

Nhóm popularity: **tail <P50, middle P50-P80, head >P80** theo `item_degree` (train). Trên MovieLens-1M: **Tail=1.755, Middle=1.071, Head=707 item** (tổng 3.533 — đã xác nhận trực tiếp từ `item_popularity_group.pt`).

---

## 9. Bảng hyperparameter đầy đủ

| Tham số | Giá trị | Nơi định nghĩa |
|---|---|---|
| `EMBEDDING_DIM` | 64 | `config.py` |
| `NUM_LAYERS` (K) | 2 (chính), 3 (thử trong sweep) | `config.py` / CLI arg |
| `LR` | 1e-3 | `config.py` |
| `BATCH_SIZE` | 4096 | `config.py` |
| `NUM_EPOCHS` | 100 (tối đa, có early stop) | `config.py` |
| `WEIGHT_DECAY` (wd) | 1e-4, cố định, không sweep | `config.py` |
| `TAU` (τ, nhiệt độ InfoNCE) | 0.2, cố định, không sweep | `config.py` |
| `DROPOUT_P_MIN / DROPOUT_P_MAX` | 0.1 / 0.4 | `config.py` |
| `λ_ILE` | Lưới sweep: {0.1, 0.5, 1.0}; baseline riêng = 0 | `train_sweep_popaware.py` |
| `λ_CL` | Lưới sweep: {0.1, 0.5}; baseline riêng = 0 | `train_sweep_popaware.py` |
| `β` (neg_pop_beta) | {0, 0.5} | `train_sweep_popaware.py` |
| `eval_every` | 5 | `popaware_training.py` |
| `patience` | 10 | `popaware_training.py` |
| Percentile nhóm popularity | P50 / P80, cố định, không sweep | `config.py` |

---

## 10. Năm cấu hình chốt (báo cáo kết quả chính thức, `train_final_seeds.py`)

| Cấu hình | λ_ILE | λ_CL | β | Ghi chú |
|---|---|---|---|---|
| LightGCN (baseline) | 0 | 0 | 0 | Backbone thuần |
| PopAware-accuracy | 0.1 | 0.1 | 0.5 | Ưu tiên accuracy |
| **PopAware-BEST** | 1.0 | 0.1 | 0.5 | Điểm cân bằng, khuyến nghị chính |
| PopAware-high-tail | 1.0 | 0.1 | 0 | TailRecall cao nhất |
| PopAware-fairness | 1.0 | 0.5 | 0 | Debias mạnh nhất, đánh đổi accuracy nhiều nhất |

Tất cả chạy 3 seed `{42, 0, 1}`, `num_layers=2`. Quy tắc chọn: max Coverage@20, ràng buộc Recall@20 không giảm quá ngưỡng so với baseline (hiện tính trên số liệu **TEST** của sweep — xem mục 11).

---

## 11. Giới hạn & lưu ý trung thực (đọc trước khi tuyên bố bất cứ điều gì)

1. **Quy tắc chọn operating point tính trên TEST**, không phải VAL — dạng dò tập test nhẹ qua nhiều lần thử, không phải zero-leakage tuyệt đối ở cấp *chọn cấu hình* (khác với cấp *chọn checkpoint*, cấp đó đúng là leakage-free).
2. **Degree-Aware Augmentation chỉ tác động Nhánh B** trong mọi kết quả báo cáo — chưa có kiểm chứng 3-seed cho việc áp dụng nó lên đồ thị chính (`aug_main=True`, có tồn tại như ablation 1-seed riêng trong `train_all_popaware.py`).
3. **InfoNCE một chiều**, không phải bản đối xứng 2 chiều.
4. **β không đơn điệu** trên toàn lưới sweep (khác λ_ILE và λ_CL, cả hai đều đơn điệu rõ ràng) — bằng chứng chắc chắn nhất về β là so sánh 3-seed BEST vs high-tail.
5. **Ablation tách riêng từng thành phần** (`train_all_popaware.py`) mới chạy 1 seed, chưa có độ tin cậy 3-seed.
6. **Chưa so sánh với baseline debiasing khác trong literature** (IPS, DICE, MACR, PD) — chỉ so với LightGCN/BPR-MF/MostPopular thuần.
7. Ngưỡng percentile (P50/P80), τ, weight_decay: **cố định, chưa sweep độ nhạy**.

*(Bộ câu hỏi phản biện đầy đủ liên quan đến từng điểm trên: xem `Slide_Review_and_Defense_QA.md` mục 4.)*

---

## 12. Lịch sử sửa lỗi quan trọng (để hiểu code hiện tại vì sao viết như vậy)

| Lỗi | Nguyên nhân | Đã sửa thành |
|---|---|---|
| ILE làm bias tệ hơn thay vì giảm | `ile_penalty = head_loss - tail_loss` (không bình phương, lật dấu, không chặn dưới) | `(head_loss - tail_loss) ** 2` — luôn ≥0, không lật dấu |
| Rò rỉ dữ liệu khi chọn model | `_evaluate` từng "nhìn" tập TEST trong lúc train | Tách rõ: VAL để chọn checkpoint, TEST chấm đúng 1 lần cuối cùng |
| Crash khi load cache trên PyTorch≥2.6 | `torch.load()` mặc định `weights_only=True` từ 2.6, không đọc được file cache có numpy scalar | Thêm `weights_only=False` (an toàn vì file do chính project sinh ra) |
| Crash `glob("**")` trên Python cluster (3.11) | Cú pháp glob khác phiên bản | Đổi sang `iterdir()` |

---

## 13. Tra cứu nhanh — "Muốn hiểu X → đọc file/hàm nào"

| Muốn hiểu | Đọc |
|---|---|
| Toàn bộ vòng lặp huấn luyện | `src/popaware_training.py: train_popaware_lightgcn()` |
| Công thức LightGCN | `src/models.py: propagate()`, `_normalized_adjacency()` |
| Công thức ILE (đã sửa lỗi) | `src/ile_losses.py: ile_loss()` |
| Dropout theo degree | `src/ile_losses.py: compute_degree_aware_dropout_probs()` + `symmetric_edge_dropout()` trong `popaware_training.py` |
| InfoNCE / Contrastive Learning đang dùng thật | `info_nce()` trong `popaware_training.py` (KHÔNG phải `contrastive_loss()` trong `ile_losses.py` — hàm đó là bản cũ, không dùng) |
| Negative sampling theo popularity | `src/neg_sampling.py` |
| Định nghĩa 10 metric | `src/metrics.py` |
| Split train/val/test | `src/data.py: split_per_user()` |
| Mọi siêu tham số | `src/config.py` |
| Số liệu 3-seed cuối cùng (Slide 11) | `results/popaware_final_meanstd_20260715_174317.csv` |
| Lưới sweep 24 điểm | `results/popaware_sweep_20260715_155058.csv` |

---

## Tài liệu liên quan trong repo

- `README.md` — tổng quan project, cách cài đặt/chạy.
- `PopAware_LightGCN_Documentation.md` — báo cáo kỹ thuật đầy đủ dùng cho báo cáo/luận văn.
- `report_method_section.tex` — phần phương pháp viết bằng LaTeX (tiếng Anh).
- `Slide_Review_and_Defense_QA.md` — review slide, 44 câu hỏi phản biện, script thuyết trình, giải thích step-by-step.
- `Architecture_Diagram_Fix_Instructions.md` — lịch sử sửa lỗi sơ đồ kiến trúc (đã hoàn tất).
