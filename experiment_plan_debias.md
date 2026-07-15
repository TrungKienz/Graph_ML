# Experiment Plan — Giảm Popularity Bias bằng Causal Debiasing (PD/MACR-style)

**Mục tiêu:** Giảm thiên lệch đề xuất về phía item quá phổ biến trên LightGCN, **không hy sinh quá nhiều accuracy**, và chứng minh phương pháp dịch được Pareto frontier (accuracy ↔ fairness) ra ngoài so với baseline.

Trạng thái: *draft để review — chưa viết code.* Người soạn: Claude (senior-researcher review). Ngày: 2026-07-15.

---

## 0. Mục tiêu đo được & Giả thuyết

**Trục fairness (mục tiêu chính — càng tốt càng giảm bias):**
- `ARP@20` ↓ (average recommendation popularity)
- `HeadExposure@20` ↓ ; `TailExposure@20` + `MiddleExposure@20` ↑
- `Coverage@20` ↑

**Trục accuracy (ràng buộc — không được tụt nhiều):**
- `Recall@20`, `NDCG@20`

**Ràng buộc thành công:** cải thiện fairness **trong khi** giữ `Recall@20` giảm ≤ **5%** (tương đối) so với LightGCN baseline cùng cấu hình.

**Metric phụ (báo cáo nhưng KHÔNG làm mục tiêu chính vì nhiễu cao):** `TailRecall@20` — trên ML-1M số user có test item thuộc tail rất ít nên metric này phương sai lớn; dùng để tham khảo, không để kết luận.

**Giả thuyết:**
- **H1:** Deconfounding độ phổ biến (PD-style) làm giảm ARP@20 và tăng Coverage@20 có ý nghĩa thống kê, với mức giảm Recall ≤ 5%.
- **H2:** PD-LightGCN đẩy frontier (Recall vs Coverage) ra **ngoài** đường của (a) LightGCN thuần và (b) baseline debias rẻ (logit adjustment) — tức tại cùng Recall cho Coverage cao hơn.
- **H3:** Cường độ debias (γ) điều khiển trade-off một cách **đơn điệu và có kiểm soát** (khác hẳn ILE cũ: càng tăng càng sụp cả hai trục).

---

## 1. Điều kiện tiên quyết — sửa lỗi đã phát hiện (làm trước mọi thí nghiệm)

Các thí nghiệm chỉ hợp lệ sau khi vá:

1. **ILE bị lật dấu + không chặn dưới** — [ile_losses.py:125](src/ile_losses.py#L125): `ile_penalty = head_loss - tail_loss` → đổi thành `(head_loss - tail_loss)**2`. Cần để "ILE (fixed)" trở thành baseline so sánh trung thực.
2. **Contrastive loss là dead code** — [ile_losses.py:135](src/ile_losses.py#L135) định nghĩa `contrastive_loss` nhưng không được gọi trong `train_model_with_ile`. Quyết định: hoặc (a) nối vào như một nhánh riêng có cờ, hoặc (b) loại khỏi phạm vi đợt này và ghi rõ "augmentation = chỉ edge dropout". → *Đề xuất: (b) để tập trung vào causal debias; đưa CL vào ablation sau.*
3. **LightGCN over-smoothing** — thêm cấu hình `NUM_LAYERS=2` để so với 3 lớp (số liệu cho thấy 3 lớp thiên lệch hơn).
4. **Đánh giá đúng** — thống nhất mask = train+val, dùng `evaluate_test_full.py` (đã có) làm công cụ đo chuẩn cho MỌI model. Không dùng lại các CSV 4-cột cũ.

---

## 2. Đặc tả phương pháp

### 2.1 Phương pháp chính đề xuất: PD/PDA-LightGCN (Backdoor adjustment)

Dựa trên *Zhang et al., "Causal Intervention for Leveraging Popularity Bias in Recommendation", SIGIR 2021* (PD/PDA). Chọn PD làm lõi vì nó **khớp tự nhiên với pairwise BPR** hiện có (khác MACR dùng pointwise BCE).

**Ký hiệu:** `s(u,i) = <e_u, e_i>` là điểm matching từ embedding LightGCN đã lan truyền. `m_i` = độ phổ biến chuẩn hoá của item (vd `deg_i / max_deg`, hoặc `deg_i` chuẩn hoá về [0,1]). `f(·)` là hàm đảm bảo dương, PD dùng `ELU'(x) = ELU(x)+1` (luôn > 0).

**Nhân quả:** độ phổ biến `Z` là **confounder** — vừa ảnh hưởng khả năng item được hiển thị (exposure) vừa ảnh hưởng click. Model thường học nhầm confounding này thành "chất lượng". PD làm **backdoor adjustment** để tách nó ra.

**Training (deconfounded score):**
```
ŷ_train(u,i) = f(s(u,i)) · (m_i)^γ
```
Huấn luyện pairwise BPR trên `ŷ_train`:
```
L_BPR = − Σ log σ( ŷ_train(u, i⁺) − ŷ_train(u, i⁻) )
```
Trực giác: vì item phổ biến đã được cộng sẵn hệ số `(m_i)^γ` khi train, phần `s(u,i)` **không còn phải mã hoá độ phổ biến** để khớp dữ liệu → `s(u,i)` trở thành điểm "sở thích thuần" đã khử nhiễu phổ biến.

**Inference (do-operator / PDA — bỏ ảnh hưởng phổ biến):**
```
rank theo  f(s(u,i))     # bỏ hẳn thừa số (m_i)^γ
```
Đây chính là bước giảm bias: ở suy luận, ảnh hưởng trực tiếp của độ phổ biến bị loại → ARP ↓, HeadExposure ↓, Coverage ↑.

**Siêu tham số:** `γ ≥ 0` điều khiển mức khử phổ biến. `γ=0` ≡ BPR thường. Quét `γ ∈ {0, 0.02, 0.05, 0.1, 0.2, 0.5}`.

**Điểm tích hợp code (chỉ ghi chú, chưa làm):**
- `config.py`: thêm `DEBIAS_METHOD ∈ {none, pd, macr, ips, logitadj}`, `PD_GAMMA`, `NUM_LAYERS` sweep.
- `models.py`: thêm hàm trả về `s(u,i)` thô (đã có `full_sort_scores`); thêm option áp `f(·)` và `m^γ`.
- `ile_training.py`: chỗ tính `total_loss` — thay score đưa vào BPR bằng `ŷ_train` khi bật PD; giữ nguyên khi tắt.
- Inference: `evaluate_test_full.py` thêm cờ `--debias pd --gamma ...` để rank theo `f(s)`.

### 2.2 Biến thể ablation: MACR-LightGCN

Dựa trên *Wei et al., "Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias", KDD 2021*.
- Thêm 2 nhánh nông: nhánh item `y_i = w_i·e_i` (bắt popularity), nhánh user `y_u = w_u·e_u` (bắt conformity).
- Điểm hợp nhất khi train: `ŷ = s(u,i) · σ(y_i) · σ(y_u)`; loss đa nhiệm `L = L_BPR(ŷ) + α·L(y_i) + β·L(y_u)`.
- Inference counterfactual: trừ ảnh hưởng trực tiếp: `rank theo (s(u,i) − c)·σ(y_i)·σ(y_u)` với `c` là hằng tham chiếu tuning trên val.
- *Ghi chú:* MACR gốc dùng BCE pointwise; cần bản pairwise-adaptation → rủi ro kỹ thuật cao hơn PD. Đưa vào như ablation/đối chứng novelty, không phải lõi.

---

## 3. Baseline & các phương pháp so sánh

| Nhóm | Model | Vai trò |
|---|---|---|
| Không cá nhân hoá | MostPopular | "trần bias" (HeadExp=1.0) |
| Baseline accuracy | BPR-MF | baseline mạnh phải vượt (đang thắng!) |
| Baseline đồ thị | LightGCN (3 lớp), LightGCN (2 lớp) | ảnh hưởng độ sâu |
| Debias rẻ (đối chứng) | **Logit Adjustment** (`s − α·log(1+deg)` lúc infer) | must-have; nếu PD không hơn cái này thì PD vô nghĩa |
| Debias reweight | IPS-BPR (positive ×(1/deg)^γ), Popularity-aware negatives (neg ∝ deg^β) | so sánh họ reweighting |
| Method cũ (đã vá) | ILE (equalization, `(h−t)²`) | chứng minh method cũ giờ ít nhất không hại |
| **Đề xuất chính** | **PD-LightGCN** (γ sweep) | đóng góp |
| Ablation | MACR-LightGCN | novelty phụ |

Tất cả dùng **cùng** embedding_dim=64, epochs, LR, seed, protocol eval. Chỉ biến thiên đúng thành phần đang khảo sát.

---

## 4. Ma trận thí nghiệm

**E0 — Chuẩn hoá lại nền (sau khi vá):** MostPopular, BPR-MF, LightGCN {2,3 lớp} → đo đủ 10 metric bằng `evaluate_test_full.py`. Xác lập baseline & frontier gốc.

**E1 — Baseline debias rẻ:** Logit Adjustment quét `α ∈ {0.1,…,2.0}`; Pop-aware negatives quét `β ∈ {0,0.5,1.0}`; IPS-BPR quét `γ_ips ∈ {0.1,0.2,0.5}`. → vẽ frontier của các cách rẻ.

**E2 — ILE (fixed):** quét `λ_ILE ∈ {0,0.1,0.5,1.0,2.0,5.0}` với công thức bình phương. Kỳ vọng: giờ đơn điệu, không sụp về MostPopular. (Kiểm chứng bug đã hết.)

**E3 — PD-LightGCN (chính):** quét `γ ∈ {0,0.02,0.05,0.1,0.2,0.5}` × `NUM_LAYERS ∈ {2,3}`. Đây là bảng kết quả chính.

**E4 — Ablation PD:**
- PD-train + rank theo `f(s)` (đầy đủ) **vs** chỉ PD-train nhưng rank theo `ŷ_train` (không bỏ m^γ) → tách đóng góp của bước inference.
- PD trên MF **vs** PD trên LightGCN → tính model-agnostic.
- (tuỳ chọn) PD + edge-dropout / + contrastive.

**E5 — MACR-LightGCN:** đối chứng novelty, quét `α,β` và `c`.

**E6 — (tuỳ chọn) Generalization:** lặp E3 trên 1 dataset thứ hai (Gowalla hoặc Amazon-Book) để chống phản biện "chỉ đúng trên ML-1M". *Cần chuẩn bị data — có thể để phase sau.*

---

## 5. Thiết kế đảm bảo công bằng & thống kê

- **Đa seed:** mỗi cấu hình chạy **3–5 seed**, báo cáo `mean ± std`. So sánh chính kèm kiểm định (paired t-test hoặc Wilcoxon) trên các seed.
- **Tuning trên VAL, báo cáo trên TEST:** chọn siêu tham số debias (α, β, γ) theo **quy tắc frontier trên val**: *tối đa Coverage@20 với ràng buộc Recall@20 giảm ≤ 5%*. Không được peek test.
- **Biến kiểm soát cố định:** kiến trúc, epochs, optimizer, batch, negative-sampling (trừ khi nó là biến), mask eval, danh sách seed.
- **Điểm dừng:** early stopping theo Recall@20 trên val giống nhau cho mọi method.
- **Đo lường:** một công cụ duy nhất (`evaluate_test_full.py`) cho mọi model để loại sai khác do code eval.

---

## 6. Tiêu chí thành công (định lượng, chốt trước khi chạy)

- **Chính (H1/H2):** tồn tại điểm PD với `ARP@20` giảm ≥ **0.3** tuyệt đối **và** `Coverage@20` tăng ≥ **20%** tương đối so với LightGCN cùng cấu hình, trong khi `Recall@20` giảm ≤ **5%**.
- **So với debias rẻ (H2):** ở cùng mức `Recall@20`, PD cho `Coverage@20` cao hơn Logit-Adjustment một cách có ý nghĩa (không chồng khoảng std). Nếu KHÔNG hơn → trung thực kết luận PD không thêm giá trị, chuyển hướng.
- **Đơn điệu (H3):** tăng γ → ARP giảm & Coverage tăng đơn điệu (spearman corr có ý nghĩa), không sụp accuracy kiểu ILE cũ.

---

## 7. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| PD với γ lớn làm tụt accuracy mạnh | quét γ nhỏ trước (0.02–0.1); dùng quy tắc frontier trên val |
| `m_i^γ` gây bất ổn số học (deg lệch nặng) | chuẩn hoá `m_i∈[0,1]`, dùng `ELU'` dương; log-clamp |
| TailRecall nhiễu dẫn tới kết luận sai | không dùng làm mục tiêu chính; báo cáo Coverage/ARP/Exposure |
| LightGCN 3 lớp che hiệu ứng | luôn chạy song song 2 lớp |
| PD chỉ "chuyển" bias sang head→middle chứ không tới tail | báo cáo cả 3 exposure, không chỉ Head |
| Overclaim so với baseline rẻ | bắt buộc có Logit-Adjustment trong mọi biểu đồ frontier |

---

## 8. Compute & timeline (ước lượng thô, theo log cũ ~2s/epoch, ~4.5 phút/model)

- E0–E2: ~15 model × ~5 phút × ~1 (×seeds sau) ≈ 1–1.5h/seed.
- E3: 6 γ × 2 lớp = 12 model ≈ 1h/seed.
- ×3 seed cho các bảng chính ≈ nửa ngày GPU. Khả thi trên A100 qua Slurm.

---

## 9. Deliverables đợt code (sau khi plan được duyệt)

1. Vá lỗi tiên quyết (mục 1) + cờ cấu hình debias trong `config.py`.
2. Cài PD-LightGCN (train + inference) dạng bật/tắt.
3. Baseline debias: Logit-Adjustment, IPS-BPR, pop-aware negatives (cờ).
4. Mở rộng `evaluate_test_full.py`: đa seed, chọn theo frontier trên val, xuất bảng mean±std.
5. Script vẽ Pareto frontier (Recall@20 ↔ Coverage@20 / ARP@20) cho mọi method.
6. (tuỳ chọn) MACR ablation.

---

## 10. Câu hỏi mở cần bạn quyết trước khi code

1. **Chuẩn hoá độ phổ biến `m_i`:** dùng `deg/max_deg`, hay `deg/sum`, hay rank-percentile? (đề xuất: `deg/max_deg`).
2. **Contrastive learning:** loại khỏi đợt này (đề xuất) hay vẫn nối vào làm ablation E4?
3. **Dataset thứ hai (E6):** làm luôn để chống phản biện generalization, hay để phase sau? (ảnh hưởng khối lượng nhiều).
4. **Số seed:** 3 (nhanh) hay 5 (chắc hơn)?
5. **Giữ ILE là đóng góp song song** với PD, hay coi ILE chỉ là baseline và PD là đóng góp duy nhất?

---

## 11. Quyết định đã chốt (khuyến nghị mặc định — 2026-07-15)

| # | Câu hỏi | Quyết định | Lý do |
|---|---|---|---|
| 1 | Chuẩn hoá `m_i` | **`deg_i / max_deg`** (bounded [0,1]) | Đơn giản, ổn định số học, đúng chuẩn PD. Ghi lại `deg` từ **train** để tránh leak. |
| 2 | Contrastive learning | **Bỏ khỏi đợt này**, để làm ablation E4 sau | Tập trung 1 lõi (causal). CL đang là dead-code, nối vào lúc này làm loãng và tăng biến. |
| 3 | Dataset thứ 2 (E6) | **Hoãn sang phase 2** | Ưu tiên dựng xong method + frontier trên ML-1M trước; thêm Gowalla để chống phản biện generalization sau khi có tín hiệu tốt. |
| 4 | Số seed | **3 seed cho sweep, 5 seed cho bảng headline** (PD vs baselines) | Cân bằng tốc độ lặp và độ tin cậy của kết luận chính. |
| 5 | Vai trò ILE | **PD là đóng góp chính duy nhất; ILE (đã vá) là baseline/ablation** | Một đóng góp mạnh, sạch hơn hai đóng góp nửa vời. Vẫn báo cáo ILE-fixed để cho thấy đã chẩn đoán & sửa; nếu PD+ILE cộng hưởng thì trình bày ILE như thành phần bổ trợ. |

**Thứ tự thực thi khi được duyệt code:** (1) vá tiên quyết → E0 baseline sạch → (2) PD-LightGCN + Logit-Adjustment baseline → E3 bảng chính → (3) frontier plot → (4) ILE-fixed E2 + ablation E4 → (5) MACR E5 nếu còn thời gian.
