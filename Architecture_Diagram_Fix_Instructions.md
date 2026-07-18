# Yêu cầu sửa sơ đồ "Popularity-Aware LightGCN (Our Proposed Model)"

*Tài liệu độc lập, dùng để giao cho người/công cụ vẽ lại sơ đồ 7-panel hiện tại. Mọi yêu cầu sửa dưới đây đã được đối chiếu trực tiếp với code thật của dự án: `src/losses.py`, `src/ile_losses.py`, `src/popaware_training.py` (dòng 96-101 và 244-272), `src/neg_sampling.py`, `src/models.py`.*

---

## Tóm tắt nhanh

Sơ đồ hiện tại đúng phần lớn nội dung (mọi số liệu và hầu hết công thức đã đối chiếu khớp với code thật). Cần sửa:
- **1 lỗi công thức rõ ràng** (mục A)
- **1 lỗi cấu trúc luồng quan trọng** — sai cách vẽ đường lan truyền, không phải chỉ sai 1 mũi tên (mục B — ưu tiên cao nhất cùng với mục A)
- **3 chỗ nên đặt tên/chú thích lại cho chuẩn** (mục C)
- **3 chỗ có thể thêm chú thích nhỏ, không bắt buộc** (mục D)
- **Mục E: số liệu Popularity Groups — đã tự kiểm tra xong, khớp 100%, không còn việc tồn đọng**

---

## A. Sửa lỗi công thức (BẮT BUỘC)

**Vị trí:** Khối ⑤② "Item Loss Equalization (ILE)"

**Hiện tại (sai):**
```
L_ILE = (l̄_item − l̄_tail)²
```

**Sửa thành (đúng):**
```
L_ILE = (l̄_head − l̄_tail)²
```

Giữ nguyên dòng định nghĩa `l̄_g = mean_{i∈g}[-log σ(s_ui+ - s_uj-)]` — chỉ đổi đúng 1 chữ `item` → `head`.

**Vì sao:** `src/ile_losses.py` đặt tên biến `head_loss`, `tail_loss`; công thức đúng là `ile_penalty = (head_loss - tail_loss) ** 2`. Chính docstring trong `src/popaware_training.py` (dòng 165) cũng viết rõ `(head−tail)²`. "item" ghép với "tail" không tạo thành cặp nhóm có nghĩa (3 nhóm popularity là head/middle/tail).

---

## B. Sửa lỗi cấu trúc luồng (BẮT BUỘC — quan trọng nhất, không phải chỉ 1 mũi tên)

**Vấn đề:** Sơ đồ hiện vẽ MỘT chuỗi nối tiếp duy nhất: khối ① → ② → ③ → ④ → ⑤ (đồ thị gốc → augment thành 2 view → LightGCN Backbone "on each view" → Scoring → Training Objectives). Cách vẽ này khiến người xem hiểu rằng **Scoring (và do đó BPR Loss, ILE Loss) cũng lấy embedding từ 2 view đã bị augment (dropout)**.

**Sự thật theo code** (`src/popaware_training.py`, dòng 244-272, xảy ra mỗi bước huấn luyện): có **HAI nhánh lan truyền hoàn toàn tách biệt, chạy song song**:

```
                              ┌── Nhánh A (chính) ─────────────────────────────┐
                              │  Đồ thị GỐC (không dropout)                     │
Khối ① Đồ thị gốc ───────────┤  → 1 lần LightGCN Backbone (propagate)          │
                              │  → embedding gốc                                │
                              │  → Khối ④ Scoring s(u,i)                        │
                              │  → Khối ⑤① BPR Loss + ⑤② ILE Loss              │
                              │  (embedding này CŨNG dùng cho Khối ⑦ Inference) │
                              └──────────────────────────────────────────────────┘

                              ┌── Nhánh B (phụ, CHỈ phục vụ Contrastive Loss) ──┐
                              │  Đồ thị gốc → Khối ② Degree-Aware Augmentation  │
                              │  → tạo 2 view MỚI, lấy mẫu random ĐỘC LẬP        │
                              │    lại từ đầu ở MỖI bước huấn luyện              │
                              │    (không phải 2 view cố định)                   │
                              │  → 2 lần LightGCN Backbone RIÊNG (propagate ×2)  │
                              │  → 2 bộ embedding (view 1, view 2)               │
                              │  → thẳng vào Khối ⑤③ Contrastive Loss (InfoNCE)  │
                              │    KHÔNG đi qua Khối ④ Scoring                   │
                              └──────────────────────────────────────────────────┘
```

**Tóm lại mỗi bước huấn luyện (khi bật Contrastive Learning) có tổng cộng 3 lần lan truyền đồ thị**, không phải 1-2 lần theo một chuỗi như hình đang vẽ:
1. 1 lần trên đồ thị gốc (nhánh A) → nuôi BPR, ILE, và Inference
2. 2 lần trên 2 view augment độc lập (nhánh B) → chỉ nuôi Contrastive Loss

**Yêu cầu vẽ lại cụ thể:**
1. Ngay sau khối ①, tách thành **2 mũi tên rẽ nhánh** (không phải 1 mũi tên nối tiếp vào khối ②).
2. Nhánh A: khối ① → mũi tên thẳng → một khối "LightGCN Backbone (đồ thị gốc, không dropout)" → khối ④ Scoring → khối ⑤①② (BPR, ILE). Vẽ thêm 1 mũi tên phụ từ khối này sang thẳng khối ⑦ Inference, để thể hiện rõ Inference dùng chung embedding với nhánh A.
3. Nhánh B: khối ① → khối ② (Degree-Aware Augmentation, tạo 2 view) → khối ③ (LightGCN Backbone, chạy 2 lần, "on each view") → mũi tên đi thẳng vào khối ⑤③ Contrastive Loss — **không** nối vào khối ④ Scoring.
4. Có thể thêm chú thích nhỏ dưới khối ②/③: "2 view được lấy mẫu dropout độc lập lại ở mỗi bước huấn luyện (không cố định)."

---

## C. Nên sửa (đặt tên/chú thích lại cho chuẩn)

**C1. Đổi tên hệ số L2 trong Total Loss**

Hiện tại: `L = L_BPR + λ_ILE·L_ILE + λ_CL·L_CL + λ_reg·L_reg`
Sửa thành: `L = L_BPR + λ_ILE·L_ILE + λ_CL·L_CL + wd·L_reg`

Vì code gọi hệ số này là `config.WEIGHT_DECAY` (`src/popaware_training.py` dòng 256: `total = loss + config.WEIGHT_DECAY * reg`), không phải "λ_reg". Đổi thành `wd` cho khớp và nhất quán với các tài liệu/slide khác của dự án.

**C2. Thêm chú thích chính xác cho L2 Regularization**

Hiện tại: `L_reg = ‖θ‖²` (dễ hiểu nhầm là chuẩn L2 của TOÀN BỘ tham số mô hình)

Thêm chú thích nhỏ bên dưới: *"θ = embedding gốc (ego, layer-0) của user, positive-item, negative-item trong batch hiện tại; L_reg = (‖u₀‖² + ‖p₀‖² + ‖n₀‖²) / |batch|"*

Vì `src/losses.py: l2_regularization()` tính đúng tổng bình phương chuẩn của 3 tensor `u0, p0, n0` (ego embedding của batch), rồi chia cho batch size — không phải toàn bộ tham số mô hình.

**C3. Đổi chiều mũi tên nét đứt giữa khối ⑤ (Total Loss) và khối ⑥ (Negative Sampling)**

Hiện tại: mũi tên 2 chiều (⇄) giữa Total Loss và Negative Sampling.
Sửa thành: mũi tên **một chiều, từ ⑥ → ⑤** (Negative Sampling luôn diễn ra trước, làm đầu vào cho việc tính điểm số và loss).

---

## D. Có thể thêm (footnote tuỳ chọn, không bắt buộc)

**D1.** Cạnh `L_CL = L_NCE^user + L_NCE^item`, thêm chú thích nhỏ: *"InfoNCE một chiều (view 1 làm query, view 2 làm key), không phải bản đối xứng 2 chiều."* (Đúng theo `info_nce()` trong `popaware_training.py` dòng 96-101: chỉ có 1 lần `cross_entropy`, không cộng thêm chiều ngược lại.)

**D2.** Khung "Key Hyperparameters (example)": tách rõ giá trị nào thuộc lưới sweep thật, giá trị nào chỉ ở baseline:
```
Lưới sweep:  λ_ILE ∈ {0.1, 0.5, 1.0}   λ_CL ∈ {0.1, 0.5}   β ∈ {0, 0.5}
Baseline riêng: λ_ILE = 0, λ_CL = 0
```
(Lưới sweep thật trong `train_sweep_popaware.py` không có giá trị 0 cho λ_ILE/λ_CL; giá trị 0 chỉ xuất hiện ở cấu hình baseline riêng biệt.)

**D3.** Dưới đồ thị ở khối ①, thêm chú thích nhỏ: *"Mũi tên minh hoạ chiều tương tác; cạnh dùng để lan truyền trong huấn luyện (edge_index) là vô hướng/2 chiều."*

---

## E. Số liệu Popularity Groups — ĐÃ XÁC NHẬN, khớp 100%

Đã chạy trực tiếp trên `preprocess_data/item_popularity_group.pt`:

```
Counter({0: 1755, 1: 1071, 2: 707})   # 0=tail, 1=middle, 2=head
total: 3533
```

Khớp chính xác tuyệt đối với khung "Popularity Groups" trên hình (Tail 1.755 / Middle 1.071 / Head 707 / tổng 3.533). **Không cần sửa gì ở khung này.**

---

## Những gì ĐÃ đúng — giữ nguyên, không sửa

- Khối ①: `|U|=6.034`, `|I|=3.533`, `Train=563.204 edges`
- Khối ②: công thức `p_i = p_min + (p_max-p_min)·log(1+deg_i)/log(1+deg_max)`, khoảng `[0.1, 0.4]`
- Khối ③: `E^(k+1) = Ã E^(k)`, `Ã = D^(-1/2) A D^(-1/2)`, layer-wise mean `1/(K+1) Σ E^(k)`
- Khối ④: `s(u,i) = ⟨e_u, e_i⟩`
- Khối ⑤①: `L_BPR = -1/|B| Σ log σ(s_ui+ - s_uj-)`
- Khối ⑥: `P(neg=i) ∝ deg_i^β`, β=0 uniform / β>0 popularity-biased
- Khối ⑦: **"Compute scores using the original (no-dropout) graph"** và **"Mask items seen in train (and val when evaluating test)"** — đây là 2 câu chính xác và quan trọng nhất trên toàn hình, tuyệt đối giữ nguyên nguyên văn.
- `K (recommend) = 20`, `β ∈ {0, 0.5}`

---

## Bảng công thức hoàn chỉnh sau khi sửa (copy trực tiếp)

| Khối | Nội dung sau khi sửa |
|---|---|
| ⑤① BPR Loss | `L_BPR = -1/\|B\| Σ_(u,i,j)∈B log σ(s_ui+ - s_uj-)` |
| ⑤② ILE | `L_ILE = (l̄_head - l̄_tail)²`, `l̄_g = mean_{i∈g}[-log σ(s_ui+ - s_uj-)]` |
| ⑤③ Contrastive Loss | `L_CL = L_NCE^user + L_NCE^item` *(InfoNCE một chiều, view1=query/view2=key)* |
| ⑤④ L2 Regularization | `L_reg = (‖u₀‖² + ‖p₀‖² + ‖n₀‖²) / \|batch\|`, với `u₀,p₀,n₀` = ego embedding batch |
| Total Loss | `L = L_BPR + λ_ILE·L_ILE + λ_CL·L_CL + wd·L_reg` |
| ⑥ Negative Sampling | `P(neg=i) ∝ deg_i^β` (β=0: uniform; β>0: popularity-biased) |
| ② Dropout probability | `p_i = p_min + (p_max-p_min)·log(1+deg_i)/log(1+deg_max)`, `p_i ∈ [0.1, 0.4]` |
| ③ Propagation | `E^(k+1) = Ã E^(k)`, `Ã = D^(-1/2) A D^(-1/2)`, `E = 1/(K+1) Σ_{k=0}^{K} E^(k)` |
| ④ Scoring | `s(u,i) = ⟨e_u, e_i⟩` |
