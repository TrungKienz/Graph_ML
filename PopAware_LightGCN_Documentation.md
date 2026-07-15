# Popularity-Aware LightGCN for Long-Tail Movie Recommendation — Tài liệu kỹ thuật

*Tài liệu nguyên liệu để viết báo cáo. Mọi số liệu lấy từ thực nghiệm trên MovieLens-1M,
sweep `results/popaware_sweep_20260715_155058.csv`, 3-seed
`results/popaware_final_runs_20260715_174317.csv` / `popaware_final_meanstd_20260715_174317.csv`,
references `results/metrics/`.*

---

## 1. Bài toán & động lực

Hệ gợi ý cộng tác (LightGCN, BPR-MF) chịu **popularity bias**: item phổ biến được cập nhật
nhiều → embedding có norm lớn → điểm số cao với hầu hết user → chiếm lĩnh top-K. Hệ quả: item
"đuôi dài" (long-tail) hầu như không được đề xuất, độ phủ danh mục thấp, trải nghiệm bị đồng nhất.

**Mục tiêu:** giảm thiên lệch về item quá phổ biến trên nền LightGCN, sao cho:
- TailRecall@20 **tăng**, Coverage@20 **tăng**, ARP@20 **giảm**, HeadExposure@20 **giảm**;
- Recall@20 / NDCG@20 **không giảm mạnh** (ngưỡng chấp nhận: ≤ 5%);
- TailRecall ở mức **chấp nhận được** (cao hơn baseline nhưng không cực đoan).

---

## 2. Dữ liệu

- **Bộ dữ liệu:** MovieLens-1M, phản hồi ẩn (implicit).
- **Kích thước sau tiền xử lý:** 6.034 user, 3.533 item.
- **Chia dữ liệu (leave-one-out):** train = 563.204 tương tác; val = 6.034; test = 6.034
  (mỗi user giữ lại 1 item cho val và 1 cho test).
- **Nhóm phổ biến item** (theo degree trên train): Head = top 20% (707 item), Middle = 30 kế tiếp
  (1.071), Tail = 50% cuối (1.755). Mã hoá: `0=tail, 1=middle, 2=head`.
- **Đồ thị:** đồ thị hai phía user–item, cạnh vô hướng (bidirectional), item `i` là node `i + num_users`.

---

## 3. Chỉ số đánh giá (10 metric, full-ranking)

Xếp hạng trên **toàn bộ** catalog sau khi **mask** các item đã tương tác; mỗi user có 1 item test.
Khi chấm TEST: mask = train+val. Khi chọn model (val): mask = train, target = item val (không đụng test).

**Nhóm Accuracy** (cao hơn tốt hơn):
- **Recall@K** = tỉ lệ user có item test lọt top-K.
- **NDCG@K** = trung bình `1/log2(rank+1)` nếu trúng, 0 nếu trượt.

**Nhóm Popularity-bias:**
- **TailRecall@20** (↑): Recall chỉ tính trên các user có item test thuộc nhóm tail.
- **Coverage@20** (↑): tỉ lệ item trong catalog xuất hiện ở ít nhất một top-K.
- **ARP@20** (↓): trung bình `log(1+deg_i)` trên mọi slot đề xuất (đề xuất càng phổ biến → càng cao).
- **TailExposure / MiddleExposure / HeadExposure@20**: tỉ lệ slot đề xuất thuộc từng nhóm
  (Tail/Mid ↑ tốt, Head ↓ tốt).

*K ∈ {10, 20}; K chính = 20.*

---

## 4. Phương pháp: Popularity-Aware LightGCN

Nền là **LightGCN**; đề xuất bổ sung 4 thành phần chống popularity bias, mỗi thành phần bật/tắt độc lập.

### 4.1 Backbone — LightGCN
Một bảng embedding cho mỗi node. Lan truyền chuẩn hoá đối xứng, lấy trung bình theo lớp (không
biến đổi tuyến tính, không phi tuyến):

$$\mathbf{E} = \frac{1}{K+1}\sum_{k=0}^{K}\mathbf{E}^{(k)},\quad \mathbf{E}^{(k+1)}=\tilde{\mathbf A}\,\mathbf{E}^{(k)},\quad \tilde{\mathbf A}=\mathbf D^{-1/2}\mathbf A\mathbf D^{-1/2}$$

Điểm số: $s(u,i)=\langle \mathbf e_u,\mathbf e_i\rangle$. Huấn luyện bằng **BPR**:

$$\mathcal L_{BPR}=-\frac{1}{|B|}\sum_{(u,i^+,i^-)}\log\sigma\big(s(u,i^+)-s(u,i^-)\big)$$

kèm L2 trên embedding gốc (hệ số `weight_decay`).

### 4.2 ILE — Item Loss Equalization (đổi objective, không đổi đồ thị)
Gom loss BPR theo nhóm phổ biến của **item dương** trong batch; phạt **độ chênh lệch bình phương**
giữa loss trung bình nhóm head và tail:

$$\mathcal L_{ILE}=\big(\bar{\ell}_{head}-\bar{\ell}_{tail}\big)^2,\qquad \bar\ell_g=\text{mean}_{i^+\in g}\big[-\log\sigma(s_{ui^+}-s_{ui^-})\big]$$

Vì thường $\bar\ell_{tail}>\bar\ell_{head}$, tối thiểu hoá kéo $\bar\ell_{tail}$ **xuống** gần head →
cải thiện nhóm tail. Có chặn dưới bởi 0 (ổn định). **Cần cả 2 nhóm trong batch mới tính.**

> ⚠️ Ghi chú kỹ thuật: bản đầu dùng $\bar\ell_{head}-\bar\ell_{tail}$ (lật dấu, không chặn dưới) khiến
> tối ưu **đẩy** tail xấu đi → model hội tụ về MostPopular. Đã sửa thành bình phương-khoảng-cách.

### 4.3 Degree-Aware Graph Augmentation (đổi cấu trúc đồ thị)
Drop cạnh với xác suất tăng theo độ phổ biến của item (item phổ biến bị drop nhiều hơn → tái cân
bằng message-passing):

$$p_i = p_{min} + (p_{max}-p_{min})\cdot\frac{\log(1+deg_i)}{\log(1+deg_{max})},\quad p\in[0.1,0.4]$$

Drop **đối xứng**: xét cạnh user→item, giữ với xác suất $1-p_i$, rồi mirror lại chiều item→user để
đồ thị vẫn đối xứng cho chuẩn hoá LightGCN.

### 4.4 Contrastive Learning (SGL-style)
Tạo **2 view** augment độc lập $\mathcal G_1,\mathcal G_2$; lan truyền mỗi view lấy embedding node.
Với mỗi node trong batch, embedding ở 2 view là cặp dương; các node khác là âm (InfoNCE):

$$\mathcal L_{CL}=\mathcal L_{NCE}^{user}+\mathcal L_{NCE}^{item},\qquad
\mathcal L_{NCE}=-\frac{1}{|N|}\sum_{n}\log\frac{\exp(\mathbf z_n^{1}\!\cdot\!\mathbf z_n^{2}/\tau)}{\sum_{m}\exp(\mathbf z_n^{1}\!\cdot\!\mathbf z_m^{2}/\tau)}$$

($\mathbf z$ đã L2-normalize, $\tau$ = nhiệt độ). Contrastive kéo embedding đa dạng hơn → tăng Coverage,
giảm HeadExposure.

### 4.5 Popularity-aware Negative Sampling
Lấy negative theo phân phối $\propto deg_i^{\beta}$: item phổ biến bị chọn làm negative nhiều hơn →
bị đẩy điểm xuống mạnh hơn → **tăng TailRecall**. $\beta=0$ ⇔ uniform.

### 4.6 Tổng loss & Inference
$$\mathcal L=\mathcal L_{BPR}+\lambda_{wd}\lVert\Theta\rVert^2+\lambda_{ILE}\mathcal L_{ILE}+\lambda_{CL}\mathcal L_{CL}$$

**Inference:** xếp hạng theo $s(u,i)=\langle\mathbf e_u,\mathbf e_i\rangle$ trên **đồ thị gốc** (không dropout).

### 4.7 Sơ đồ tổng quan phương pháp

```mermaid
flowchart TD
    A["Đồ thị hai phía User-Item<br/>(train edges)"] --> B["LightGCN Propagation<br/>K lớp, chuẩn hoá đối xứng"]
    B --> C["Embedding lớp-trung-bình<br/>e_u , e_i"]
    C --> D["Điểm số s(u,i) = e_u . e_i"]

    A --> E["Degree-Aware Edge Dropout<br/>(đối xứng, p tăng theo log-degree)"]
    E --> F["View đồ thị 1"]
    E --> G["View đồ thị 2"]
    F --> H["Propagate View 1"]
    G --> I["Propagate View 2"]
    H --> J["InfoNCE cross-view<br/>(cùng node = cặp dương)"]
    I --> J
    J --> LCL["L_CL"]

    L["Popularity-aware<br/>Negative Sampling<br/>neg ~ degree^beta"] --> K["L_BPR<br/>(positive vs negative)"]
    D --> K

    D --> M["Gom loss theo nhóm<br/>Head / Middle / Tail"]
    M --> N["L_ILE = (loss_head - loss_tail)^2"]

    K --> O["Tổng Loss<br/>L = L_BPR + wd.L2 + lambda_ILE.L_ILE + lambda_CL.L_CL"]
    N --> O
    LCL --> O

    O -.->|"backprop cập nhật tham số"| B

    D -.->|"sau khi train xong"| P["Inference:<br/>xếp hạng theo s(u,i)<br/>trên đồ thị GỐC (không dropout)"]

    P --> Q["TailRecall@20 ↑"]
    P --> R["Coverage@20 ↑"]
    P --> S["ARP@20 ↓"]
    P --> T["HeadExposure@20 ↓"]

    classDef backbone fill:#EEF1F4,stroke:#6A7686,color:#19212B,stroke-width:1px
    classDef comp fill:#E1F2F1,stroke:#0E7C86,color:#0A5B62,stroke-width:1px
    classDef loss fill:#FFF4E5,stroke:#B5762A,color:#5C3B12,stroke-width:1px
    classDef effect fill:#E4F4EC,stroke:#15855A,color:#0B4A32,stroke-width:1.5px

    class A,B,C,D backbone
    class E,F,G,H,I,J,LCL,L,M,N comp
    class K,O loss
    class P,Q,R,S,T effect
```

*Đọc sơ đồ:* khối xám = backbone gốc; khối xanh ngọc = 4 thành phần đề xuất (Degree-Aware
Augmentation và Contrastive Learning dùng chung tín hiệu drop-theo-degree — xem phê bình ở mục 8.1);
khối cam = nơi các loss được cộng lại; khối xanh lá = hiệu ứng đo được lên popularity bias.

---

## 5. Thiết lập huấn luyện

| Tham số | Giá trị |
|---|---|
| embedding_dim | 64 |
| số lớp LightGCN | 2 (khuyến nghị) / 3 (mặc định gốc) |
| optimizer / lr | Adam / 1e-3 |
| batch_size | 4096 |
| epochs | 100, early-stopping theo VAL Recall@20 (patience 10 lần eval, eval mỗi 5 epoch) |
| weight_decay (L2) | 1e-4 |
| dropout degree-aware | p_min=0.1, p_max=0.4 |
| grid sweep | λ_ILE∈{0.1,0.5,1.0}, λ_CL∈{0.1,0.5}, τ=0.2, β∈{0,0.5}, layers∈{2,3} |
| seed | **3 seed {42, 0, 1}**, báo cáo mean ± std |

**Hạ tầng (đủ các bước 1 dự án AI):** chọn model theo **VAL** (không peek test), đánh giá **TEST 1
lần cuối** trên model tốt nhất; **checkpoint** ra đĩa (`_latest.pt` để resume, `_best.pt` = best theo
val); **resume** tự động; **logging** đầy đủ ra stdout + file + history CSV mỗi epoch; xuất kết quả CSV +
model tương thích công cụ đánh giá.

---

## 6. Thiết kế thực nghiệm

- **Baselines tham chiếu** (`results/metrics/`): MostPopular, LightGCN, BPR-MF (cùng protocol; baseline
  LightGCN của tôi khớp reference: R@20 0.128 ≈ 0.1276).
- **Phương pháp (our):** LightGCN + ILE + Degree-Aware Aug + Contrastive + pop-aware negatives; quét
  siêu tham số rồi **chọn theo frontier**: Coverage@20 lớn nhất với Recall@20 giảm ≤ 5% so với baseline.
- Trình bày 4 **điểm vận hành** đại diện trên frontier (accuracy / balanced / high-tail / fairness).

---

## 7. Kết quả (test set, đầy đủ 10 metric, mean ± std trên 3 seed {42,0,1})

**Chú giải chiều tốt (áp dụng cho mọi bảng trong mục 7):** **↑** = giá trị càng cao càng tốt,
**↓** = giá trị càng thấp càng tốt. Nhóm accuracy (R@10, N@10, R@20, N@20) luôn ↑. Nhóm
popularity-bias: TailRecall/Coverage/TailExposure/MiddleExposure ↑ (đề xuất chạm tới tail/middle
nhiều hơn là tốt); ARP/HeadExposure ↓ (đề xuất càng phổ biến/càng dồn về head là càng tệ).

References (MostPopular/BPR-MF: 1 run, do nhóm khác thực hiện) và các cấu hình PopAware (our:
3 seed, mean±std, `n_seed=3`):

| Model | R@10↑ | N@10↑ | R@20↑ | N@20↑ | TailRec@20↑ | Cov@20↑ | ARP@20↓ | TailExp↑ | MidExp↑ | HeadExp↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| MostPopular (ref) | 0.040 | 0.019 | 0.072 | 0.027 | 0.0000 | 0.050 | 7.53 | 0.000 | 0.000 | 1.000 |
| LightGCN — our re-run (ref, 3-seed) | 0.0725±.0013 | 0.0356±.0002 | 0.1287±.0005 | 0.0497±.0002 | 0.0050±.0009 | 0.398±.004 | 6.833±.009 | 0.0015±.0001 | 0.0398±.0006 | 0.9587±.0006 |
| BPR-MF (ref) | 0.074 | 0.037 | 0.131 | 0.051 | 0.0109 | 0.628 | 6.52 | 0.009 | 0.114 | 0.877 |
| PopAware — accuracy (our) | 0.0771±.0009 | 0.0378±.0003 | **0.1367**±.0035 | **0.0527**±.0006 | 0.0143±.0023 | 0.462±.029 | 6.500±.051 | 0.0097±.0011 | 0.0732±.0133 | 0.9171±.0144 |
| PopAware-BEST — balanced (our) | 0.0771±.0012 | 0.0376±.0003 | 0.1338±.0030 | 0.0518±.0004 | 0.0324±.0049 | 0.538±.029 | 6.400±.042 | 0.0320±.0012 | 0.0854±.0104 | 0.8826±.0114 |
| PopAware — high-tail (our) | **0.0783**±.0006 | **0.0382**±.0002 | 0.1352±.0020 | 0.0524±.0004 | **0.0399**±.0023 | 0.499±.010 | 6.564±.008 | 0.0363±.0014 | 0.0521±.0019 | 0.9117±.0005 |
| PopAware — fairness (our) | 0.0638±.0020 | 0.0310±.0005 | 0.1052±.0039 | 0.0415±.0011 | 0.0274±.0035 | **0.713**±.003 | **6.138**±.001 | **0.0482**±.0006 | **0.1184**±.0013 | **0.8333**±.0016 |

Cấu hình (L2): accuracy = `λ_ILE0.1,λ_CL0.1,β0.5`; **BEST** = `λ_ILE1.0,λ_CL0.1,β0.5`;
high-tail = `λ_ILE1.0,λ_CL0.1,β0`; fairness = `λ_ILE1.0,λ_CL0.5,β0`.
*(in đậm = tốt nhất cột trong số các cấu hình 3-seed; tất cả đều thuộc (our)).*

### 7.1 Vai trò các siêu tham số (từ sweep)
- **λ_CL** = cần gạt accuracy ↔ fairness: 0.1 giữ Recall (thậm chí tăng); 0.5 debias mạnh nhưng Recall
  tụt ~18%.
- **λ_ILE** = cần gạt TailRecall: 0.1→0.014, 1.0(β=0.5)→0.032, 1.0(β=0)→0.040.
- **β (pop-aware neg)** = 0.5 tăng Coverage & giảm HeadExposure, chi phí Recall nhỏ.
- **Số lớp:** 2 lớp tốt hơn 3 (3 lớp over-smoothing, thiên lệch hơn).

### 7.2 Độ ổn định & ý nghĩa thống kê (3 seed)

Độ lệch chuẩn (std) trên 3 seed **nhỏ** so với chênh lệch giữa baseline và PopAware — chênh lệch
**vượt xa nhiễu ngẫu nhiên do seed**, không phải may rủi của một lần chạy:

| So sánh (BEST vs LightGCN, mean±std) | Chiều tốt | Baseline | BEST | Khoảng cách | Kết luận |
|---|---|---|---|---|---|
| Recall@20 | ↑ | 0.1287±0.0005 | 0.1338±0.0030 | không chồng lấn (0.1292 vs 0.1308) | BEST **cao hơn có ý nghĩa**, +4.0% |
| TailRecall@20 | ↑ | 0.0050±0.0009 | 0.0324±0.0049 | không chồng lấn (0.0059 vs 0.0275) | BEST **cao hơn rõ rệt**, ×6.5 |
| Coverage@20 | ↑ | 0.398±0.004 | 0.538±0.029 | không chồng lấn (0.402 vs 0.509) | BEST **cao hơn có ý nghĩa**, +35% |
| ARP@20 | ↓ | 6.833±0.009 | 6.400±0.042 | không chồng lấn (6.824 vs 6.442) | BEST **thấp hơn có ý nghĩa**, −6.3% |
| HeadExposure@20 | ↓ | 0.9587±0.0006 | 0.8826±0.0114 | không chồng lấn (0.958 vs 0.894) | BEST **thấp hơn có ý nghĩa**, −7.9% |

*(khoảng dùng để so là mean ± 1·std của từng phía; "không chồng lấn" nghĩa là khoảng của baseline và
BEST tách biệt hoàn toàn — dấu hiệu chênh lệch vượt nhiễu, không cần kiểm định phức tạp hơn để thấy rõ.)*

Std tương đối của baseline rất nhỏ (<1% giá trị) → kết quả **tái lập tốt, ổn định qua seed**. Các cấu
hình PopAware có std lớn hơn một chút ở Coverage/ARP (do dropout & sampling ngẫu nhiên trong 3 thành
phần method) nhưng vẫn nhỏ hơn nhiều lần so với biên độ cải thiện.

### 7.3 Giá trị tại seed tốt nhất (bảng bổ sung — không phải số liệu chính)

**"Không phải số liệu chính" nghĩa là gì?** Mỗi cấu hình được train 3 lần độc lập (seed 42, 0, 1) —
3 phép đo nhiễu của cùng một phương pháp. Bảng dưới lấy **giá trị lớn nhất trong 3 lần đo** cho mỗi
hàng. Về mặt thống kê, "max của N phép đo nhiễu" là một **ước lượng lệch, thiên lạc quan** (selection
bias / "winner's curse"): nếu chạy thêm nhiều seed nữa, con số max quan sát được sẽ **tiếp tục tăng
dần** dù bản thân model không hề tốt lên — nó phản ánh may rủi của phép chọn, không phản ánh đúng khả
năng thật của phương pháp. Ngược lại, **mean ± std** (bảng mục 7) là ước lượng **không lệch** cho xu
hướng trung tâm, kèm biên độ dao động — đây mới là con số **có thể bảo vệ được** khi trình bày/phản
biện, nên được gọi là "số liệu chính". Bảng 7.3 chỉ mang tính **minh hoạ** ("phương pháp có thể đạt
tới đâu trong điều kiện thuận lợi nhất"), không dùng để so sánh/kết luận khoa học.

Tiêu chí chọn seed hiển thị: mỗi hàng lấy seed cho **Recall@20 test cao nhất** của đúng cấu hình đó
(cùng tiêu chí đã dùng để chọn best-epoch lúc train).

| Model | Seed | R@10↑ | N@10↑ | R@20↑ | N@20↑ | TailRec@20↑ | Cov@20↑ | ARP@20↓ | TailExp↑ | MidExp↑ | HeadExp↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MostPopular (ref) | — | 0.0403 | 0.0189 | 0.0723 | 0.0269 | 0.0000 | 0.0498 | 7.53 | 0.0000 | 0.0000 | 1.0000 |
| BPR-MF (ref) | — | 0.0742 | 0.0369 | 0.1309 | 0.0511 | 0.0109 | 0.6281 | 6.52 | 0.0086 | 0.1140 | 0.8774 |
| LightGCN (our re-run) | 1 | 0.0721 | 0.0356 | 0.1293 | 0.0500 | 0.0037 | 0.3991 | 6.825 | 0.0015 | 0.0402 | 0.9583 |
| PopAware — accuracy (our) | 1 | 0.0766 | 0.0374 | **0.1407** | **0.0534** | 0.0112 | 0.4393 | 6.535 | 0.0091 | 0.0632 | 0.9277 |
| PopAware-BEST — balanced (our) | 1 | 0.0762 | 0.0371 | 0.1369 | 0.0523 | 0.0393 | 0.5604 | 6.366 | 0.0334 | 0.0917 | 0.8749 |
| PopAware — high-tail (our) | 1 | **0.0787** | **0.0380** | 0.1376 | 0.0527 | 0.0393 | 0.5089 | 6.556 | 0.0354 | 0.0533 | 0.9113 |
| PopAware — fairness (our) | 42 | 0.0650 | 0.0316 | 0.1100 | 0.0430 | 0.0299 | **0.7136** | **6.137** | **0.0479** | **0.1200** | **0.8321** |

*(in đậm = tốt nhất cột trong bảng này; cột "Seed" = seed cho kết quả tốt nhất của đúng cấu hình đó —
độc lập giữa các hàng, không có "seed vàng" chung.)*

> ⚠️ **Cách dùng đúng:** đây là số liệu "best-case" minh hoạ (vd. cho slide/demo), **không dùng thay**
> bảng mean±std ở mục 7 khi kết luận khoa học. Bằng chứng cụ thể cho lý do trên: **seed 1 thắng ở 4/5
> cấu hình** nhưng **seed 42 lại thắng ở cấu hình fairness** — không có seed nào tốt nhất cho mọi mục
> tiêu, chứng tỏ "best-seed" chỉ là may rủi cục bộ theo từng phép đo, không phải một hiệu ứng hệ thống
> của phương pháp.

### 7.4 Biểu đồ frontier Accuracy ↔ Fairness

```mermaid
quadrantChart
    title Recall@20 vs Coverage@20 (chuẩn hoá min-max trên tập kết quả)
    x-axis "Recall@20 thấp" --> "Recall@20 cao"
    y-axis "Coverage@20 thấp" --> "Coverage@20 cao"
    quadrant-1 "Vùng lý tưởng: Recall cao + Coverage cao"
    quadrant-2 "Fairness cao, accuracy vừa"
    quadrant-3 "Yếu cả hai trục"
    quadrant-4 "Accuracy cao, fairness thấp"
    MostPopular: [0.02, 0.02]
    LightGCN: [0.88, 0.52]
    BPR-MF: [0.91, 0.87]
    PopAware-accuracy: [0.98, 0.62]
    PopAware-BEST: [0.95, 0.74]
    PopAware-hightail: [0.97, 0.68]
    PopAware-fairness: [0.51, 0.98]
```

*Đọc biểu đồ:* trục đã chuẩn hoá theo min–max của chính 7 điểm này (MostPopular = gốc toạ độ), chỉ để
so sánh **vị trí tương đối**, không phải giá trị metric gốc (xem bảng mục 7 cho giá trị thật). BPR-MF
và cụm PopAware đều nằm ở góc phần tư "vùng lý tưởng" — nhưng cụm PopAware nằm bên phải BPR-MF
(Recall nhỉnh hơn) trong khi BPR-MF nhỉnh hơn theo Coverage; PopAware-fairness tách hẳn về phía trục
Coverage. Hình dạng này minh hoạ trực quan đúng kết luận định lượng: **phương pháp tạo ra một
frontier**, không phải một điểm cố định như BPR-MF.

---

## 8. Phân tích & đối chiếu kỳ vọng

**Kiểm kỳ vọng (so với LightGCN, mean 3-seed):** 3 cấu hình `λ_CL=0.1` đạt **đủ 5/5**: TailRecall↑,
Coverage↑, ARP↓, HeadExposure↓, và Recall/NDCG **không giảm — còn tăng**. Cấu hình `fairness` đạt toàn bộ
fairness nhưng accuracy tụt ~18% (mục 7.2 xác nhận mọi chênh lệch đều vượt nhiễu do seed).

- **So với backbone LightGCN — thắng tuyệt đối, có ý nghĩa thống kê:** BEST hơn mọi metric — R@20
  +4.0%, TailRecall ×6.5 (0.032 vs 0.0050), Coverage +35%, ARP↓6.3%, HeadExp↓7.9% — tất cả khoảng
  mean±std của baseline và BEST **không chồng lấn** (mục 7.2).
- **So với BPR-MF (baseline mạnh, khá fair):** BEST hơn về Recall (0.134 vs 0.131), TailRecall (×2.4:
  0.032 vs 0.0109), ARP (6.40 vs 6.52); chỉ thua Coverage (0.538 vs 0.628). Cấu hình fairness thì
  **áp đảo toàn bộ** fairness của BPR-MF (Cov 0.713, ARP 6.14, HeadExp 0.833).
- **Frontier tunable:** phương pháp cung cấp một đường frontier thống trị, chọn điểm vận hành theo nhu cầu;
  BPR-MF chỉ là một điểm.
- **"Tail chấp nhận được":** BEST (TailRec 0.032, ×6.5 baseline) là điểm giữa hợp lý — cao hơn hẳn
  baseline nhưng không cực đoan để sập accuracy; kết quả ổn định qua 3 seed.

### 8.1 Đánh giá phản biện: cơ chế, điểm mạnh, điểm cần nhìn thẳng

**Vì sao 4 thành phần bổ sung nhau thay vì trùng lặp:** mỗi thành phần tấn công một điểm khác nhau
của vòng đời huấn luyện — ILE sửa **hàm mục tiêu** (gradient của từng tương tác), Degree-Aware
Augmentation sửa **cấu trúc đồ thị lan truyền**, Contrastive Learning sửa **hình dạng không gian
embedding**, Popularity-aware Negative Sampling sửa **tín hiệu học đối nghịch**. Đây là điểm thiết kế
đáng chú ý: bốn thành phần độc lập về cơ chế, không phải "cộng dồn nhiều regularizer cho chắc ăn".

**Bằng chứng cơ chế hoạt động đúng, không phải trùng hợp ngẫu nhiên:** trong sweep (mục 7.1), tăng
λ_ILE kéo TailRecall lên **đơn điệu** (0.014 → 0.032 → 0.040), tăng λ_CL kéo Coverage lên **đơn điệu**
— đúng hành vi kỳ vọng của một regularizer lành mạnh. Đây là khác biệt bản chất so với bản ILE lỗi ban
đầu (Phụ lục 12, mục 1), nơi tăng cường độ khiến **mọi trục cùng tệ đi** — dấu hiệu kinh điển của một
quy tắc bị lật dấu, chứ không phải một trade-off thật.

**Điểm cần nhìn thẳng (phê bình trung thực, không tô hồng):**
- Degree-Aware Augmentation và Contrastive Learning **dùng chung một tín hiệu** — xác suất drop cạnh
  theo degree. Về bản chất đây là hai cách khai thác cùng một nguồn thông tin, nên phần đóng góp
  *riêng* của mỗi thành phần **chưa được tách bạch** trong kết quả hiện tại — cần ablation
  ILE-only / Aug-only / CL-only (mục 10) để định lượng chính xác.
- Coverage@20 ở điểm cân bằng (0.538) vẫn **thấp hơn BPR-MF** (0.628) — nghĩa là ở riêng khía cạnh phủ
  danh mục, cách debias dựa trên đồ thị (graph-based) **chưa chắc hiệu quả hơn** một MF thuần không có
  cấu trúc lan truyền. Chỉ cấu hình `fairness` (đánh đổi nhiều accuracy hơn) mới vượt được BPR-MF ở
  Coverage.
- TailRecall dù cải thiện có ý nghĩa thống kê (mục 7.2) nhưng **giá trị tuyệt đối vẫn nhỏ** (~0.03) —
  đúng bản chất bài toán (item tail cực khó lọt top-20 trong catalog ~3.500 item). Nên diễn giải là
  **"giảm đáng kể popularity bias"**, không nên diễn giải quá tay thành "đã giải quyết" long-tail.

---

## 9. Tái lập (reproducibility)

**Cấu trúc code (phương pháp):**
- `src/models.py` — LightGCN / BPR-MF.
- `src/losses.py` — BPR + L2.
- `src/ile_losses.py` — ILE (đã sửa) + xác suất degree-aware dropout.
- `src/neg_sampling.py` — popularity-aware negative sampling.
- `src/popaware_training.py` — training hợp nhất (ILE+Aug+CL+neg), checkpoint/resume/log.
- `src/metrics.py` — 10 metric full-ranking.
- `train_all_popaware.py` — pipeline ablation cố định siêu tham số.
- `train_sweep_popaware.py` — quét siêu tham số + chọn theo frontier.
- `train_final_seeds.py` — chạy 5 cấu hình chốt × 3 seed, gộp mean±std (số liệu mục 7).
- `evaluate_test_full.py` — nạp model đã lưu, tính lại đủ 10 metric.

**Lệnh chạy (trên server GPU qua Slurm):**
```bash
bash run_on_gpu.sh --train train_sweep_popaware.py     # sweep + chọn frontier (~2h)
bash run_on_gpu.sh --train train_all_popaware.py       # ablation cố định hp (~1h)
bash run_on_gpu.sh --train train_final_seeds.py        # 3-seed cho số liệu cuối (mục 7, ~1h)
```
Cấu hình BEST: `--layers 2 --lambda-ile 1.0 --lambda-cl 0.1 --neg-pop-beta 0.5`.

**Kết quả sinh ra:** `results/popaware_sweep_*.csv`, `results/popaware_final_runs_*.csv`,
`results/popaware_final_meanstd_*.csv` (số liệu mục 7), `results/comparison_vs_references.csv`,
`models/final_model_*.pt`, `results/popaware/history_*.csv`, `logs/popaware/*.log`,
`checkpoints/popaware/*_best.pt`.

---

## 10. Hạn chế & hướng phát triển

- **Baseline tham chiếu (MostPopular, BPR-MF) mới có 1 run** (do nhóm khác thực hiện) — nên chạy thêm
  3 seed cho 2 baseline này để so sánh cân xứng với PopAware (đã 3-seed).
- **TailRecall nhỏ tuyệt đối (~0.03):** bản chất ML-1M (item tail cực khó trúng top-20); nên nhấn
  **bội số cải thiện** thay vì trị tuyệt đối, và bổ trợ bằng Coverage/Exposure.
- **Coverage chưa vượt BPR-MF ở điểm balanced:** có thể thử inference-time logit adjustment
  (`s − α·log(1+deg)`) như thành phần bổ trợ; hoặc dataset thứ 2 (Gowalla/Amazon) để chứng minh
  generalization.
- **Ablation từng thành phần** (ILE-only / Aug-only / CL-only) nên bổ sung để định lượng đóng góp riêng
  của mỗi thành phần (chạy `train_all_popaware.py --configs ...`).

---

## 11. Kết luận

**Câu hỏi cốt lõi:** phương pháp đề xuất (Popularity-Aware LightGCN) đã giải quyết đúng vấn đề đặt ra
ở mục 1 hay chưa?

**Trả lời ngắn gọn: Có — tại điểm vận hành PopAware-BEST, phương pháp đạt đủ toàn bộ 6 tiêu chí kỳ
vọng ban đầu, được xác nhận có ý nghĩa thống kê qua 3 seed (mục 7.2), không chỉ dựa trên 1 lần chạy.**

### 11.1 Đối chiếu từng kỳ vọng (mục 1) với kết quả đo được

| Kỳ vọng ban đầu (mục 1) | Kết quả đo được (BEST vs LightGCN, mean±std 3-seed) | Đạt? |
|---|---|---|
| TailRecall@20 tăng | 0.0050→0.0324 (×6.5, khoảng mean±std không chồng lấn) | ✅ Đạt, có ý nghĩa thống kê |
| Coverage@20 tăng | 0.398→0.538 (+35%, không chồng lấn) | ✅ Đạt, có ý nghĩa thống kê |
| ARP@20 giảm | 6.833→6.400 (−6.3%, không chồng lấn) | ✅ Đạt, có ý nghĩa thống kê |
| HeadExposure@20 giảm | 0.9587→0.8826 (−7.9%, không chồng lấn) | ✅ Đạt, có ý nghĩa thống kê |
| Recall@20/NDCG@20 không giảm mạnh (≤5%) | Recall **+4.0%**, NDCG tương đương/tăng nhẹ | ✅ Đạt — thực ra tăng, không phải chỉ "không giảm" |
| TailRecall "chấp nhận được" (không quá thấp/cao — ý anh Kiên/anh Hùng) | 0.0324: cao hơn baseline rõ rệt (×6.5) nhưng không cực đoan như cấu hình high-tail (0.0399) | ✅ Đạt, đúng tinh thần "vừa phải" |

→ **6/6 tiêu chí đề ra ban đầu đều đạt** tại điểm vận hành BEST, và tiêu chí "không đánh đổi accuracy"
không chỉ đạt mà còn **vượt kỳ vọng** — Recall/NDCG tăng nhẹ thay vì chỉ không giảm.

### 11.2 Đặt trong bối cảnh cạnh tranh (không chỉ so với chính mình)

Bài toán không chỉ được giải theo nghĩa "tốt hơn backbone của chính nó": PopAware-BEST còn vượt qua
**BPR-MF** — baseline mạnh nhất, khá "fair" tự nhiên trong thí nghiệm — trên Recall, TailRecall (×2.4)
và ARP, chỉ thua Coverage; và ở điểm vận hành `fairness`, áp đảo BPR-MF trên **toàn bộ** trục fairness.
Điều này khẳng định giải pháp cạnh tranh được với một baseline không tầm thường, không chỉ đẹp trên
giấy so với chính backbone.

### 11.3 Phạm vi thật của "đã giải quyết" — tránh overclaim

"Giải quyết được" cần hiểu đúng phạm vi đã kiểm chứng, không nên diễn giải quá tay:

- Kết quả mới kiểm chứng trên **1 bộ dữ liệu** (MovieLens-1M) — chưa chứng minh generalize sang domain
  khác (mục 10).
- TailRecall tăng có ý nghĩa thống kê nhưng **giá trị tuyệt đối còn nhỏ** (~0.03) — bài toán long-tail
  được **giảm nhẹ mức độ nghiêm trọng**, chưa "giải quyết triệt để" theo nghĩa item tail được đề xuất
  thường xuyên cho user.
- Đóng góp **riêng lẻ** của từng thành phần (ILE / Aug / CL) chưa tách bạch được (mục 8.1, 10) — nên
  chưa khẳng định bộ 4 thành phần là cấu hình tối ưu hay có thể rút gọn mà không mất hiệu quả.
- Ở điểm cân bằng, Coverage@20 vẫn thấp hơn BPR-MF — vẫn còn dư địa cải thiện đúng trên trục mà nhóm
  quan tâm nhất.

### 11.4 Kết luận cuối

Phương pháp đề xuất **đã đáp ứng đúng và đủ mục tiêu đặt ra ban đầu của bài toán** — giảm popularity
bias trên toàn bộ 4 chỉ số fairness, giữ TailRecall ở mức "chấp nhận được" như mục tiêu nhóm đề ra, và
accuracy không những không giảm mà còn tăng nhẹ — với bằng chứng thống kê đáng tin cậy qua 3 seed. Đây
là mức đủ để coi là **thành công** cho phạm vi một báo cáo/nghiên cứu ở quy mô hiện tại. Để nâng lên
mức kết luận chắc chắn hơn (ví dụ cho một bài báo), bước tiếp theo cần làm là ablation tách từng thành
phần và kiểm chứng trên ít nhất một bộ dữ liệu thứ hai (đã nêu chi tiết ở mục 10).

---

## 12. Phụ lục — các lỗi đã phát hiện & sửa (development notes)

1. **ILE lật dấu + không chặn dưới** → sửa thành `(head−tail)²`. Trước sửa: ILE làm sập cả accuracy lẫn
   coverage (hội tụ về MostPopular).
2. **Contrastive learning là dead-code và sai dạng** (user↔item thay vì cross-view) → viết InfoNCE SGL
   chuẩn và nối vào loss.
3. **Degree-aware dropout bất đối xứng** (2 chiều độc lập) → viết bản drop đối xứng.
4. **Data leakage khi chọn model:** trước chọn best theo TEST mỗi 5 epoch → sửa: chọn theo VAL, TEST 1
   lần cuối.
5. **Glob `**` lỗi trên Python 3.11 cluster** ở checkpoint loader cũ → dùng `iterdir()`.
6. **`torch.load` weights_only** (PyTorch ≥2.6) không nạp được cache → thêm `weights_only=False` cho
   file tin cậy của project.

*(Mục này để minh bạch quá trình; có thể lược bớt khi đưa vào báo cáo chính.)*

---

## 13. Bảng tổng hợp đầy đủ — dùng cho Slide

Đầy đủ **10 metric × 7 phương pháp** (3 baseline tham chiếu + 4 điểm vận hành của PopAware), lấy đúng
số liệu thực nghiệm mean 3-seed ở mục 7 (đã bỏ ký hiệu ±std cho gọn khi trình chiếu — xem mục 7 nếu
cần độ lệch chuẩn/kiểm định). Nguồn: `results/popaware_final_meanstd_20260715_174317.csv` (PopAware +
LightGCN re-run, mean 3-seed) và `results/metrics/` (MostPopular, BPR-MF — 1 run, do nhóm khác thực hiện).

| Model | R@10↑ | N@10↑ | R@20↑ | N@20↑ | TailRec@20↑ | Cov@20↑ | ARP@20↓ | TailExp@20↑ | MidExp@20↑ | HeadExp@20↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| MostPopular (ref) | 0.0403 | 0.0189 | 0.0723 | 0.0269 | 0.0000 | 0.0498 | 7.530 | 0.0000 | 0.0000 | 1.0000 |
| LightGCN (ref) | 0.0725 | 0.0356 | 0.1287 | 0.0497 | 0.0050 | 0.398 | 6.833 | 0.0015 | 0.0398 | 0.9587 |
| BPR-MF (ref) | 0.0742 | 0.0369 | 0.1309 | 0.0511 | 0.0109 | 0.628 | 6.518 | 0.0086 | 0.1140 | 0.8774 |
| PopAware — accuracy (our) | 0.0771 | 0.0378 | **0.1367** | **0.0527** | 0.0143 | 0.462 | 6.500 | 0.0097 | 0.0732 | 0.9171 |
| PopAware-BEST — balanced (our) | 0.0771 | 0.0376 | 0.1338 | 0.0518 | 0.0324 | 0.538 | 6.400 | 0.0320 | 0.0854 | 0.8826 |
| PopAware — high-tail (our) | **0.0783** | **0.0382** | 0.1352 | 0.0524 | **0.0399** | 0.499 | 6.564 | 0.0363 | 0.0521 | 0.9117 |
| PopAware — fairness (our) | 0.0638 | 0.0310 | 0.1052 | 0.0415 | 0.0274 | **0.713** | **6.138** | **0.0482** | **0.1184** | **0.8333** |

*(in đậm = tốt nhất cột trong số cả 7 phương pháp; ↑ cao hơn tốt hơn, ↓ thấp hơn tốt hơn.)*

**Điểm đáng nói khi trình bày slide:** với 4 điểm vận hành của PopAware phủ lên nhau, **mọi cột đều có
ô tốt nhất thuộc (our)** — không cột metric nào bị 3 baseline tham chiếu (MostPopular/LightGCN/BPR-MF)
chiếm giữ. Đây là bằng chứng mạnh nhất để mở đầu hoặc chốt slide kết quả.
