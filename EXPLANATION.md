# EXPLANATION.md — Popularity-Aware LightGCN: Giáo trình giải thích từ đầu

*Tài liệu này viết cho người **chưa từng học** LightGCN hay Recommender Systems. Mọi công thức, hành vi, con số cụ thể (degree, số user/item, tên hàm...) đều lấy trực tiếp từ sơ đồ/code thật của dự án. Những đoạn giải thích **trực giác/lý do chung** (vì sao kỹ thuật X thường hoạt động tốt, so sánh với kỹ thuật khác...) là kiến thức nền phổ biến trong ngành Machine Learning, được đánh dấu rõ bằng *(kiến thức nền)* — không phải phát biểu của tác giả bài báo/đồ án.*

> **Nguyên tắc trình bày của tài liệu này:** các khối được sắp xếp theo đúng **thứ tự phụ thuộc logic** — một khối chỉ được giải thích SAU KHI mọi khái niệm nó cần đã được giới thiệu trước đó. Vì vậy **thứ tự trong tài liệu này khác với số ①-⑦ trên sơ đồ gốc** (sơ đồ vẽ theo bố cục không gian, không phải theo thứ tự nên đọc để dễ hiểu nhất). Mỗi khối đều có ghi chú "tương ứng khối nào trên sơ đồ".

> **Về ví dụ số xuyên suốt tài liệu:** để bạn thấy rõ dữ liệu thật sự "chảy" qua các khối như thế nào (không phải mỗi khối một ví dụ rời rạc), toàn bộ tài liệu dùng **CHUNG một bộ dữ liệu đồ chơi duy nhất** (định nghĩa ngay bên dưới). Số liệu ở mỗi khối sau đều được **tính ra thật** từ số liệu của khối trước — không có khối nào bịa số mới không liên quan. Vài chỗ tính toán quá dài (ví dụ lan truyền đầy đủ trên 2 view augment) sẽ được rút gọn và ghi rõ "minh hoạ đơn giản hoá", nhưng vẫn dùng đúng user/item/degree của bộ dữ liệu chung này.

### Bộ dữ liệu đồ chơi dùng xuyên suốt (đọc 1 lần, dùng lại ở mọi khối)

- **4 user:** u1, u2, u3, u4. **3 item:** i1 (phổ biến nhất — nhóm *head*), i2 (trung bình — nhóm *middle*), i3 (ít phổ biến nhất — nhóm *tail*).
- **Cạnh train:** (u1,i1), (u2,i1), (u3,i1), (u4,i1), (u1,i2), (u3,i2), (u4,i3) — 7 cạnh.
- **Degree:** `deg(i1)=4` (head), `deg(i2)=2` (middle), `deg(i3)=1` (tail); `deg(u1)=2, deg(u2)=1, deg(u3)=2, deg(u4)=2`.
- **Embedding ban đầu `E^(0)`** (2 chiều, để tính tay được — thật ra là 64 chiều):
  `u1=[0.10,-0.20]`, `u2=[0.05,0.15]`, `u3=[-0.10,0.05]`, `u4=[0.20,-0.05]`, `i1=[0.30,0.10]`, `i2=[-0.05,-0.15]`, `i3=[0.15,0.25]`.

```
u1 ──┬── i1 (head, deg=4) ──┬── u2
     └── i2 (middle, deg=2)─┤
u3 ──┬── i1                 ├── u3
     └── i2                 │
u4 ──┬── i1                 └── u4
     └── i3 (tail, deg=1) ── u4
```

---

## Trước khi bắt đầu: 3 khái niệm nền tảng

- **Embedding**: một vector số thực (ví dụ 64 con số) đại diện cho một user hoặc một item. Hai thực thể "giống nhau về sở thích/đặc điểm" sẽ có embedding gần nhau trong không gian nhiều chiều đó. *(kiến thức nền)*
- **Đồ thị hai phía (bipartite graph)**: cấu trúc dữ liệu gồm 2 loại đỉnh (user, item), chỉ có cạnh nối giữa 2 loại khác nhau.
- **Popularity Bias**: hiện tượng hệ thống gợi ý có xu hướng chỉ đề xuất item đã phổ biến sẵn, khiến item ít phổ biến (long-tail) gần như không bao giờ được nhìn thấy — đây là vấn đề toàn bộ mô hình được thiết kế để giảm bớt.

## Bảng ký hiệu — tra cứu bất cứ lúc nào nếu quên một ký hiệu là gì

Tài liệu dùng khá nhiều ký hiệu gần giống nhau; bảng dưới đây tách rõ nghĩa từng cái để không nhầm lẫn khi đọc các khối sau:

| Ký hiệu | Ý nghĩa | Là gì về mặt dữ liệu |
|---|---|---|
| `e_u`, `e_i` | Embedding của **một** user/item cụ thể | 1 vector (ví dụ `[0.09,-0.28]`) |
| `E`, `E^(0)`, `E^(k)` | Embedding của **TẤT CẢ** user+item cùng lúc | 1 ma trận nhiều dòng (mỗi dòng 1 node) |
| Số mũ `(0)`, `(k)`, `(K)` | **Lớp lan truyền** thứ mấy — không phải số mũ toán học (không phải "E bình phương") | `(0)`=chưa lan truyền gì, `(k)`=sau `k` vòng "hỏi hàng xóm" |
| `deg(x)` | Degree (số cạnh) của node `x` | 1 số nguyên |
| `s(u,i)`, `s⁺`, `s⁻` | Điểm dự đoán (Khối 4) — `s⁺`=điểm item đã thích, `s⁻`=điểm item chưa thích | 1 số thực |
| `ℓ` (chữ thường) | Loss của **một** interaction/cặp cụ thể | 1 số thực |
| `L` hoặc `𝓛` (chữ hoa, có subscript như `L_BPR`) | Loss đã **cộng/trung bình** trên cả batch — một loại, ví dụ chỉ riêng BPR | 1 số thực |
| `𝓛` không subscript (Total Loss) | Tổng có trọng số của **cả 4** loss cộng lại | 1 số thực, số cuối cùng gọi `.backward()` |
| `z1, z2` | Embedding đã **chuẩn hoá độ dài = 1** (chỉ dùng ở Contrastive Loss, Khối 10) | 1 vector, khác với `e` (embedding gốc, độ dài bất kỳ) |

**Quy tắc chung dễ nhớ:** chữ thường = của một node/một cặp cụ thể; chữ hoa = đã gộp/tính trên nhiều node hoặc nhiều cặp.

## Bảng ký hiệu toán học tổng quát — tra cứu khi gặp 1 ký hiệu lạ trong công thức

Đây là các ký hiệu toán thuần tuý (không riêng của đồ án), xuất hiện lặp lại ở nhiều công thức. Đọc 1 lần ở đây, các khối sau chỉ nhắc lại ngắn gọn:

| Ký hiệu | Tên gọi | Ý nghĩa |
|---|---|---|
| `∪` | Hợp (union) | Gộp 2 tập hợp thành 1 tập lớn hơn, chứa mọi phần tử của cả 2 (không lặp lại). |
| `∈` | Thuộc (element of) | "Là một phần tử của" — ví dụ `x∈[0.1,0.4]` nghĩa là `x` nằm trong khoảng đó. |
| `⊆` | Tập con (subset) | "Là một phần (có thể bằng toàn bộ) của tập lớn hơn". |
| `×` (giữa 2 tập hợp) | Tích Descartes (Cartesian product) | Tập hợp TẤT CẢ các cặp có thể ghép từ 2 tập — ví dụ `U×I` là mọi cặp (user,item) có thể có, kể cả cặp chưa từng tương tác. |
| `Σ` (sigma hoa) | Tổng (sum) | "Cộng dồn" biểu thức phía sau, lặp lại theo chỉ số ghi dưới/trên ký hiệu Σ — ví dụ `Σ_{d=1}^{64}` nghĩa là cộng từ `d=1` đến `d=64`. |
| `⟨a,b⟩` | Tích vô hướng (dot product) | Tên gọi khác của phép "nhân từng toạ độ tương ứng của 2 vector rồi cộng lại" — xem công thức khai triển ngay bên cạnh mỗi lần ký hiệu này xuất hiện. |
| `∝` | Tỷ lệ thuận (proportional to) | Vế trái tăng/giảm THEO ĐÚNG TỶ LỆ với vế phải, nhưng **chưa phải dấu bằng** — muốn có xác suất/giá trị thật, phải chia cho tổng của tất cả các vế phải có thể có (bước "chuẩn hoá"). |
| `‖v‖` | Chuẩn (norm) — độ dài vector | `‖v‖ = \sqrt{\sum_d v[d]^2}` (căn bậc 2 của tổng bình phương từng toạ độ). `‖v‖²` (chuẩn bình phương, hay gặp hơn) bỏ luôn dấu căn: chỉ là tổng bình phương từng toạ độ. |
| `Aᵀ` hoặc `A^⊤` | Chuyển vị (transpose) | Lật ma trận: đổi hàng thành cột. Dùng khi nhân 2 ma trận cần "khớp chiều" với nhau. |
| `x̄` (gạch ngang trên đầu) | Trung bình (mean) | Ký hiệu quy ước cho "giá trị trung bình của x". |
| `log`, `√` | Logarit tự nhiên, căn bậc 2 | Hai phép toán quen thuộc — dùng để "nén" khoảng giá trị lớn (log) hoặc chuẩn hoá độ lớn (căn bậc 2). |

## Bản đồ tổng thể — đọc trước, quay lại tra cứu bất cứ lúc nào

Đây là "khung xương" của toàn bộ tài liệu. Mỗi khi đọc xong 1 khối, hãy quay lại nhìn bản đồ này để biết mình đang ở đâu:

```
PHẦN I — Nền tảng dữ liệu
  [1] Original Bipartite Graph

PHẦN II — Cơ chế học biểu diễn dùng CHUNG cho cả 2 nhánh
  [2] LightGCN Backbone  ◄── cơ chế lan truyền, dùng lại ở cả Nhánh A và Nhánh B

PHẦN III — Nhánh chính: dự đoán & độ chính xác (Branch A)
  [3] Branch A (áp dụng [2] một lần, trên đồ thị SẠCH)
  [4] Scoring (biến embedding thành điểm số)
  [5] BPR Loss (dạy mô hình xếp hạng đúng)
  [6] Popularity-Aware Negative Sampling (chọn "câu hỏi luyện tập" thông minh hơn cho [5])
  [7] Item Loss Equalization — ILE (công bằng hoá loss giữa item phổ biến/ít phổ biến)

PHẦN IV — Nhánh phụ: học biểu diễn bền vững hơn (Branch B)
  [8] Degree-Aware Graph Augmentation (tạo 2 bản đồ thị "nhiễu có chủ đích")
  [9] Branch B (áp dụng [2] hai lần, trên 2 bản đồ thị ở [8])
  [10] Contrastive Loss (ép 2 view của cùng 1 thực thể phải giống nhau)

PHẦN V — Gộp lại & Vận hành thật
  [11] L2 Regularization (chống overfitting)
  [12] Total Loss (gộp cả 4 loss: [5]+[6]→BPR, [7]→ILE, [10]→CL, [11]→L2)
  [13] Top-K Inference (suy luận thật — CHỈ dùng nhánh [3], bỏ hết Phần IV)
```

**Vì sao chia thành 2 nhánh?** Nhánh chính (Phần III) lo việc dự đoán chính xác — gần như y hệt LightGCN gốc. Nhánh phụ (Phần IV) chỉ tồn tại để "huấn luyện thêm" cho embedding tốt hơn, không bao giờ được dùng để đưa ra gợi ý cuối cùng. Đọc hết Phần III trước sẽ dễ hiểu vì sao Phần IV lại được thiết kế như vậy.

### Câu hỏi quan trọng cần trả lời TRƯỚC khi đọc chi tiết: Nhánh A và Nhánh B "kết hợp" ở đâu?

Nhìn bản đồ trên, dễ có cảm giác 2 nhánh chỉ "gặp nhau" đúng 1 chỗ — Khối 12 (Total Loss), nơi 2 con số loss được cộng lại. Từ đó một câu hỏi tự nhiên (và rất đáng hỏi): **vậy 2 nhánh có TRỘN EMBEDDING (representation) với nhau ở đâu không?**

**Trả lời — đây là điểm dễ hiểu lầm nhất của toàn bộ kiến trúc:** **KHÔNG.** Trong suốt lượt truyền xuôi (forward pass), embedding của Nhánh A (từ đồ thị sạch, Khối 3) và 2 bộ embedding của Nhánh B (từ 2 view augment, Khối 9) là **3 bộ số hoàn toàn độc lập** — không có bước concatenate/cộng/trộn nào giữa chúng ở bất kỳ khối nào. Nhánh B **không tạo ra bất cứ thứ gì được Nhánh A dùng lại.**

Vậy Nhánh B ảnh hưởng tới Nhánh A bằng cách nào? Bí mật nằm ở chính Khối 2: cả Nhánh A (Khối 3) lẫn Nhánh B (Khối 9) đều gọi **đúng cùng một hàm lan truyền**, xuất phát từ **đúng cùng một bảng embedding ban đầu `E^(0)`** — đây là **MỘT bảng tham số học được duy nhất** (trong code là `self.embedding.weight` của model, dùng chung, không phải 2 bản sao riêng cho 2 nhánh). Khi Khối 12 gọi `.backward()` trên Total Loss (đã cộng cả BPR+ILE từ Nhánh A và Contrastive Loss từ Nhánh B), cơ chế lan truyền ngược tự động **cộng dồn gradient từ CẢ HAI nhánh vào đúng cùng bảng `E^(0)` đó**, rồi bước cập nhật (`optimizer.step()`) chỉnh sửa nó **một lần duy nhất** dựa trên tổng gradient đã cộng dồn ấy.

Nói ngắn gọn: **2 nhánh không kết hợp ở cấp độ embedding, mà kết hợp ở cấp độ gradient, trên đúng MỘT bộ tham số học được dùng chung.** Đây chính xác là lý do vì sao sau khi huấn luyện xong, ta có thể bỏ hẳn Nhánh B (Khối 8, 9, 10) và chỉ cần Nhánh A (Khối 3) để suy luận (Khối 13): mọi "bài học" mà Nhánh B truyền đạt trong lúc train đã được ghi lại vĩnh viễn vào `E^(0)` rồi — Nhánh B không cần tồn tại nữa để `E^(0)` giữ được điều đó.

| Cấp độ | 2 nhánh kết hợp ở đâu? | Cụ thể là gì? |
|---|---|---|
| **Representation (embedding)** | **Không bao giờ kết hợp** | Nhánh A và Nhánh B luôn cho ra các bộ embedding độc lập, phục vụ 2 mục đích khác nhau (Scoring vs. Contrastive Loss) — không có phép concat/cộng embedding nào giữa 2 nhánh. |
| **Loss** | Khối 12 (Total Loss) | Cộng có trọng số 4 con số loss (`L_BPR, L_ILE, L_CL, L_reg`) thành 1 số duy nhất. |
| **Gradient / Tham số học được** | Ngay sau Khối 12, lúc `.backward()` | Gradient từ Nhánh A (qua BPR+ILE) **và** gradient từ Nhánh B (qua CL) cùng cộng dồn vào đúng **một** bảng `E^(0)` dùng chung — đây mới là điểm 2 nhánh thực sự "ảnh hưởng lẫn nhau". |

Ghi nhớ điều này khi đọc Phần III và Phần IV: các khối ở Phần IV không hề "nuôi" Phần III bằng dữ liệu hay embedding nào cả trong forward pass — chúng chỉ cùng nhau chỉnh sửa MỘT bảng tham số dùng chung (`E^(0)`) mà Phần III sẽ dùng lại ở vòng lặp huấn luyện tiếp theo, và ở bước suy luận cuối cùng.

---
---

# PHẦN I — NỀN TẢNG DỮ LIỆU

## KHỐI 1 — Original Bipartite Graph
*(Tương ứng Khối ① trên sơ đồ gốc)*

### 1. Khối này làm gì?

Biến dữ liệu thô (ai đã xem/mua/đánh giá gì) thành một **đồ thị** — cấu trúc mà mọi kỹ thuật ở các Phần sau đều cần để hoạt động. Không có bước này, không có gì để các khối sau xử lý.

### 2. Ý tưởng trực quan

Netflix ghi lại: "user A đã xem phim 1, phim 2"; "user B đã xem phim 2, phim 3". Thay vì lưu dưới dạng bảng, ta vẽ thành mạng lưới: mỗi user là một chấm, mỗi phim là một chấm, nối dây giữa user với phim họ đã xem. Nhìn vào mạng lưới: "user A và B cùng thích phim 2 → có thể A cũng thích phim 3 mà B đã xem" — đây là trực giác của toàn bộ hệ gợi ý dựa trên đồ thị. *(kiến thức nền)*

### 3. Input

Ma trận nhị phân user-item (1 = đã tương tác, rating≥4; 0 = chưa quan sát). Chỉ dùng phần dữ liệu thuộc tập **TRAIN**.

**Vì sao lại "rating≥4"?** Đây là quy ước tiền xử lý riêng của dataset cụ thể trong đồ án (dữ liệu gốc có thang điểm rating, ví dụ 1-5 sao) — chọn ngưỡng 4 để coi "rating cao" là tín hiệu "thực sự thích", biến **phản hồi tường minh** (explicit feedback — rating có thang điểm) thành **phản hồi ngầm định nhị phân** (implicit feedback — chỉ còn 0/1). *(kiến thức nền)* Nếu dữ liệu gốc chỉ có lượt xem/click (không có rating, như log xem video), bước này không cần thiết — mọi tương tác quan sát được mặc nhiên là 1, không có ngưỡng nào để chọn. Toàn bộ mô hình từ Khối 2 trở đi chỉ nhìn thấy ma trận 0/1 này, không quan tâm giá trị rating gốc là bao nhiêu.

**Ví dụ cụ thể** (bảng thô — đúng bộ dữ liệu đồ chơi dùng xuyên suốt tài liệu, 4 user × 3 item):

|      | i1 (head) | i2 (middle) | i3 (tail) |
|------|-----------|-------------|-----------|
| u1   | 1         | 1           | 0         |
| u2   | 1         | 0           | 0         |
| u3   | 1         | 1           | 0         |
| u4   | 1         | 0           | 1         |

### 4. Output

Đồ thị hai phía `𝒢 = (U ∪ I, E_train)` — về mặt cài đặt, đây là một **danh sách cạnh** (edge list).

**Ví dụ cụ thể** (chính xác từ bảng input ở trên):
```
E_train = { (u1,i1), (u2,i1), (u3,i1), (u4,i1), (u1,i2), (u3,i2), (u4,i3) }
```
Từ danh sách cạnh này suy ra ngay degree: `deg(i1)=4` (xuất hiện trong 4 cạnh → nhóm *head*), `deg(i2)=2` (nhóm *middle*), `deg(i3)=1` (nhóm *tail*). Ba con số degree này sẽ quay lại ở **Khối 6, 7, 8** — không tính lại, dùng đúng 3 số này.

Output này sẽ được dùng làm điểm khởi đầu cho **Khối 3 (Branch A)** ở Phần III và **Khối 8 (Augmentation)** ở Phần IV — nhưng ta sẽ đọc cách dùng nó chi tiết sau, khi tới đúng khối đó.

Số liệu thật của toàn bộ dataset (không phải ví dụ nhỏ trên): **|U| = 6.034 user**, **|I| = 3.533 item**, **563.204 cạnh** train.

### 5. Công thức toán học

$$\mathcal{G} = (U \cup I,\ E_{train}), \quad E_{train} \subseteq U \times I$$

**Giải thích từng thành phần:**
- `𝒢` (chữ G viết hoa kiểu script): tên gọi ta đặt cho đồ thị đang xây dựng — chỉ là 1 ký hiệu, không có phép tính gì bên trong nó.
- `U`: tập hợp TẤT CẢ user (trong ví dụ của ta: `U={u1,u2,u3,u4}`).
- `I`: tập hợp TẤT CẢ item (trong ví dụ: `I={i1,i2,i3}`).
- `U ∪ I`: hợp 2 tập trên — gộp thành 1 tập "đỉnh" duy nhất của đồ thị (không phân biệt user/item về mặt tập hợp, dù bản chất chúng khác loại — đây là lý do đồ thị này gọi là "hai phía": các đỉnh có 2 LOẠI khác nhau).
- `E_train`: tập hợp các **cạnh** — mỗi phần tử là 1 cặp `(user, item)` đã tương tác thật (trong ví dụ: 7 cặp, xem mục 4).
- `⊆`: `E_train` chỉ là MỘT PHẦN (không nhất thiết toàn bộ) của tập bên phải.
- `U × I`: tích Descartes — tập hợp TẤT CẢ các cặp `(user, item)` CÓ THỂ CÓ, kể cả những cặp chưa từng tương tác (trong ví dụ: `4×3=12` cặp có thể có). `E_train ⊆ U×I` nói đúng 1 điều: "cạnh thật chỉ là một tập con nhỏ trong số mọi cặp có thể có" — ở ví dụ của ta là 7/12 cặp.

### 6. Ví dụ tính toán từng bước

```
u1 xem i1, i2
u2 xem i1
u3 xem i1, i2
u4 xem i1, i3
```
→ 7 đỉnh (4 user + 3 item), 7 cạnh. Đếm số cạnh chạm mỗi item ra đúng degree: i1 chạm 4 lần (head), i2 chạm 2 lần (middle), i3 chạm 1 lần (tail).

### 7. Hình minh hoạ ASCII

```
u1 ──┬── i1 (head, deg=4) ──┬── u2
     └── i2 (middle, deg=2)─┤
u3 ──┬── i1                 ├── u3
     └── i2                 │
u4 ──┬── i1                 └── u4
     └── i3 (tail, deg=1) ── u4
```

### 8. Vai trò trong toàn bộ mô hình

Là điểm khởi đầu **duy nhất** của toàn bộ pipeline.

### 9. Nếu bỏ khối này

Không thể bỏ — không có gì để các khối sau xử lý.

### 10. Điểm mới (Contribution)

**Không phải điểm mới** — biểu diễn dữ liệu dưới dạng đồ thị hai phía là kỹ thuật nền tảng dùng chung bởi mọi mô hình graph-based collaborative filtering.

> **Tiếp theo:** trước khi xem đồ thị này được "đi" theo 2 hướng nào, ta cần hiểu cơ chế lan truyền dùng chung cho cả 2 hướng đó — đây là Khối 2.

---
---

# PHẦN II — CƠ CHẾ HỌC BIỂU DIỄN DÙNG CHUNG

## KHỐI 2 — LightGCN Backbone
*(Tương ứng Khối ③ trên sơ đồ gốc — nhưng đọc TRƯỚC Branch A/B vì cả hai đều tái sử dụng chính xác cơ chế này)*

### 1. Khối này làm gì?

Đây là "bộ máy" học biểu diễn: nhận vào một đồ thị + một bảng embedding ban đầu (toàn số ngẫu nhiên, chưa có ý nghĩa gì), rồi cho các user/item "trao đổi thông tin" với hàng xóm của mình qua nhiều bước, để cuối cùng ra một embedding **có ý nghĩa**, phản ánh đúng cấu trúc quan hệ trong đồ thị. **Đây là cơ chế trung tâm, được tái sử dụng ở cả Nhánh A (1 lần) và Nhánh B (2 lần) — học kỹ khối này trước sẽ giúp các khối sau ngắn gọn hơn nhiều.**

### 2. Ý tưởng trực quan

"Bạn là trung bình cộng của những người bạn chơi cùng" *(kiến thức nền)*. Trên Netflix: nếu nhiều người có gu giống bạn đã thích một phim bạn chưa xem, khả năng cao bạn cũng thích nó. Backbone làm đúng việc này có hệ thống: mỗi "vòng" lan truyền, một user "hỏi ý kiến" các item mình đã tương tác; một item "hỏi ý kiến" các user đã tương tác với nó — lặp lại nhiều vòng để thông tin lan xa hơn (bạn của bạn của bạn).

### 3. Input

Một đồ thị bất kỳ (`edge_index`) + bảng embedding ban đầu `E^(0)` (khởi tạo ngẫu nhiên).

**Ví dụ cụ thể — `edge_index`** (dùng lại `E_train` ở Khối 1, đánh số u1=0,u2=1,u3=2,u4=3, i1=4,i2=5,i3=6; lưu **cả 2 chiều** vì đồ thị vô hướng):
```
edge_index = [[0,4, 1,4, 2,4, 3,4, 0,5, 2,5, 3,6],
              [4,0, 4,1, 4,2, 4,3, 5,0, 5,2, 6,3]]
              # hàng trên = điểm đầu cạnh, hàng dưới = điểm cuối cạnh
              # 7 cạnh gốc × 2 chiều = 14 cột
```

**Ví dụ cụ thể — `E^(0)`** (embedding ngẫu nhiên ban đầu — đúng bộ dữ liệu chung, rút gọn 2 chiều thay vì 64, cho 7 node u1,u2,u3,u4,i1,i2,i3):
```
E^(0) = [[ 0.10, -0.20],   ← u1
         [ 0.05,  0.15],   ← u2
         [-0.10,  0.05],   ← u3
         [ 0.20, -0.05],   ← u4
         [ 0.30,  0.10],   ← i1
         [-0.05, -0.15],   ← i2
         [ 0.15,  0.25]]   ← i3
```

**Lưu ý quan trọng:** khối này được thiết kế để dùng LẶP LẠI — Nhánh A sẽ đưa vào đồ thị gốc (Khối 1), còn Nhánh B (Phần IV) sẽ đưa vào 2 phiên bản đồ thị đã bị augment. Công thức bên dưới không đổi dù đồ thị đầu vào là gì.

### 4. Output

Bảng embedding cuối cùng `E`, tách thành `user_emb` (`[num_users, 64]`) và `item_emb` (`[num_items, 64]`).

**Ví dụ cụ thể** (tiếp ví dụ trên, sau khi lan truyền 1 lớp K=1 và lấy trung bình 2 lớp — tính tay đầy đủ ở mục 6 bên dưới):
```
user_emb = [[ 0.0906, -0.1198],   ← u1
            [ 0.10,    0.10  ],   ← u2
            [-0.00945, 0.0052],   ← u3
            [ 0.2061,  0.0811]]   ← u4

item_emb = [[ 0.1979,  0.0522],   ← i1 (head)
            [-0.025,  -0.1125],   ← i2 (middle)
            [ 0.1457,  0.1073]]   ← i3 (tail)
```
Đây chính là 2 ma trận sẽ được truyền tới **Khối 4 (Scoring)** — và toàn bộ các khối phía sau (5, 6, 7, 11, 12, 13) sẽ dùng lại đúng những con số này, không tính ví dụ riêng lẻ nào khác nữa.

### 5. Công thức toán học

**Bước 1 — Chuẩn hoá đối xứng:**
$$\tilde{A} = D^{-1/2} A D^{-1/2}$$

**Giải thích từng thành phần:**
- `A` (ma trận kề — adjacency matrix): 1 bảng số vuông kích thước `(số user + số item) × (số user + số item)`; tại vị trí hàng `x`, cột `y`, giá trị là `1` nếu có cạnh nối `x-y`, ngược lại là `0`. **Của riêng đồ thị hai phía** — chỉ có 1 ở vị trí `(user, item)` khi có cạnh nối; **không có** cạnh nào giữa 2 user với nhau hay giữa 2 item với nhau (đúng định nghĩa "bipartite" ở Khối 1).
- `D` (ma trận đường chéo degree): cùng kích thước với `A`, nhưng chỉ có số khác 0 trên đường chéo — tại vị trí `(x,x)` là `deg(x)` (degree của chính node đó); mọi vị trí khác đều là 0.
- `D^{-1/2}`: cùng là ma trận đường chéo, nhưng mỗi số trên đường chéo được thay bằng `1/\sqrt{\text{số đó}}` — ví dụ nếu `D` có `deg(i1)=4` tại một vị trí, thì `D^{-1/2}` có `1/\sqrt{4}=0.5` tại đúng vị trí đó.
- `D^{-1/2} A D^{-1/2}`: nhân 3 ma trận liên tiếp (từ trái sang phải) — kết quả là `Ã`, ma trận kề đã "chia tỷ lệ" theo degree ở cả 2 đầu của mỗi cạnh.
- `Ã` (A có dấu ngã, gọi là "ma trận kề đã chuẩn hoá"): kết quả cuối cùng của Bước 1, dùng ở Bước 2.

**Vì sao chia cho `D^{-1/2}`?** Nếu không chuẩn hoá, user/item có nhiều cạnh (degree cao) sẽ có tổng tín hiệu lan truyền lớn hơn hẳn, chỉ vì "nhiều hàng xóm hơn" chứ không phải vì quan hệ mạnh hơn. Chuẩn hoá giúp cân bằng lại, giống "trung bình có trọng số" thay vì "cộng dồn". *(kiến thức nền — nguyên lý GCN chuẩn)*

**Bước 2 — Lan truyền qua các lớp:**
$$E^{(k+1)} = \tilde{A} \cdot E^{(k)}$$

**Giải thích từng thành phần:**
- `E^(k)`: ma trận embedding của TẤT CẢ node, tại lớp lan truyền thứ `k` (xem Bảng ký hiệu ở đầu tài liệu — số mũ trong ngoặc là "lớp", không phải luỹ thừa).
- `E^(k+1)`: ma trận embedding ở lớp KẾ TIẾP, tính ra từ `E^(k)`.
- `Ã · E^(k)`: nhân ma trận `Ã` (Bước 1) với ma trận `E^(k)` — kết quả là `E^(k+1)`.

Mỗi lớp `k`, embedding mới của 1 node = tổng có trọng số embedding các hàng xóm ở lớp trước. Không có phép biến đổi phi tuyến — đặc trưng của LightGCN: đơn giản hoá tối đa so với GCN gốc.

**Cầu nối giữa công thức ma trận (trên) và công thức từng-node (mục 6 dưới đây):** phép nhân ma trận `Ã·E^(k)` chỉ là cách viết GỌN của việc, với MỖI node `x`, cộng embedding các hàng xóm của `x`, mỗi hàng xóm nhân với đúng hệ số `Ã_{x,y} = 1/(\sqrt{\deg(x)}\sqrt{\deg(y)})` nếu có cạnh `x-y` (và `Ã_{x,y}=0` nếu không có cạnh). Nói cách khác: dòng thứ `x` của ma trận `Ã` chính là "công thức trung bình có trọng số" áp dụng riêng cho node `x` — mục 6 chỉ đơn giản là viết tường minh 1 dòng đó ra thay vì viết cả ma trận.

**Trường hợp đặc biệt — node bị cô lập (degree=0):** nếu một node không còn cạnh nào (ví dụ sau khi bị dropout ở Khối 8), về mặt toán học `1/\sqrt{\deg}=1/\sqrt{0}` không xác định. Code xử lý bằng cách **đặt thẳng hệ số này = 0** cho node đó (`deg_inv_sqrt = 0` khi `deg=0`, thay vì to Infinity/NaN) — tương đương với "node cô lập không nhận và không gửi tín hiệu lan truyền ở lớp đó", nhưng **vẫn giữ nguyên `E^(0)` của chính nó** trong phép lấy trung bình ở Bước 3. Ví dụ cụ thể của trường hợp này ở Khối 9.

**Bước 3 — Lấy trung bình các lớp:**
$$E = \frac{1}{K+1} \sum_{k=0}^{K} E^{(k)}$$

**Giải thích từng thành phần:**
- `K` (chữ hoa, khác `k` thường): tổng số lớp lan truyền — 1 hyperparameter cấu hình sẵn trước khi train (ví dụ `K=1` trong ví dụ tính tay của tài liệu này để đơn giản; thực nghiệm thật thường dùng `K=2` đến `K=4`).
- `Σ_{k=0}^{K}`: cộng dồn `E^(k)` với `k` chạy từ `0` (lớp gốc, chưa lan truyền) đến `K` (lớp cuối cùng) — tổng cộng có `K+1` số hạng (vì tính cả lớp 0).
- `1/(K+1)`: chia cho đúng số lượng lớp đã cộng (`K+1` lớp), để ra TRUNG BÌNH chứ không phải tổng thô.
- `E` (không có số mũ): embedding CUỐI CÙNG, sau khi đã lấy trung bình — đây là kết quả trả về của cả Backbone, dùng cho mọi khối sau.

Mỗi lớp nắm bắt thông tin ở một "khoảng cách" khác nhau (lớp 0 = chính nó, lớp 1 = hàng xóm trực tiếp, lớp 2 = hàng xóm của hàng xóm...). Nếu chỉ dùng lớp cuối, thông tin dễ bị "làm mượt quá mức" (over-smoothing — các node ở xa dần giống hệt nhau). Lấy trung bình giữ cân bằng cả thông tin gần lẫn xa. *(kiến thức nền)*

**Đang tối ưu gì?** Bản thân Backbone không "tối ưu" trực tiếp (không có tham số học riêng ngoài `E^(0)`) — nó là một phép biến đổi có cấu trúc cố định; việc tối ưu thực sự diễn ra khi Total Loss (Khối 12) lan truyền ngược để chỉnh `E^(0)`.

### 6. Ví dụ tính toán từng bước

Dùng đúng bộ dữ liệu chung (K=1, tức 1 lớp lan truyền). Nhắc lại degree: `deg(u1)=2, deg(u2)=1, deg(u3)=2, deg(u4)=2, deg(i1)=4, deg(i2)=2, deg(i3)=1`.

**Lớp 1, cập nhật `e_u1`** (hàng xóm của u1 là i1, i2):
$$e_{u1}^{(1)} = \frac{1}{\sqrt{\deg(u1)}\sqrt{\deg(i1)}} e_{i1}^{(0)} + \frac{1}{\sqrt{\deg(u1)}\sqrt{\deg(i2)}} e_{i2}^{(0)} = \frac{1}{\sqrt{2}\sqrt{4}}[0.30,0.10] + \frac{1}{\sqrt{2}\sqrt{2}}[-0.05,-0.15]$$
$$= 0.3536\times[0.30,0.10] + 0.5\times[-0.05,-0.15] = [0.1061,0.0354] + [-0.025,-0.075] = [0.0811,-0.0396]$$

**Lớp 1, cập nhật `e_i1`** (hàng xóm của i1 là u1,u2,u3,u4):
$$e_{i1}^{(1)} = \sum_{u \in \{u1,u2,u3,u4\}} \frac{1}{\sqrt{\deg(i1)}\sqrt{\deg(u)}} e_u^{(0)}$$
$$= \frac{1}{\sqrt{4}\sqrt{2}}[0.10,-0.20] + \frac{1}{\sqrt{4}\sqrt{1}}[0.05,0.15] + \frac{1}{\sqrt{4}\sqrt{2}}[-0.10,0.05] + \frac{1}{\sqrt{4}\sqrt{2}}[0.20,-0.05]$$
$$= 0.3536[0.10,-0.20] + 0.5[0.05,0.15] + 0.3536[-0.10,0.05] + 0.3536[0.20,-0.05]$$
$$= [0.0354,-0.0707]+[0.025,0.075]+[-0.0354,0.0177]+[0.0707,-0.0177] = [0.0957,0.0043]$$

(Các node khác — `e_u2^(1), e_u3^(1), e_u4^(1), e_i2^(1), e_i3^(1)` — tính hoàn toàn tương tự, cùng công thức, chỉ đổi tập hàng xóm.)

**Trung bình 2 lớp (K=1)** cho u1 và i1:
$$e_{u1} = \frac{1}{2}\left(e_{u1}^{(0)} + e_{u1}^{(1)}\right) = \frac{1}{2}\big([0.10,-0.20]+[0.0811,-0.0396]\big) = [0.0906,-0.1198]$$
$$e_{i1} = \frac{1}{2}\left(e_{i1}^{(0)} + e_{i1}^{(1)}\right) = \frac{1}{2}\big([0.30,0.10]+[0.0957,0.0043]\big) = [0.1979,0.0522]$$

Đây chính xác là 2 dòng `u1` và `i1` đã xuất hiện ở bảng `user_emb`/`item_emb` trong mục 4 — không phải số minh hoạ riêng, mà là kết quả tính tay thật của đúng ví dụ đó. Các dòng còn lại (`u2,u3,u4,i2,i3`) được tính bằng đúng quy trình 2 bước này.

### 7. Hình minh hoạ ASCII

```
E^(0) ──lan truyền──► E^(1) ──lan truyền──► ... ──► E^(K)
  │                     │                              │
  └─────────────────────┴──────────── trung bình ──────┘
                              │
                              ▼
                    E (embedding cuối cùng)
```

### 8. Vai trò trong toàn bộ mô hình

Được **dùng chung** — Nhánh A gọi hàm này 1 lần (Khối 3), Nhánh B gọi hàm này 2 lần (Khối 9). Cùng một công thức, chỉ khác đồ thị đầu vào.

### 9. Nếu bỏ khối này

Mô hình mất khả năng tận dụng cấu trúc đồ thị — suy biến về Matrix Factorization thuần (embedding cố định, không "học hỏi" từ hàng xóm), giống baseline BPR-MF trong đồ án.

### 10. Điểm mới (Contribution)

**Hoàn toàn không phải điểm mới.** Đây chính xác là LightGCN nguyên bản (He et al., 2020), giữ nguyên không đổi.

> **Tiếp theo:** giờ ta đã hiểu cơ chế lan truyền, hãy xem Nhánh A dùng nó như thế nào để tạo ra embedding phục vụ dự đoán.

---
---

# PHẦN III — NHÁNH CHÍNH: DỰ ĐOÁN & ĐỘ CHÍNH XÁC (Branch A)

*Toàn bộ Phần III trả lời câu hỏi: "Làm sao mô hình dự đoán được user thích item nào?" — đây là con đường gần như y hệt LightGCN gốc.*

## KHỐI 3 — Branch A (Nhánh chính)
*(Tương ứng "Branch A — Main path" trên sơ đồ gốc)*

### 1. Khối này làm gì?

Áp dụng Backbone (Khối 2) **đúng 1 lần**, trên **đồ thị gốc, không bị can thiệp gì** (không dropout). Đây là "đường đi an toàn": mọi điểm số dự đoán và mọi gợi ý thật cuối cùng đều xuất phát từ đây.

### 2. Ý tưởng trực quan

Một công ty có sản phẩm chính bán chạy (LightGCN) — thay vì mạo hiểm thay đổi sản phẩm chính để thử nghiệm ý tưởng mới, công ty giữ nguyên dây chuyền sản xuất chính (Nhánh A), và chạy phòng R&D riêng (Nhánh B, Phần IV) để phát triển cải tiến — chỉ áp dụng kết quả R&D vào việc huấn luyện, không đụng vào sản phẩm đang chạy thật. *(kiến thức nền, ví von)*

### 3. Input

Đồ thị gốc từ Khối 1 — **không dropout** (`aug_main=False` trong mọi cấu hình đã báo cáo). Chính là `edge_index` ví dụ ở Khối 2, **không sửa đổi gì thêm**.

### 4. Output

`user_emb`, `item_emb` — chính xác là 2 ma trận ví dụ đã tính ở Khối 2 (mục 4). Dùng cho **Khối 4 (Scoring)** ngay sau đây, và sẽ được dùng lại **nguyên vẹn, không tính lại** ở **Khối 13 (Inference)** cuối tài liệu.

### 5. Công thức toán học

Dùng đúng công thức của Khối 2 — không có gì khác biệt ngoài việc đồ thị đầu vào không bị dropout.

### 6. Ví dụ tính toán từng bước

Xem lại ví dụ số ở Khối 2 — Branch A chỉ là "chạy Backbone 1 lần trên đồ thị nguyên bản".

### 7. Hình minh hoạ ASCII

```
Đồ thị gốc (không dropout)
        │
        ▼
  Khối 2: LightGCN Backbone  (gọi 1 lần)
        │
        ▼
  user_emb, item_emb ──► Khối 4 (Scoring) ──► Khối 5,7 (BPR, ILE)
        │
        └───────────────► Khối 13 (Inference, ở cuối tài liệu)
```

### 8. Vai trò trong toàn bộ mô hình

Nhánh **quyết định kết quả thật**.

### 9. Nếu bỏ khối này

Không thể bỏ — đây chính là "mô hình gợi ý" thật sự.

### 10. Điểm mới (Contribution)

Cơ chế lan truyền là kế thừa (xem Khối 2). Nhưng **quyết định kiến trúc** "tách riêng 1 nhánh sạch hoàn toàn, không chạm augmentation" — để đảm bảo augmentation ở Nhánh B không ảnh hưởng accuracy — là lựa chọn thiết kế của phương pháp đề xuất.

> **Tiếp theo:** ta đã có `user_emb`, `item_emb`. Làm sao biến 2 vector này thành 1 con số dự đoán? Đó là Khối 4.

## KHỐI 4 — Scoring
*(Tương ứng Khối ④ trên sơ đồ gốc)*

### 1. Khối này làm gì?

Biến 2 vector embedding (1 user, 1 item) thành **một con số duy nhất**: mô hình dự đoán user thích item này đến mức nào.

### 2. Ý tưởng trực quan

Tích vô hướng đo "độ tương đồng có định hướng": nếu embedding của user A và phim X đều nghiêng mạnh về "hướng hành động, kịch tính" trong không gian 64 chiều, dot product của chúng sẽ cao. *(kiến thức nền)*

### 3. Input

`e_u`, `e_i` — 2 vector, lấy từ đúng 2 dòng tương ứng trong `user_emb`, `item_emb` (output của Khối 3).

**Ví dụ cụ thể:** lấy dòng `u1` và `i1` từ output thật đã tính ở Khối 2/3: `e_u1 = [0.0906, -0.1198]`, `e_i1 = [0.1979, 0.0522]` — đây là cặp **có tương tác thật** trong `E_train` (positive). Để có ví dụ negative, lấy thêm `e_i3 = [0.1457, 0.1073]` (u1 **chưa từng** tương tác với i3).

### 4. Output

Số thực `s(u,i)`.

**Positive — `s(u1,i1)`:**
$$s(u1,i1) = 0.0906\times0.1979 + (-0.1198)\times0.0522 = 0.01793 - 0.00625 = 0.01168 \approx 0.0117$$

**Negative — `s(u1,i3)`:**
$$s(u1,i3) = 0.0906\times0.1457 + (-0.1198)\times0.1073 = 0.01320 - 0.01285 = 0.00035 \approx 0.0003$$

Cả 2 điểm đều gần 0 vì embedding ví dụ mới ở vòng lan truyền đầu tiên, chưa được huấn luyện — nhưng `s(u1,i1) > s(u1,i3)`, đúng hướng mong muốn (item đã tương tác được điểm cao hơn). 2 số `s⁺=0.0117, s⁻=0.0003` này truyền thẳng vào **Khối 5 (BPR)** và **Khối 7 (ILE)**.

### 5. Công thức toán học

$$s(u,i) = \langle e_u, e_i \rangle = \sum_{d=1}^{64} e_u[d] \cdot e_i[d]$$

**Giải thích từng thành phần:**
- `s(u,i)`: điểm số dự đoán cho cặp `(user u, item i)` — kết quả cuối cùng của công thức, 1 số thực duy nhất.
- `⟨e_u,e_i⟩`: ký hiệu toán học nghĩa là "tích vô hướng (dot product) của 2 vector `e_u` và `e_i`" — chỉ là TÊN GỌI của phép toán được viết tường minh ở vế ngay sau dấu `=`.
- `Σ_{d=1}^{64}`: cộng dồn, với `d` chạy từ `1` đến `64` (64 = số chiều của embedding thật; ví dụ trong tài liệu rút gọn còn 2 chiều).
- `e_u[d]`: giá trị số thực tại toạ độ (chiều) thứ `d` của vector `e_u` (ví dụ `e_u[1]` là số đầu tiên trong vector).
- `e_u[d] · e_i[d]`: nhân 2 số thực tại CÙNG toạ độ `d` của 2 vector.

Nhân từng cặp toạ độ tương ứng rồi cộng lại. Không có biến đổi hay activation nào thêm. (Ví dụ trên chỉ dùng 2 chiều thay vì 64, nhưng phép tính giống hệt.)

### 6. Ví dụ tính toán từng bước

Đã tính đầy đủ ở mục 4 — nhắc lại từng bước cho `s(u1,i1)`:
1. Nhân toạ độ thứ nhất: `0.0906 × 0.1979 = 0.01793`
2. Nhân toạ độ thứ hai: `(-0.1198) × 0.0522 = -0.00625`
3. Cộng lại: `0.01793 + (-0.00625) = 0.01168 ≈ 0.0117`

### 7. Hình minh hoạ ASCII

```
e_u1 = [ 0.0906, -0.1198]
e_i1 = [ 0.1979,  0.0522]
          ×          ×
       0.01793  + (-0.00625)  = 0.0117 = s(u1,i1)  ← positive, cao hơn
e_i3 = [ 0.1457,  0.1073]
       0.01320  + (-0.01285)  = 0.0003 = s(u1,i3)  ← negative, thấp hơn
```

### 8. Vai trò trong toàn bộ mô hình

Cầu nối giữa học biểu diễn (Khối 2, 3) và học tối ưu (Khối 5, 7) — đồng thời là công thức dùng lại ở Khối 13.

### 9. Nếu bỏ khối này

Không có cách nào biến embedding thành con số so sánh được — không thể xếp hạng.

### 10. Điểm mới (Contribution)

**Không phải điểm mới** — dot product scoring là kỹ thuật chuẩn của Matrix Factorization và LightGCN gốc.

> **Tiếp theo:** ta có điểm số rồi — nhưng điểm số một mình chưa "dạy" được gì cho mô hình. Cần so sánh điểm của item đã thích với điểm của item CHƯA thích. Đó là BPR Loss.

## KHỐI 5 — BPR Loss
*(Tương ứng Khối ⑤① trên sơ đồ gốc)*

### 1. Khối này làm gì?

Dạy mô hình quy tắc: **điểm số của item user đã tương tác phải cao hơn điểm số của item họ chưa tương tác**. Đây là loss nền tảng, luôn hiện diện.

### 2. Ý tưởng trực quan

Thay vì cố "đoán chính xác" một con số rating, BPR chỉ học **thứ tự tương đối**: "cái này tôi thích hơn cái kia". Cách học này khớp trực tiếp mục tiêu thật của hệ gợi ý — chỉ cần xếp hạng đúng để đưa top-K lên đầu. *(kiến thức nền)*

### 3. Input

`s⁺_ui` (điểm positive), `s⁻_uj` (điểm negative) — 2 số thực, từ Khối 4. Negative `j` được chọn theo cơ chế ở **Khối 6** (đọc ngay sau đây).

**Ví dụ cụ thể:** dùng đúng 2 số vừa tính ở Khối 4: `s⁺_u1,i1 = 0.0117` (cặp `(u1,i1)` đã tương tác thật), `s⁻_u1,i3 = 0.0003` (giả sử ở lượt lấy mẫu này, negative bốc trúng `i3`).

### 4. Output

Một số thực (loss) — **ví dụ cụ thể: `ℓ ≈ 0.6874`** (tính chi tiết ở mục 6 bên dưới) — cộng vào Total Loss (Khối 12).

### 5. Công thức toán học

$$\mathcal{L}_{BPR} = -\frac{1}{|B|} \sum_{(u,i,j) \in B} \log \sigma(s^+_{ui} - s^-_{uj})$$

**Giải thích từng thành phần:**
- `B`: batch — tập hợp các bộ ba `(u,i,j)` được lấy mẫu ngẫu nhiên trong 1 lần cập nhật (ví dụ thật: 4.096 bộ ba/batch; ví dụ tính tay trong tài liệu này chỉ dùng 1 bộ ba).
- `|B|`: kích thước batch — SỐ LƯỢNG bộ ba có trong `B` (ví dụ 4.096), dùng để chia lấy trung bình.
- `Σ_{(u,i,j)∈B}`: cộng dồn biểu thức phía sau, lặp lại cho MỖI bộ ba `(u,i,j)` có trong batch `B`.
- `s⁺_ui, s⁻_uj`: điểm dự đoán positive/negative (đã tính ở Khối 4).
- `σ(x)=1/(1+e^{-x})`: hàm sigmoid, ép số thực bất kỳ về khoảng `(0,1)` — hiểu như "xác suất positive tốt hơn negative".
- `log`: logarit tự nhiên — vì `σ(x)∈(0,1)` nên `log σ(x)` LUÔN ≤ 0.
- Dấu `-` ở đầu công thức (trước phân số `1/|B|`): vì `log σ(...)` luôn ≤ 0 (giải thích ở trên), nếu không đổi dấu thì loss sẽ luôn ≤ 0, ngược với quy ước "loss càng nhỏ (dương, gần 0) càng tốt" — dấu `-` biến nó thành số ≥ 0.
- `1/|B|`: chia cho kích thước batch để ra TRUNG BÌNH loss trên mỗi bộ ba, thay vì tổng thô (giá trị tổng sẽ phụ thuộc vào batch to hay nhỏ, không so sánh được giữa các cấu hình khác nhau).

Nếu `s⁺-s⁻` lớn → `σ` gần 1 → `log σ` gần 0 → loss nhỏ (tốt). Nếu dự đoán sai → loss lớn (bị phạt nặng). **Đang tối ưu gì?** Tối đa hoá xác suất xếp hạng đúng positive cao hơn negative, với mọi bộ ba trong dữ liệu.

### 6. Ví dụ tính toán từng bước

`s⁺=0.0117`, `s⁻=0.0003`:
$$s^+-s^- = 0.0114, \quad \sigma(0.0114) \approx 0.5028, \quad \ell = -\log(0.5028) \approx 0.6874$$

Loss vẫn còn gần `log(2)≈0.693` (mức loss của một dự đoán "đoán mò 50/50") — hợp lý, vì embedding ở ví dụ này mới qua đúng 1 lớp lan truyền, gần như chưa học được gì. Loss thật sẽ giảm dần về gần 0 sau hàng nghìn bước cập nhật.

**Vì sao 0.693 đúng là "đoán mò 50/50"?** Nếu mô hình hoàn toàn chưa học được gì, `s⁺≈s⁻` (không phân biệt được positive/negative) → `s⁺-s⁻≈0` → `σ(0) = 1/(1+e^0) = 1/2 = 0.5` → `ℓ = -\log(0.5) = \log(2) ≈ 0.693`. Đây chính xác là điểm khởi đầu của mọi mô hình BPR mới khởi tạo — loss ví dụ ở trên (0.6874) rất gần 0.693, xác nhận đúng embedding "gần như ngẫu nhiên, chưa học gì" mà ta đang minh hoạ.

### 7. Hình minh hoạ ASCII

```
s⁺=0.0117, s⁻=0.0003 → s⁺-s⁻=0.0114 → σ(0.0114)≈0.5028 → -log(0.5028)≈0.6874
```

### 8. Vai trò trong toàn bộ mô hình

Loss **nền tảng**, luôn có mặt. Mọi thành phần khác (Khối 7, 10) được **cộng thêm**, không thay thế.

### 9. Nếu bỏ khối này

Mất tín hiệu học chính — embedding không học được gì có ý nghĩa.

### 10. Điểm mới (Contribution)

**Không phải điểm mới.** BPR (Rendle et al., 2009) là loss chuẩn, giữ nguyên để so sánh công bằng.

> **Tiếp theo:** BPR Loss cần một item "negative" — vậy item đó được CHỌN như thế nào? Đây chính là cải tiến đầu tiên của phương pháp đề xuất.

## KHỐI 6 — Popularity-Aware Negative Sampling
*(Tương ứng Khối ⑥ trên sơ đồ gốc)*

### 1. Khối này làm gì?

Thay vì chọn item "chưa tương tác" ngẫu nhiên đều để làm negative (Khối 5), khối này **ưu tiên chọn item phổ biến** làm negative với xác suất cao hơn.

### 2. Ý tưởng trực quan

Một bài kiểm tra tốt nên có câu hỏi thử thách đúng ranh giới kiến thức, không quá dễ. Nếu chọn 1 item hoàn toàn xa lạ làm "user A không thích", điều đó không dạy được gì nhiều — user A **chưa từng biết đến item đó**. Nhưng nếu chọn 1 item rất phổ biến (mà hầu như ai cũng từng thấy) làm negative, và user A thực sự không tương tác — đây là tín hiệu rất mạnh: "user A đã có cơ hội thấy item này nhưng chọn không quan tâm". *(kiến thức nền + suy luận về ý nghĩa "exposure")*

### 3. Input

`item_degree` (bảng độ phổ biến từng item), hệ số `β` (0 hoặc 0.5 trong thực nghiệm).

**Ví dụ cụ thể** (đúng 3 item và degree của bộ dữ liệu chung, tính từ Khối 1):

| Item | degree | nhóm |
|---|---|---|
| i1 | 4 | head |
| i2 | 2 | middle |
| i3 | 1 | tail |

### 4. Output

Một item cụ thể được **bốc thăm** làm negative `j⁻`, theo đúng phân phối xác suất tính ở mục 5/6 (`P(i1)=45.3%, P(i2)=32.0%, P(i3)=22.7%` với β=0.5) — quay ngược trở lại nuôi Khối 4 (Scoring) và Khối 5 (BPR).

**Nhưng bốc thăm này có loại trừ item user ĐÃ tương tác không?** Có — bằng cơ chế **rejection sampling** (lấy trực tiếp từ code `src/neg_sampling.py`, không phải suy đoán): bước bốc thăm ở trên **không** biết trước ai đã tương tác gì — nó bốc trên đúng phân phối `P(i1)=45.3%, P(i2)=32.0%, P(i3)=22.7%` áp dụng chung cho MỌI user. Sau đó, code kiểm tra: nếu item vừa bốc trúng nằm trong danh sách item user đó **đã** tương tác → **huỷ kết quả, bốc lại** — lặp lại tối đa 20 lần cho tới khi ra được 1 item hợp lệ (chưa tương tác).

**Hệ quả cụ thể cho 4 user trong ví dụ của ta** (dựa theo `E_train` ở Khối 1):
| user | đã thích | còn lại để làm negative | có bị ép buộc không? |
|---|---|---|---|
| u1 | {i1, i2} | chỉ {i3} | **Ép buộc 100% ra i3** — không phải "giả sử", mà là kết quả chắc chắn vì chỉ còn 1 lựa chọn |
| u2 | {i1} | {i2, i3} | Có chọn thật giữa 2 item — xem tính lại bên dưới |
| u3 | {i1, i2} | chỉ {i3} | **Ép buộc 100% ra i3**, giống hệt `u1` |
| u4 | {i1, i3} | chỉ {i2} | **Ép buộc 100% ra i2** |

Với `u2` (2 lựa chọn), rejection sampling tương đương về mặt toán học với việc **chuẩn hoá lại phân phối gốc chỉ trên 2 item còn hợp lệ**: dùng đúng trọng số thô đã tính ở mục 6 (`w_{i2}=1.414, w_{i3}=1.0`, bỏ `w_{i1}` vì `i1` không hợp lệ với `u2`):
$$P(i2\,|\,u2) = \frac{1.414}{1.414+1.0} \approx 58.6\%, \qquad P(i3\,|\,u2) = \frac{1.0}{1.414+1.0}\approx 41.4\%$$
Ở ví dụ tại Khối 4/5/7, ta giả định lần bốc của `u2` rơi vào nhánh xác suất thấp hơn (`i3`, 41.4%) — đây là kết quả **hợp lệ, có thể xảy ra thật**, chỉ không phải kết quả có xác suất cao nhất.

**Điểm quan trọng rút ra:** với catalog chỉ 3 item, popularity-aware sampling chỉ thực sự "phát huy tác dụng" (tức là còn quyền chọn) cho những user chưa thích gần hết catalog (ở đây là `u2`). Với dataset thật (3.533 item, mỗi user chỉ thích vài chục), tình huống "bị ép buộc" như `u1`/`u3`/`u4` gần như không xảy ra — hầu như user nào cũng có hàng nghìn lựa chọn hợp lệ, và lúc đó trọng số `deg_i^β` mới thực sự chi phối kết quả trên diện rộng.

### 5. Công thức toán học

$$P(neg=i) \propto \deg_i^{\ \beta}$$

**Giải thích từng thành phần:**
- `P(neg=i)`: xác suất mà item được CHỌN làm negative chính là item `i` (đây là 1 số trong khoảng `(0,1)`, tổng cộng của mọi item phải bằng 1).
- `deg_i`: degree của item `i` (đã tính ở Khối 1).
- `β` (beta): hyperparameter cấu hình sẵn, kiểm soát "mức độ thiên vị item phổ biến" (`β=0` hoặc `β=0.5` trong thực nghiệm của đồ án).
- `deg_i^β`: `deg_i` (một số nguyên) được nâng luỹ thừa `β` (một số thực) — ví dụ `4^{0.5} = \sqrt{4} = 2`.
- `∝` (tỷ lệ thuận — xem Bảng ký hiệu toán học ở đầu tài liệu): vế trái KHÔNG bằng trực tiếp vế phải — đây chỉ là **trọng số thô (raw weight)**, chưa phải xác suất hợp lệ (chưa chắc tổng bằng 1). Muốn có xác suất thật, phải **chuẩn hoá**: chia trọng số của mỗi item cho TỔNG trọng số của mọi item (xem phép tính cụ thể ở mục 6 — bước "tổng" và phép chia ngay sau).

`β=0`: `deg^0=1` mọi item → lấy mẫu đều. `β>0`: item degree càng cao, trọng số càng lớn theo hàm mũ.

### 6. Ví dụ tính toán từng bước

3 item: `deg(i1)=4, deg(i2)=2, deg(i3)=1`, `β=0.5`:
$$w_{i1}=4^{0.5}=2.0,\quad w_{i2}=2^{0.5}\approx1.414,\quad w_{i3}=1^{0.5}=1.0$$
$$\text{tổng} = 2.0+1.414+1.0 = 4.414$$
$$P(i1)=\frac{2.0}{4.414}\approx45.3\%,\quad P(i2)=\frac{1.414}{4.414}\approx32.0\%,\quad P(i3)=\frac{1.0}{4.414}\approx22.7\%$$

So sánh với `β=0` (lấy mẫu đều): mỗi item đều có `P=33.3%` — với `β=0.5`, item head (`i1`) được ưu tiên chọn làm negative nhiều hơn hẳn (45.3% so với 33.3%), còn item tail (`i3`) bị chọn ít hơn (22.7% so với 33.3%).

### 7. Hình minh hoạ ASCII

```
β=0 (uniform):        β=0.5 (popularity-biased):
i1 ▓▓▓▓▓▓ 33.3%        i1 ▓▓▓▓▓▓▓▓▓ 45.3%
i2 ▓▓▓▓▓▓ 33.3%        i2 ▓▓▓▓▓▓ 32.0%
i3 ▓▓▓▓▓▓ 33.3%        i3 ▓▓▓▓ 22.7%
```

### 8. Vai trò trong toàn bộ mô hình

Diễn ra **trước** Khối 4 trong trình tự thực thi thật (chọn negative trước khi tính điểm) — nhưng ta học sau Khối 5 vì cần hiểu "negative dùng để làm gì" trước khi hiểu "sao phải chọn negative thông minh hơn".

### 9. Nếu bỏ khối này (giữ β=0)

Quay về hành vi lấy mẫu gốc của BPR/LightGCN.

### 10. Điểm mới (Contribution)

**Có, đây là điểm mới.**

> **Tiếp theo:** BPR Loss (Khối 5) chỉ nhìn vào loss TRUNG BÌNH CHUNG — điều này khiến mô hình dễ học tốt item phổ biến hơn hẳn item ít phổ biến. Khối 7 sửa đúng vấn đề này.

## KHỐI 7 — Item Loss Equalization (ILE)
*(Tương ứng Khối ⑤② trên sơ đồ gốc)*

### 1. Khối này làm gì?

Thay vì chỉ nhìn loss BPR trung bình chung, ILE tách riêng loss trung bình của nhóm item phổ biến (head) và ít phổ biến (tail), rồi **phạt** nếu 2 giá trị chênh lệch quá nhiều.

### 2. Ý tưởng trực quan

Một lớp học có điểm trung bình 7/10 — nghe ổn. Nhưng nếu nhóm giỏi toàn 9-10, nhóm yếu toàn 3-4, điểm trung bình 7 đã **che giấu vấn đề thật**. ILE giống việc thầy giáo nhìn riêng điểm trung bình nhóm giỏi và nhóm yếu, rồi có biện pháp nếu khoảng cách quá lớn. Ở đây "nhóm giỏi" là item phổ biến (mô hình dễ học vì nhiều dữ liệu), "nhóm yếu" là item long-tail. *(kiến thức nền, ví von)*

### 3. Input

`s⁺_ui, s⁻_uj` của **cả batch** (không phải 1 cặp, mà hàng nghìn cặp — từ Khối 4, cùng dữ liệu với Khối 5), và `item_group` (nhãn head/middle/tail của positive item, xác định từ Khối 1).

**Ví dụ cụ thể** (dùng đúng `user_emb`/`item_emb` đã tính ở Khối 2, lấy toàn bộ 4 cạnh train tới `i1` làm batch — mỗi cạnh tự bốc 1 negative theo đúng cơ chế rejection sampling ở Khối 6, KHÔNG được tự chọn tuỳ ý — để có đủ cả head lẫn tail; batch chỉ có 3 item nên 1 interaction tail là đủ minh hoạ công thức):

| user | positive | nhóm | negative (theo Khối 6) | `s⁺` | `s⁻` | loss BPR riêng `ℓ=-\log\sigma(s^+-s^-)` |
|---|---|---|---|---|---|---|
| u1 | i1 | head | i3 *(ép buộc — u1 đã thích cả i1,i2)* | 0.0117 | 0.0003 | 0.6874 |
| u2 | i1 | head | i3 *(có chọn, rơi vào nhánh 41.4%)* | 0.0250 | 0.0253 | 0.6934 |
| u3 | i1 | head | i3 *(ép buộc — u3 đã thích cả i1,i2)* | -0.0016 | -0.0008 | 0.6934 |
| u4 | i3 | tail | i2 *(ép buộc — u4 đã thích cả i1,i3)* | 0.0387 | -0.0143 | 0.6675 |

(3 dòng đầu dùng `i1` — item **head** — làm positive, vì cả `u1,u2,u3` đều có cạnh tới `i1`; dòng cuối dùng cạnh `(u4,i3)` — `i3` là item **tail** duy nhất trong dữ liệu nên chỉ có 1 interaction tail khả dụng. Cách tính từng `s` giống hệt Khối 4, chỉ đổi cặp user/item. **Chú ý:** dòng `u1` ở đây dùng đúng negative `i3` — giống hệt ví dụ ở Khối 4/5, vì đó chính xác là cùng 1 interaction `(u1,i1,i3)`, không phải 2 ví dụ độc lập.)

### 4. Output

Số thực (penalty) — **ví dụ cụ thể: `L_ILE ≈ 0.00057`** (tính từ đúng bảng trên, xem mục 6) — cộng vào Total Loss (Khối 12) với trọng số `λ_ILE`.

### 5. Công thức toán học

**Bước 1 — Loss trung bình mỗi nhóm:**
$$\bar{\ell}_g = \text{mean}_{i \in g}\left[-\log \sigma(s^+_{ui} - s^-_{uj})\right]$$

**Giải thích từng thành phần:**
- `g`: tên 1 nhóm popularity — chỉ nhận 1 trong 2 giá trị `head` hoặc `tail` ở công thức này (xem lý do "middle" không xuất hiện ngay bên dưới).
- `ℓ̄_g` (gạch ngang trên đầu `ℓ` — xem Bảng ký hiệu toán học): nghĩa là "loss TRUNG BÌNH của nhóm `g`" — 1 số thực duy nhất đại diện cho cả nhóm, không phải 1 danh sách.
- `mean_{i∈g}[...]`: lấy trung bình biểu thức trong ngoặc vuông, tính riêng cho từng interaction có positive item `i` thuộc nhóm `g` (chi tiết "trung bình theo cái gì" xem lưu ý ngay dưới).
- `-\log\sigma(s^+_{ui}-s^-_{uj})`: chính là công thức tính loss của MỘT interaction đơn lẻ — giống hệt biểu thức bên trong `Σ` của công thức BPR ở Khối 5, chỉ khác là ở đây tính riêng cho từng interaction rồi nhóm lại theo `g`, thay vì gộp chung tất cả.

**Lưu ý dễ hiểu nhầm về ký hiệu trên:** chỉ số `i∈g` gợi ý "trung bình theo từng item riêng biệt thuộc nhóm g", nhưng thực ra công thức trung bình theo từng **lượt tương tác** (interaction/triplet) có positive item thuộc nhóm `g` — nếu 1 item phổ biến (như `i1`) xuất hiện làm positive của 3 user khác nhau trong batch, nó được tính **3 lần** (3 dòng riêng trong bảng ở mục 3), không phải tính 1 lần rồi lấy trung bình theo số item. Đây đúng là cách bảng ví dụ ở mục 3 đã làm (`i1` xuất hiện 3 dòng, ứng với `u1,u2,u3`).

**Nhóm "middle" ở đâu?** Công thức trên chỉ có `head` và `tail` — nhóm **middle hoàn toàn không xuất hiện trong `L_ILE`** (xác nhận trực tiếp từ code `src/ile_losses.py`: chỉ có `head_mask`, `tail_mask`, không có biến nào cho middle). Các interaction có positive item thuộc nhóm middle vẫn được tính bình thường trong `L_BPR` (Khối 5), nhưng **không được ILE dùng để kéo/đẩy gì cả** — middle chỉ là một nhãn phân loại dùng để thống kê (ví dụ bảng Popularity Groups), không phải một "cực" thứ 3 trong ILE.

**Bước 2 — Phạt chênh lệch:**
$$\mathcal{L}_{ILE} = (\bar{\ell}_{head} - \bar{\ell}_{tail})^2$$

**Trường hợp batch không có đủ cả 2 nhóm thì sao?** Xác nhận từ code: nếu batch không có bất kỳ interaction nào thuộc head, HOẶC không có bất kỳ interaction nào thuộc tail, `L_ILE` của batch đó được trả về thẳng **= 0** (bỏ qua, không tính) — không phải lỗi, không phải giá trị mặc định nguy hiểm, chỉ đơn giản là "batch này không đủ dữ liệu để so sánh 2 nhóm, nên bỏ qua ILE cho batch này". Điều kiện áp dụng: chỉ tính khi batch có **cả** item head lẫn tail.

**Vì sao bình phương, không phải trị tuyệt đối?** Bình phương đảm bảo kết quả luôn không âm và có đạo hàm mượt tại mọi điểm (kể cả tại 0), thuận lợi cho lan truyền ngược. *(kiến thức nền: kỹ thuật chọn hàm phạt)*

**Vì sao PHẢI bình phương (không chỉ trừ đơn thuần)?** Đây là chi tiết đặc thù của chính đồ án, không phải kiến thức nền chung: phiên bản đầu tiên của công thức trong code từng dùng trực tiếp `head_loss - tail_loss` không bình phương — theo đúng comment còn lưu trong code hiện tại, phiên bản đó "lật dấu + không chặn dưới, khiến càng train càng đẩy về phía item phổ biến" (làm bias tệ hơn). Bình phương là bản sửa lỗi chính thức hiện dùng.

### 6. Ví dụ tính toán từng bước

Nhóm head có 3 interaction (`u1,u2,u3`, đều có positive `i1` — đúng bảng ở mục 3, đã sửa negative của `u1` thành `i3`): loss `[0.6874, 0.6934, 0.6934]`:
$$\bar{\ell}_{head} = \frac{0.6874+0.6934+0.6934}{3} = \frac{2.0742}{3} = 0.6914$$

Nhóm tail có 1 interaction (`u4` với `i3`): loss `[0.6675]`:
$$\bar{\ell}_{tail} = 0.6675$$

$$\mathcal{L}_{ILE} = (\bar{\ell}_{head}-\bar{\ell}_{tail})^2 = (0.6914-0.6675)^2 = (0.0239)^2 \approx 0.00057$$

Chênh lệch ở ví dụ này rất nhỏ vì embedding vẫn gần như ngẫu nhiên (mới 1 lớp lan truyền, chưa huấn luyện) — tất cả các loss riêng lẻ đều gần `log(2)≈0.693`, tức mô hình đang "đoán mò" đều cho mọi nhóm. Trong thực tế huấn luyện thật (đã học nhiều epoch), `l̄_head` giảm nhanh hơn hẳn `l̄_tail` do item head có nhiều dữ liệu hơn — đó chính là lúc `L_ILE` tăng lên rõ rệt và bắt đầu phát huy tác dụng "kéo" 2 giá trị lại gần nhau.

### 7. Hình minh hoạ ASCII

```
Trước ILE (giai đoạn train sau, minh hoạ xu hướng):     Sau ILE (mục tiêu):
loss                                                     loss
 █          head thấp (học nhanh, nhiều data)            █      head ≈ tail
 █  ▄▄▄▄▄▄▄▄ tail cao (học chậm, ít data)                █  ▄▄▄▄ (kéo gần lại)
─┴──┴────┴──                                            ─┴──┴────┴──
  head  tail                                              head  tail
(ở ví dụ số tại mục 6, 2 giá trị 0.6914/0.6675 CÒN GẦN NHAU vì embedding chưa học gì — độ chênh sẽ rõ hơn khi train lâu hơn)
```

### 8. Vai trò trong toàn bộ mô hình

Dùng lại chính `s⁺, s⁻` từ Khối 4 (không cần lan truyền riêng), chỉ nhóm lại theo `item_group`. Chỉ hoạt động khi `use_ile=True`.

### 9. Nếu bỏ khối này

Mô hình chỉ tối ưu BPR trung bình chung — vì item phổ biến có nhiều dữ liệu hơn, mô hình tự nhiên học tốt nhóm này hơn (đúng vấn đề Popularity Bias ban đầu).

### 10. Điểm mới (Contribution)

**Có, đây là điểm mới** — đóng góp trung tâm của phương pháp đề xuất.

> **Tổng kết Phần III:** đến đây, ta đã có đủ 1 mô hình LightGCN hoàn chỉnh + 2 cải tiến (Negative Sampling thông minh hơn, ILE công bằng hơn) — toàn bộ chỉ dùng MỘT nhánh (Nhánh A), MỘT đồ thị (gốc). Phần IV sẽ giới thiệu một nhánh HOÀN TOÀN riêng biệt, chỉ để giúp embedding "học tốt hơn", không liên quan gì đến việc dự đoán ở Phần III.

---
---

# PHẦN IV — NHÁNH PHỤ: HỌC BIỂU DIỄN BỀN VỮNG HƠN (Branch B)

*Toàn bộ Phần IV trả lời câu hỏi: "Làm sao giúp embedding ổn định và công bằng hơn, mà không đụng vào đường dự đoán ở Phần III?" — con đường này KHÔNG BAO GIỜ được dùng để tạo gợi ý cuối cùng.*

## KHỐI 8 — Degree-Aware Graph Augmentation
*(Tương ứng Khối ② trên sơ đồ gốc)*

### 1. Khối này làm gì?

Tạo ra 2 phiên bản "bị nhiễu có chủ đích" của đồ thị gốc (Khối 1), bằng cách xoá bớt cạnh — **ưu tiên xoá cạnh nối tới item phổ biến**. Mục đích: chuẩn bị nguyên liệu cho Contrastive Learning (Khối 10), ép mô hình chú ý nhiều hơn tới item ít phổ biến.

### 2. Ý tưởng trực quan

Giống TikTok: nếu thuật toán luôn thấy đầy đủ dữ liệu của video viral, nó chỉ học được "viral thì tốt" — không học được điều tinh tế hơn về vì sao 1 video ít người xem lại hợp với 1 nhóm khán giả nhỏ. Cố tình "che bớt" một phần dữ liệu của item phổ biến trong lúc luyện tập buộc mô hình chú ý nhiều hơn đến quan hệ tinh tế liên quan item ít phổ biến. *(kiến thức nền, ví von)*

### 3. Input

Đồ thị gốc (Khối 1) + `item_degree`.

**Ví dụ cụ thể:** dùng đúng `E_train` (7 cạnh) và degree thật ở Khối 1: `deg(i1)=4` (head, phổ biến nhất), `deg(i2)=2` (middle), `deg(i3)=1` (tail).

### 4. Output

2 đồ thị mới — View 1, View 2 (lấy mẫu độc lập). Dùng làm input cho **Khối 9 (Branch B)**.

**Ví dụ cụ thể** (minh hoạ 1 lần lấy mẫu cụ thể — vì `i1` có `p_drop=0.4` cao nhất, xác suất cạnh nối tới `i1` bị xoá là cao nhất; giả sử lần này View 1 xoá đúng cạnh `(u2,i1)`, View 2 xoá đúng cạnh `(u3,i1)`):
```
View 1 = { (u1,i1), (u3,i1), (u4,i1), (u1,i2), (u3,i2), (u4,i3) }   ← thiếu (u2,i1); deg(i1) trong View 1 = 3
View 2 = { (u1,i1), (u2,i1), (u4,i1), (u1,i2), (u3,i2), (u4,i3) }   ← thiếu (u3,i1); deg(i1) trong View 2 = 3
```
Chú ý: `u2` chỉ có duy nhất 1 cạnh (`(u2,i1)`) trong đồ thị gốc — khi cạnh đó bị xoá ở View 1, `u2` trở thành **node cô lập** (không còn hàng xóm nào) trong lần lan truyền đó. Đây là một trường hợp biên có thật của cơ chế dropout, sẽ quay lại ở Khối 9.

### 5. Công thức toán học

$$p_{drop}(i) = p_{min} + (p_{max} - p_{min}) \cdot \frac{\log(1 + \deg_i)}{\log(1 + \deg_{max})}, \quad p_{drop}(i) \in [0.1,\ 0.4]$$

**Giải thích từng thành phần:**
- `p_drop(i)`: xác suất mà MỖI cạnh nối tới item `i` bị xoá (dropout) — kết quả cuối của công thức, 1 số trong khoảng `[0.1, 0.4]`.
- `p_min, p_max`: 2 hằng số cấu hình sẵn (`0.1` và `0.4`) — giới hạn dưới/trên của xác suất xoá, để không item nào bị xoá 100% (mất hẳn) hay 0% (không bao giờ bị xoá) cạnh.
- `deg_i`: degree của item `i` (đã tính ở Khối 1).
- `deg_max`: degree LỚN NHẤT trong toàn bộ tập item (ví dụ trong dữ liệu của ta: `deg_max = deg(i1) = 4`).
- `log(1+\deg_i)` và `log(1+\deg_{max})`: logarit tự nhiên của degree cộng thêm 1 (cộng 1 để tránh `log(0)` không xác định khi có item degree bằng 0).
- Tỷ số `\frac{\log(1+\deg_i)}{\log(1+\deg_{max})}`: luôn nằm trong khoảng `[0,1]` — bằng `0` khi `deg_i=0` (item ít phổ biến nhất có thể), bằng `1` khi `deg_i=deg_{max}` (item phổ biến nhất trong dữ liệu).
- `(p_max-p_min)\cdot\text{tỷ số trên}`: "trải" tỷ số `[0,1]` ra thành khoảng `[0, p_max-p_min]`.
- Cộng thêm `p_min` ở đầu công thức: dịch chuyển kết quả từ `[0, p_max-p_min]` sang đúng khoảng mong muốn `[p_min, p_max] = [0.1, 0.4]`.
- `∈ [0.1, 0.4]` (xem Bảng ký hiệu — `∈` là "thuộc"): khẳng định kết quả `p_drop(i)` luôn nằm trong khoảng đóng này, không bao giờ vượt ra ngoài.

`log(1+x)`: nén khoảng giá trị degree (item cực phổ biến không phóng đại quá mức chênh lệch). *(kiến thức nền: lý do dùng log)*

**Dropout đối xứng:** quyết định giữ/xoá chỉ thực hiện 1 lần trên mỗi cạnh vô hướng (xét theo item), rồi mirror sang cả 2 chiều — giữ đồ thị vô hướng hợp lệ (cần cho công thức `D^(-1/2)AD^(-1/2)` ở Khối 2 có ý nghĩa đúng khi áp dụng lại).

### 6. Ví dụ tính toán từng bước

`deg_max=4` (chính là `deg(i1)`, item phổ biến nhất trong dữ liệu). Áp dụng công thức cho cả 3 item:

$$p_{drop}(i1) = 0.1+0.3\times\frac{\log(1+4)}{\log(1+4)} = 0.1+0.3\times1 = 0.4\ (40\%)$$
$$p_{drop}(i2) = 0.1+0.3\times\frac{\log(1+2)}{\log(1+4)} = 0.1+0.3\times\frac{1.0986}{1.6094} = 0.1+0.3(0.6826) = 0.3048\ (30.5\%)$$
$$p_{drop}(i3) = 0.1+0.3\times\frac{\log(1+1)}{\log(1+4)} = 0.1+0.3\times\frac{0.6931}{1.6094} = 0.1+0.3(0.4307) = 0.2292\ (22.9\%)$$

→ Item phổ biến nhất (`i1`, head) có xác suất mất cạnh (40%) gần gấp đôi item ít phổ biến nhất (`i3`, tail, 22.9%) — đúng cạnh `(u2,i1)` và `(u3,i1)` giả định bị xoá ở mục 4 đều là cạnh của `i1`, phù hợp với việc `i1` có `p_drop` cao nhất.

### 7. Hình minh hoạ ASCII

```
Đồ thị gốc:                 View 1 (kết quả giả định ở mục 4):
u2 --- i1 (head, p=0.4)     u2 - - - i1   (cạnh bị xoá, nét đứt — xác suất cao nhất)
u4 --- i3 (tail, p=0.229)   u4 ──── i3    (cạnh được giữ, xác suất xoá thấp hơn nhiều)
```

### 8. Vai trò trong toàn bộ mô hình

Bước đầu của Nhánh B, cung cấp input cho Khối 9.

### 9. Nếu bỏ khối này (thay bằng dropout ngẫu nhiên đều)

Đây chính là cách làm gốc của kỹ thuật self-supervised graph learning phổ biến (SGL) *(kiến thức nền)* — Contrastive Learning vẫn hoạt động, nhưng mất tính "nhắm mục tiêu vào popularity bias".

### 10. Điểm mới (Contribution)

**Có, đây là điểm mới.** Dropout ngẫu nhiên đều đã có trong literature (SGL); làm xác suất dropout phụ thuộc degree là cải tiến của phương pháp đề xuất.

> **Tiếp theo:** ta có 2 đồ thị "nhiễu có chủ đích" — giờ đưa chúng qua đúng Backbone (Khối 2) đã học, để ra 2 bộ embedding.

## KHỐI 9 — Branch B (Nhánh phụ)
*(Tương ứng "Branch B — Auxiliary path" trên sơ đồ gốc)*

### 1. Khối này làm gì?

Áp dụng Backbone (Khối 2) **2 lần riêng biệt**, một lần trên View 1, một lần trên View 2 (Khối 8) — tạo ra 2 bộ embedding hơi khác nhau cho cùng một tập user/item.

**Lưu ý quan trọng (xem lại phần "Câu hỏi quan trọng" đầu tài liệu nếu chưa đọc):** 2 bộ embedding này **không hề** được gửi tới Nhánh A hay tới Khối 13 (Inference) — chúng chỉ dùng để tính Contrastive Loss (Khối 10) trong batch hiện tại rồi "biến mất". Điểm khối này thực sự tác động lâu dài là: cả 2 lần gọi Backbone ở đây đều xuất phát từ **đúng bảng `E^(0)` mà Nhánh A cũng đang dùng** — nên gradient của Contrastive Loss sẽ quay lại chỉnh sửa đúng bảng `E^(0)` đó, ảnh hưởng gián tiếp tới Nhánh A ở **bước cập nhật kế tiếp**, chứ không phải bằng cách "gửi embedding sang" cho Nhánh A ngay trong batch này.

### 2. Ý tưởng trực quan

Giống vận động viên bơi lội: thi đấu chính thức (Nhánh A) luôn bơi đúng chuẩn. Nhưng buổi tập, huấn luyện viên cho bơi với tạ nhẹ ở tay chân (đồ thị bị dropout) để ép phát triển thêm nhóm cơ ít dùng — khi thi đấu thật, tháo tạ ra bơi bình thường, nhưng cơ thể đã khoẻ hơn nhờ bài tập. *(kiến thức nền, ví von)*

### 3. Input

View 1, View 2 từ Khối 8 — chính là 2 danh sách cạnh ví dụ ở Khối 8.

### 4. Output

"View-1 embeddings" và "View-2 embeddings" — 2 ma trận riêng biệt, cùng shape `[num_users+num_items, dim]` nhưng **giá trị khác nhau** (vì lan truyền trên 2 đồ thị khác nhau).

**Ví dụ cụ thể** (tính tay đầy đủ ở mục 6 bên dưới, cho item `i1` — chọn `i1` vì đây là item vừa bị 2 view xoá 2 cạnh khác nhau, nên thấy rõ nhất sự khác biệt):
```
i1 trong View-1 embedding = [0.1908, 0.0092]   (thiếu cạnh (u2,i1) → deg(i1) trong view này = 3)
i1 trong View-2 embedding = [0.2257, 0.0423]   (thiếu cạnh (u3,i1) → deg(i1) trong view này = 3)
```
Hai vector này **khác nhau rõ rệt dù cùng là `i1`** — chỉ vì view nào xoá cạnh của user nào. Chính sự khác biệt này là thứ Contrastive Loss (Khối 10) sẽ cố "kéo lại gần nhau". Output này **chỉ** dùng làm input cho **Khối 10**, không đi qua Khối 4 (Scoring).

### 5. Công thức toán học

Dùng đúng công thức Khối 2, áp dụng 2 lần độc lập.

### 6. Ví dụ tính toán từng bước

Tính `e_i1` trong View 1 (hàng xóm còn lại của `i1`: `u1, u3, u4`, vì `(u2,i1)` đã bị xoá — `deg(i1)_view1=3`; degree của `u1,u3,u4` không đổi so với đồ thị gốc: `deg(u1)=2,deg(u3)=2,deg(u4)=2`):

$$e_{i1}^{(1)}\big|_{view1} = \sum_{u\in\{u1,u3,u4\}} \frac{1}{\sqrt{3}\sqrt{\deg(u)}} e_u^{(0)} = \frac{1}{\sqrt{3}\sqrt{2}}\big([0.10,-0.20]+[-0.10,0.05]+[0.20,-0.05]\big)$$
$$= \frac{1}{2.449}[0.20,-0.20] = [0.0816,-0.0816]$$
$$e_{i1}\big|_{view1} = \frac{1}{2}\big([0.30,0.10]+[0.0816,-0.0816]\big) = [0.1908,0.0092]$$

Tính `e_i1` trong View 2 (hàng xóm còn lại: `u1, u2, u4`, vì `(u3,i1)` bị xoá — `deg(i1)_view2=3`; chú ý `deg(u2)=1`, khác `deg(u1)=deg(u4)=2`):

$$e_{i1}^{(1)}\big|_{view2} = \frac{1}{\sqrt{3}\sqrt{2}}[0.10,-0.20] + \frac{1}{\sqrt{3}\sqrt{1}}[0.05,0.15] + \frac{1}{\sqrt{3}\sqrt{2}}[0.20,-0.05]$$
$$= [0.0408,-0.0816]+[0.0289,0.0866]+[0.0816,-0.0204] = [0.1513,-0.0155]$$
$$e_{i1}\big|_{view2} = \frac{1}{2}\big([0.30,0.10]+[0.1513,-0.0155]\big) = [0.2257,0.0423]$$

Hai kết quả `[0.1908,0.0092]` và `[0.2257,0.0423]` khác nhau chủ yếu vì `u2` (degree thấp, `deg=1`) chỉ xuất hiện ở View 2, đóng góp trọng số hàng xóm khác hẳn so với khi nó vắng mặt ở View 1.

**Còn `u2` thì sao — nó "biến mất" ở View 1 thì embedding của chính nó là gì?** Đây là câu hỏi bỏ ngỏ ở Khối 8: `u2` chỉ có duy nhất 1 cạnh `(u2,i1)`, cạnh này bị xoá ở View 1 → `deg(u2)_view1 = 0` (node cô lập, không còn hàng xóm nào). Áp dụng đúng cách xử lý đã nêu ở Khối 2 (`deg_inv_sqrt=0` khi `deg=0`, xác nhận từ code):
$$e_{u2}^{(1)}\big|_{view1} = 0 \quad \text{(tổng trên tập hàng xóm rỗng — không có ai để "hỏi ý kiến")}$$
$$e_{u2}\big|_{view1} = \frac{1}{2}\Big(e_{u2}^{(0)} + e_{u2}^{(1)}\big|_{view1}\Big) = \frac{1}{2}\big([0.05,0.15]+[0,0]\big) = [0.025,0.075]$$
Nghĩa là: `u2` bị cô lập **không** làm chương trình lỗi hay ra `NaN` — nó chỉ đơn giản "không nhận thêm thông tin từ hàng xóm ở lớp 1", nên embedding cuối cùng của nó ở View 1 chính là **một nửa** embedding gốc `E^(0)` (vì lớp 1 đóng góp 0). Ở View 2, `u2` vẫn còn cạnh `(u2,i1)` nên không bị ảnh hưởng — embedding của nó ở View 2 được tính bình thường như mọi node khác.

### 7. Hình minh hoạ ASCII

```
Đồ thị gốc (Khối 1)
    │
    ▼
Khối 8: tạo 2 view
    │
    ├──────────────┐
    ▼              ▼
  View 1          View 2
    │              │
    ▼              ▼
Khối 2: Backbone  Khối 2: Backbone   (CHẠY LẠI, 2 lần riêng biệt)
    │              │
    ▼              ▼
Emb view 1     Emb view 2
    └──────┬───────┘
           ▼
   Khối 10: Contrastive Loss
```

### 8. Vai trò trong toàn bộ mô hình

"Nguồn nuôi" duy nhất của Contrastive Loss. Chạy **song song** với Nhánh A (Khối 3) mỗi bước huấn luyện — tổng cộng mỗi bước có **3 lần lan truyền đồ thị**: 1 lần Nhánh A (Khối 3), 2 lần Nhánh B (khối này).

### 9. Nếu bỏ khối này

Contrastive Loss (Khối 10) không còn tồn tại — mô hình quay về chỉ có BPR + ILE (+ L2).

### 10. Điểm mới (Contribution)

Ý tưởng "một nhánh phụ chỉ phục vụ contrastive learning, tách biệt hoàn toàn khỏi luồng dự đoán chính" là điểm thiết kế của phương pháp đề xuất.

> **Tiếp theo:** ta có 2 bộ embedding của cùng một tập user/item, nhưng từ 2 "góc nhìn" đồ thị khác nhau. Làm sao dùng chúng để dạy mô hình?

## KHỐI 10 — Contrastive Loss
*(Tương ứng Khối ⑤③ trên sơ đồ gốc)*

### 1. Khối này làm gì?

So sánh 2 bộ embedding (Khối 9), ép embedding của **cùng một** user/item ở 2 view phải giống nhau, trong khi embedding của các user/item **khác nhau** phải khác nhau — tạo biểu diễn ổn định hơn, ít phụ thuộc chi tiết ngẫu nhiên của đồ thị.

### 2. Ý tưởng trực quan

Nhận ra một người bạn dù nhìn từ 2 góc chụp ảnh khác nhau — bản chất người đó không đổi dù "góc nhìn" khác. Contrastive Learning dạy: "dù đồ thị bị thay đổi một chút (mất vài cạnh), bản chất của user/item này vẫn phải được nhận diện là chính nó" — từ đó mô hình học đặc điểm cốt lõi, ổn định. *(kiến thức nền, ví von)*

### 3. Input

`u1, i1` (toàn bộ embedding user/item của View 1), `u2, i2` (toàn bộ embedding user/item của View 2) — từ Khối 9. Chú ý: `u1` ở đây nghĩa là "ma trận embedding user của View 1", không phải "user thứ nhất" — hơi dễ nhầm vì cùng ký hiệu. Phần dưới đây minh hoạ riêng nhánh **item** (`L_NCE^item`) vì Khối 9 đã tính sẵn embedding của `i1` ở 2 view; nhánh user (`L_NCE^user`) tính hoàn toàn tương tự, chỉ đổi sang embedding user.

**Ví dụ cụ thể** (batch minh hoạ gồm 2 item, `i1` và `i3`, lấy đúng từ Khối 9 và Khối 2): 
- `i1`: View 1 → `[0.1908,0.0092]` (đã tính ở Khối 9); View 2 → `[0.2257,0.0423]` (đã tính ở Khối 9).
- `i3`: neighbor duy nhất của `i3` là `u4`, cạnh này **không** bị xoá ở view nào trong ví dụ Khối 8 → embedding `i3` giống hệt nhau ở cả 2 view, bằng đúng giá trị đã tính ở Khối 2: `[0.1457,0.1073]`.

### 4. Output

Số thực (loss) — **ví dụ cụ thể: `L_NCE^{item} ≈ 0.4250`** (tính đầy đủ ở mục 6 từ đúng 2 item `i1, i3` trên) — cộng vào Total Loss (Khối 12) với trọng số `λ_CL` (sau khi cộng thêm `L_NCE^user`, tính bằng đúng cơ chế này nhưng trên embedding user).

### 5. Công thức toán học

$$\mathcal{L}_{CL} = \mathcal{L}_{NCE}^{user} + \mathcal{L}_{NCE}^{item}$$

**Giải thích:** `L_NCE^user` và `L_NCE^item` là 2 giá trị loss TÁCH RIÊNG — một cái tính trên embedding user của 2 view, một cái tính trên embedding item của 2 view (công thức bên dưới áp dụng y hệt cho cả hai, chỉ đổi "user" thành "item"). Cộng thẳng lại (không có trọng số riêng ở bước này — trọng số `λ_CL` chỉ áp dụng ở Khối 12, cho TỔNG `L_CL`).

Tổng 2 InfoNCE riêng biệt (user, item), mỗi cái:
$$\text{logits} = \frac{z_1 \cdot z_2^\top}{\tau}, \qquad \mathcal{L}_{NCE} = \text{CrossEntropy}(\text{logits},\ \text{diag})$$

**Giải thích từng thành phần:**
- `z1, z2`: 2 ma trận embedding (của View 1, View 2 — Khối 9) đã **chuẩn hoá về độ dài 1** (chia mỗi vector cho chính độ dài/chuẩn của nó — xem công thức `z=e/‖e‖` ở mục 6) — so sánh chỉ dựa trên "hướng", không bị ảnh hưởng bởi độ lớn vector.
- `z2^⊤` (chuyển vị — xem Bảng ký hiệu): lật ma trận `z2` từ "mỗi dòng 1 node" thành "mỗi cột 1 node", để phép nhân ma trận `z1 · z2^⊤` cho ra đúng 1 ma trận vuông kích thước `[số node × số node]`, trong đó phần tử ở hàng `a`, cột `b` chính là tích vô hướng giữa `z1[a]` và `z2[b]`.
- `τ` (tau, "nhiệt độ" — temperature): hyperparameter chia vào MỌI phần tử của ma trận trên — càng nhỏ thì các khác biệt giữa các phần tử càng bị "phóng đại", khiến mô hình phân biệt dứt khoát hơn giữa cặp đúng và cặp sai. *(kiến thức nền: vai trò temperature)*
- `logits`: tên gọi của ma trận điểm số THÔ (chưa qua softmax/xác suất hoá) — đầu vào trực tiếp của CrossEntropy bên dưới.
- `CrossEntropy(logits, diag)`: hàm loss phân loại chuẩn *(kiến thức nền)* — với MỖI hàng của ma trận `logits`, coi đó là "điểm số dự đoán cho từng lựa chọn có thể", áp dụng softmax để biến thành xác suất, rồi lấy `-log(xác suất của đáp án đúng)`. Kết quả cuối `L_NCE` là TRUNG BÌNH của giá trị này qua mọi hàng.
- `diag` (đường chéo): quy ước "đáp án đúng" — với hàng thứ `k` (ứng với node `k` ở View 1), đáp án đúng là CỘT thứ `k` (chính node `k` đó, nhưng ở View 2) — vì đây là 2 view của CÙNG một node, embedding của chúng phải được "nhận ra nhau".
- **Một chiều, không đối xứng**: chỉ tính `z1→z2` (dùng `z1` làm "hàng", `z2` làm "cột"), không cộng thêm chiều ngược `z2→z1` như một số biến thể khác trong literature. *(chi tiết lấy trực tiếp từ code)*

### 6. Ví dụ tính toán từng bước

**Bước 1 — Chuẩn hoá về độ dài 1** (`z = e/‖e‖`):
$$z_1[i1]=\frac{[0.1908,0.0092]}{0.1910}=[0.9990,0.0482],\quad z_2[i1]=\frac{[0.2257,0.0423]}{0.2296}=[0.9831,0.1842]$$
$$z_1[i3]=z_2[i3]=\frac{[0.1457,0.1073]}{0.1810}=[0.8051,0.5931]\ \ (\text{giống hệt nhau ở 2 view, vì } i3\text{ không đổi})$$

**Bước 2 — Ma trận logits** `= (z1 @ z2ᵀ)/τ`, chọn `τ=0.2`:
```
                z2[i1]=[0.9831,0.1842]   z2[i3]=[0.8051,0.5931]
z1[i1]=[..]        0.9911/0.2=4.955         0.8329/0.2=4.164
z1[i3]=[..]        0.9009/0.2=4.504         1.0000/0.2=5.000
```

**Bước 3 — CrossEntropy, nhãn đúng nằm trên đường chéo** (hàng `i1` → đúng là cột `i1`; hàng `i3` → đúng là cột `i3`):
$$\text{softmax hàng } i1: \frac{e^{4.955}}{e^{4.955}+e^{4.164}} = \frac{141.8}{141.8+64.4}=0.688 \Rightarrow \ell_{i1}=-\log(0.688)=0.374$$
$$\text{softmax hàng } i3: \frac{e^{5.000}}{e^{4.504}+e^{5.000}} = \frac{148.4}{90.4+148.4}=0.622 \Rightarrow \ell_{i3}=-\log(0.622)=0.476$$

**Bước 4 — Trung bình:**
$$\mathcal{L}_{NCE}^{item} = \frac{0.374+0.476}{2} = 0.4250$$

Nhận xét: hàng chéo (`i1↔i1`: 4.955, `i3↔i3`: 5.000) đều lớn hơn hàng lệch chéo (4.164, 4.504) — đúng hướng "pull" (kéo gần đúng cặp) mà InfoNCE khuyến khích, dù chưa lớn hơn nhiều vì embedding vẫn ở vòng lan truyền đầu.

### 7. Hình minh hoạ ASCII

```
z1[i1]=[0.999,0.048] ──┐
                         ├── cos sim=0.991, logit=4.955 (cao nhất hàng i1) ──► pull đúng hướng
z2[i1]=[0.983,0.184] ──┘

z1[i1] ──┐
          ├── cos sim=0.833, logit=4.164 (thấp hơn) ──► push (i1 phải xa i3 hơn xa chính nó)
z2[i3] ──┘
```

### 8. Vai trò trong toàn bộ mô hình

Nhận input trực tiếp từ Khối 9 — **không đi qua Khối 4 (Scoring)**. Chỉ hoạt động khi `use_cl=True`.

### 9. Nếu bỏ khối này

Embedding chỉ định hình bởi BPR + ILE (Nhánh A) — mất tín hiệu tự-giám-sát bổ sung.

### 10. Điểm mới (Contribution)

Cơ chế InfoNCE không hoàn toàn mới (đã có trong SGL). Đóng góp nằm ở việc **kết hợp** với Degree-Aware Augmentation (Khối 8) thay vì dropout ngẫu nhiên đều.

> **Tổng kết Phần IV:** Nhánh B (Khối 8, 9, 10) hoạt động hoàn toàn song song với Nhánh A, không bao giờ giao nhau ngoại trừ việc CÙNG đóng góp vào Total Loss (Khối 12). Giờ ta gộp tất cả lại.

---
---

# PHẦN V — GỘP LẠI & VẬN HÀNH THẬT

## KHỐI 11 — L2 Regularization
*(Tương ứng Khối ⑤④ trên sơ đồ gốc)*

### 1. Khối này làm gì?

Ngăn embedding "phình to" không kiểm soát trong lúc huấn luyện, bằng cách phạt nhẹ embedding có độ lớn quá cao.

### 2. Ý tưởng trực quan

Nếu không ràng buộc, mô hình có thể "gian lận" giảm BPR loss bằng cách phóng to toàn bộ embedding (vì dot product tỉ lệ thuận độ lớn cả 2 vector) — thay vì học đúng *hướng* quan hệ. L2 giống một "giới hạn ngân sách", buộc mô hình dùng đúng *hướng* để biểu đạt sở thích. *(kiến thức nền)*

### 3. Input

`u₀, p₀, n₀` — embedding gốc (layer 0, TRƯỚC khi qua Khối 2 — tức là `E^(0)` ban đầu, chưa lan truyền) của user/positive/negative-item trong batch.

**Ví dụ cụ thể** (dùng đúng bộ ba `(u1,i1,i3)` đã dùng xuyên suốt Khối 4/5/7 — lấy đúng dòng `E^(0)` ở Khối 2, KHÔNG phải embedding đã lan truyền): `u₀=e_{u1}^{(0)}=[0.10,-0.20]`, `p₀=e_{i1}^{(0)}=[0.30,0.10]`, `n₀=e_{i3}^{(0)}=[0.15,0.25]`.

### 4. Output

Số thực (penalty nhỏ) — **ví dụ cụ thể: `L_reg = 0.235`** (tính ở mục 6, batch chỉ có đúng 1 interaction nên không chia gì thêm) — cộng vào Total Loss (Khối 12) với trọng số `wd`.

### 5. Công thức toán học

$$\mathcal{L}_{reg} = \frac{\|u_0\|^2 + \|p_0\|^2 + \|n_0\|^2}{|batch|}$$

**Giải thích từng thành phần:**
- `u₀, p₀, n₀`: embedding **gốc** (layer 0, `E^(0)`, KHÔNG phải đã lan truyền qua Khối 2, KHÔNG phải toàn bộ tham số mô hình) của user/positive-item/negative-item — chỉ đúng dòng tương ứng batch hiện tại.
- `‖u₀‖` (chuẩn — xem Bảng ký hiệu toán học): độ dài của vector `u₀`, tính bằng `\sqrt{\sum_d u_0[d]^2}`.
- `‖u₀‖²` (chuẩn BÌNH PHƯƠNG — dạng hay gặp hơn `‖u₀‖`): vì có bình phương ở ngoài, dấu căn trong định nghĩa chuẩn bị TRIỆT TIÊU — kết quả chỉ đơn giản là **tổng bình phương từng toạ độ** của `u₀` (không cần tính căn rồi bình phương lại, xem phép tính trực tiếp ở mục 6).
- `|batch|`: kích thước batch (số interaction) — chia để ra penalty TRUNG BÌNH trên mỗi interaction, không phụ thuộc batch to hay nhỏ.

Chia `|batch|` giữ penalty không phụ thuộc kích thước batch.

### 6. Ví dụ tính toán từng bước

$$\|u_0\|^2 = 0.10^2+(-0.20)^2 = 0.01+0.04 = 0.05$$
$$\|p_0\|^2 = 0.30^2+0.10^2 = 0.09+0.01 = 0.10$$
$$\|n_0\|^2 = 0.15^2+0.25^2 = 0.0225+0.0625 = 0.085$$
$$\mathcal{L}_{reg} = \frac{0.05+0.10+0.085}{1} = 0.235$$

(Batch size ở ví dụ này = 1 vì chỉ minh hoạ đúng 1 interaction `(u1,i1,i3)`; với batch thật 4.096 interaction, tử số là tổng bình phương chuẩn của TOÀN BỘ batch, chia cho 4.096.)

### 7. Hình minh hoạ ASCII

Không cần — thuần phép tính số học.

### 8. Vai trò trong toàn bộ mô hình

Luôn hoạt động, cộng vào Total Loss với trọng số cố định `wd=1e-4`.

### 9. Nếu bỏ khối này

Nguy cơ overfitting tăng, đặc biệt khi kết hợp nhiều loss phức tạp (ILE+CL).

### 10. Điểm mới (Contribution)

**Không phải điểm mới** — giống hệt cách LightGCN gốc áp dụng L2.

> **Tiếp theo:** ta đã có đủ cả 4 thành phần loss (Khối 5, 7, 10, 11). Giờ gộp chúng thành một con số duy nhất.

## KHỐI 12 — Total Loss
*(Khung "Total Loss" trên sơ đồ gốc)*

### 1. Khối này làm gì?

Gộp 4 thành phần loss (BPR – Khối 5, ILE – Khối 7, Contrastive – Khối 10, L2 – Khối 11) thành **một con số duy nhất** để tối ưu.

### 2. Ý tưởng trực quan

Một bảng điểm tổng kết có trọng số: không chỉ lấy 1 môn, mà cộng có trọng số nhiều môn (Toán×hệ số A + Văn×hệ số B...). Các `λ` là "mức độ quan trọng" ta gán cho mỗi thành phần — hệ số 0 nghĩa là "tắt". *(kiến thức nền, ví von)*

### 3. Input

4 số thực đã tính từ trước — `L_BPR` (Khối 5), `L_ILE` (Khối 7), `L_CL` (Khối 10), `L_reg` (Khối 11) — cùng 3 hệ số cấu hình sẵn `λ_ILE, λ_CL, wd`.

**Ví dụ cụ thể** (nối trực tiếp 4 con số đã tính ở các khối trước, xem mục 6 để rõ nguồn gốc từng số): `L_BPR=0.6874` (Khối 5), `L_ILE≈0.00057` (Khối 7), `L_CL≈0.825` (Khối 10 — xem chú thích ở mục 6), `L_reg=0.235` (Khối 11), với `λ_ILE=1.0, λ_CL=0.1, wd=0.0001` (đúng cấu hình PopAware-BEST).

### 4. Output

Một số thực `L` — **ví dụ cụ thể: `L≈0.7705`** — nơi lệnh `.backward()` được gọi để tính gradient, quay lại cập nhật embedding ở Khối 2.

### 5. Công thức toán học

$$\mathcal{L} = \mathcal{L}_{BPR} + \lambda_{ILE} \cdot \mathcal{L}_{ILE} + \lambda_{CL} \cdot \mathcal{L}_{CL} + wd \cdot \mathcal{L}_{reg}$$

**Giải thích từng thành phần:**

| Ký hiệu | Là gì | Giá trị trong ví dụ (Khối 5,7,10,11) | Giá trị cấu hình PopAware-BEST |
|---|---|---|---|
| `𝓛_BPR` | Loss BPR đã tính ở Khối 5 | 0.6874 | (không có hệ số riêng — ngầm định luôn nhân 1) |
| `λ_ILE` | Hệ số (trọng số) nhân với `L_ILE` | — | 1.0 |
| `𝓛_ILE` | Loss ILE đã tính ở Khối 7 | 0.00057 | — |
| `λ_CL` | Hệ số nhân với `L_CL` | — | 0.1 |
| `𝓛_CL` | Loss Contrastive đã tính ở Khối 10 | ≈0.825 | — |
| `wd` | Hệ số nhân với `L_reg` (tên gọi từ code: `WEIGHT_DECAY`) | — | 0.0001 |
| `𝓛_reg` | Loss L2 đã tính ở Khối 11 | 0.235 | — |
| `𝓛` (không subscript) | Kết quả cuối — TỔNG có trọng số của cả 4 loss trên | ≈0.7705 | — |

Mỗi `λ` là "núm vặn" độc lập, đặt sẵn TRƯỚC khi train (không học được — khác với embedding) — đặt 0 là tắt hẳn thành phần đó. `L_BPR` không có hệ số riêng (ngầm định =1) vì là loss nền tảng, luôn bật.

### 6. Ví dụ tính toán từng bước

Ghép đúng 4 số đã tính ở các khối trước (không phải số log thật, mà là số tính tay từ ví dụ xuyên suốt tài liệu):
- `L_BPR = 0.6874` — tính ở **Khối 5**, từ cặp `(u1,i1,i3)`.
- `L_ILE ≈ 0.00057` — tính ở **Khối 7**, từ 3 interaction head (`u1,u2,u3`) + 1 interaction tail (`u4`).
- `L_CL ≈ 0.825` — ở **Khối 10** ta đã tính đầy đủ `L_NCE^{item} = 0.4250`. `L_NCE^{user}` tính bằng đúng cơ chế đó nhưng trên embedding user của 2 view (không tính chi tiết lại ở đây để tránh dài dòng) — **giả định minh hoạ** giá trị cùng độ lớn, `L_NCE^{user}≈0.40`, nên `L_CL = L_NCE^{user}+L_NCE^{item} ≈ 0.40+0.425 = 0.825`. Đây là điểm DUY NHẤT trong ví dụ xuyên suốt có một số bị giả định thay vì tính đầy đủ — được đánh dấu rõ để không nhầm với 3 số còn lại (tính chính xác 100% từ ví dụ).
- `L_reg = 0.235` — tính ở **Khối 11**, từ `(u1,i1,i3)` ego embedding.

$$\mathcal{L} = 0.6874 + (1.0)(0.00057) + (0.1)(0.825) + (0.0001)(0.235) = 0.6874+0.00057+0.0825+0.0000235 \approx 0.7705$$

Nhận xét: `L_CL` (0.825) có vẻ "lớn" hơn hẳn `L_BPR` (0.6874) trước khi nhân hệ số, nhưng sau khi nhân `λ_CL=0.1`, đóng góp thực tế (0.0825) nhỏ hơn nhiều — đây chính là vai trò của `λ`: cân bằng các loss có thang giá trị khác nhau, để không loss nào "át" các loss còn lại trong quá trình tối ưu.

### 7. Hình minh hoạ ASCII

```
Khối 5  (BPR,        0.6874        )  ──────────────┐
Khối 7  (ILE×λ_ILE,  0.00057×1.0   )  ───────────────┤
Khối 10 (CL×λ_CL,    0.825×0.1     )  ───────────────┼──► Σ ──► L≈0.7705 ──► .backward()
Khối 11 (Reg×wd,     0.235×0.0001  )  ───────────────┘                            │
                                                                                    ▼
                                                                     Cập nhật embedding (Khối 2)
```

### 8. Vai trò trong toàn bộ mô hình

Điểm **hội tụ** của toàn bộ pipeline huấn luyện — nhận cả 4 loss, xuất tín hiệu gradient duy nhất. Đây chính xác là điểm mà Nhánh A (BPR+ILE) và Nhánh B (Contrastive) **thực sự "gặp nhau"** — không phải bằng cách trộn embedding, mà bằng cách gradient của cả 2 nhánh cùng cộng dồn vào đúng một bảng `E^(0)` dùng chung (xem "Câu hỏi quan trọng" ở đầu tài liệu để hiểu đầy đủ cơ chế này).

### 9. Nếu bỏ khối này

Không thể "bỏ" — đây là cơ chế kết hợp bắt buộc để huấn luyện đồng thời nhiều mục tiêu bằng một thuật toán tối ưu.

### 10. Điểm mới (Contribution)

Từng loss riêng đã phân loại ở trên. **Việc kết hợp cả 4 thành một hàm mục tiêu, huấn luyện đồng thời (joint training)** với đúng bộ 3 hệ số này là thiết kế của phương pháp đề xuất.

> **Tiếp theo — bước cuối cùng:** sau khi huấn luyện xong bằng toàn bộ cơ chế trên, khi cần đưa gợi ý cho người dùng THẬT, mô hình dùng gì?

## KHỐI 13 — Top-K Inference
*(Tương ứng Khối ⑦ trên sơ đồ gốc)*

### 1. Khối này làm gì?

Tạo ra gợi ý thật khi hệ thống được triển khai — khác hẳn mọi bước huấn luyện phức tạp ở Phần III, IV.

### 2. Ý tưởng trực quan

Vận động viên tập luyện với đủ bài tập phụ trợ (Phần IV) — nhưng ngày thi đấu chính thức chỉ dùng đúng kỹ năng cốt lõi đã rèn luyện, không mang "dụng cụ tập" lên sân thi đấu. *(kiến thức nền, ví von — nhất quán với Khối 3, 9)*

### 3. Input

`user_emb, item_emb` (toàn bộ, đã huấn luyện xong) từ **Khối 3 (Branch A)** — **không phải** từ Khối 9 (Branch B) — cộng danh sách item user đã tương tác (để loại trừ; theo `E_train` ở Khối 1, `u1` đã tương tác `{i1, i2}`).

### 4. Output

Danh sách Top-K item (K=20 với dataset thật; ví dụ nhỏ dưới đây chỉ còn lại 1 item hợp lệ vì cả catalog chỉ có 3 item), xếp hạng giảm dần — sản phẩm cuối cùng giao cho người dùng.

**Ví dụ cụ thể (xem chi tiết ở mục 6):** dùng đúng `e_u1` và `item_emb` đã tính ở Khối 2 — điểm số của `u1` với cả 3 item là `[i1:0.0117, i2:0.0112, i3:0.0003]`; `u1` đã xem `{i1,i2}` → output cuối cùng chỉ còn **`[i3]`**.

### 5. Công thức toán học

Tái sử dụng Khối 4 (Scoring), tính cho **toàn bộ catalog cùng lúc**:
$$\text{Scores} = E_{user} \cdot E_{item}^\top \quad [\text{num\_users} \times \text{num\_items}]$$

**Giải thích từng thành phần:**
- `E_user`: ma trận embedding TẤT CẢ user (Khối 3), kích thước `[num_users × 64]` — mỗi dòng 1 user.
- `E_item`: ma trận embedding TẤT CẢ item (Khối 3), kích thước `[num_items × 64]` — mỗi dòng 1 item.
- `E_item^⊤` (chuyển vị — xem Bảng ký hiệu): lật `E_item` thành kích thước `[64 × num_items]`, để 2 ma trận "khớp chiều" khi nhân (số cột của ma trận trái phải bằng số dòng của ma trận phải).
- `E_user · E_item^⊤`: nhân ma trận — kết quả là 1 bảng số kích thước `[num_users × num_items]`, trong đó phần tử ở hàng `u`, cột `i` CHÍNH LÀ `s(u,i)` — công thức y hệt Khối 4 (`⟨e_u,e_i⟩`), chỉ khác là tính cho MỌI cặp `(u,i)` cùng lúc bằng 1 phép nhân ma trận duy nhất, thay vì lặp từng cặp.
- `[num_users × num_items]`: kích thước của ma trận kết quả `Scores` — ví dụ dataset thật: `[6.034 × 3.533]`, khoảng 21 triệu điểm số được tính trong 1 lần.

Che item đã có trong train (và val, khi đánh giá test) → `argsort`/`topk` giảm dần.

### 6. Ví dụ tính toán từng bước

Điểm số của `u1` với cả 3 item, dùng đúng `e_u1=[0.0906,-0.1198]` (Khối 2) và `item_emb` (Khối 2):
1. `s(u1,i1)=0.0117` (đã tính ở Khối 4) — nhưng `i1 ∈ {i1,i2}` (đã xem) → **bị che**.
2. `s(u1,i2)=0.0906\times(-0.025)+(-0.1198)\times(-0.1125)=-0.00227+0.01348=0.01121` — `i2` cũng đã xem → **bị che**.
3. `s(u1,i3)=0.0003` (đã tính ở Khối 4) — `i3` **chưa** xem → giữ lại.
4. Sau khi che, chỉ còn `i3` trong danh sách ứng viên → Top-K (K≥1) = **`[i3]`**.

Với catalog chỉ 3 item, kết quả này tầm thường (chỉ còn đúng 1 lựa chọn) — nhưng **cơ chế** thì giống hệt hệt dataset thật: với `|I|=3.533` item thật, một user trung bình chỉ tương tác vài chục item, nên sau khi che vẫn còn hàng nghìn ứng viên để `argsort` xếp hạng và lấy Top-20 — ví dụ nhỏ này chỉ đơn giản hoá về mặt số lượng, không đơn giản hoá về cơ chế.

### 7. Hình minh hoạ ASCII

```
Item:    i1       i2       i3
Score: 0.0117   0.0112   0.0003
Mask:  ✗seen    ✗seen     ✓
                            │  chỉ còn 1 ứng viên (vì catalog ví dụ chỉ có 3 item)
                            ▼
                    Gợi ý: [i3]

(Dataset thật: |I|=3.533, một user chỉ che vài chục item đã xem
 → còn lại hàng nghìn ứng viên → Top-20 lấy từ đó, cùng cơ chế argsort/topk)
```

### 8. Vai trò trong toàn bộ mô hình

Điểm **kết thúc** pipeline — nơi duy nhất mô hình "giao tiếp" với người dùng thật. Không có Khối 8, 9, 10 (Phần IV) nào tham gia ở đây.

### 9. Nếu bỏ khối này

Mô hình vẫn học được embedding, nhưng không có cách nào biến embedding thành danh sách gợi ý thực tế.

### 10. Điểm mới (Contribution)

**Không phải điểm mới** — quy trình suy luận chuẩn, giống hệt LightGCN gốc. **Đây chính là điểm được giữ nguyên có chủ đích**, đảm bảo tốc độ suy luận không bị ảnh hưởng bởi các thành phần huấn luyện phức tạp hơn (Phần IV).

---
---

# TỔNG QUAN TOÀN BỘ PIPELINE

Mọi thứ bắt đầu từ **dữ liệu tương tác thô** — ai đã xem phim gì. Dữ liệu này được biến thành một **đồ thị hai phía** (Khối 1).

Từ đồ thị đó, mô hình đi theo **2 con đường song song** mỗi bước huấn luyện, cả 2 đều dùng chung một cỗ máy: **LightGCN Backbone** (Khối 2) — nơi mỗi user/item "học hỏi" từ hàng xóm qua nhiều lớp lan truyền.

Con đường thứ nhất, **Nhánh A** (Khối 3), giữ nguyên đồ thị gốc, chạy Backbone 1 lần, cho ra embedding. Từ 2 embedding (1 user, 1 item), **Scoring** (Khối 4) — một phép tích vô hướng đơn giản — cho ra điểm số dự đoán. Điểm số này được dùng để tính **BPR Loss** (Khối 5) — quy tắc "item đã thích phải điểm cao hơn item chưa thích" — và item "chưa thích" (negative) đó được chọn thông minh hơn nhờ **Popularity-Aware Negative Sampling** (Khối 6): ưu tiên chọn item phổ biến làm negative, vì đây là tín hiệu học có giá trị cao hơn. Đồng thời, các điểm số này cũng được nhóm theo độ phổ biến để tính **ILE** (Khối 7) — phạt nếu mô hình học item phổ biến tốt hơn hẳn item ít phổ biến.

Con đường thứ hai, **Nhánh B** (Khối 9), không tham gia việc tạo điểm số dự đoán. Nó lấy đồ thị gốc, qua **Degree-Aware Augmentation** (Khối 8) tạo 2 phiên bản bị xoá bớt cạnh — ưu tiên xoá cạnh của item phổ biến — rồi cũng chạy qua chính Backbone đó (2 lần riêng biệt). Hai bộ embedding kết quả được so sánh bằng **Contrastive Loss** (Khối 10): ép embedding của cùng 1 thực thể ở 2 view phải giống nhau.

Bốn loss — BPR, ILE, Contrastive, và **L2 Regularization** (Khối 11, chống overfitting) — được cộng lại có trọng số thành **Total Loss** (Khối 12). Từ con số này, thuật toán lan truyền ngược và cập nhật embedding một chút. Lặp lại hàng nghìn lần.

Sau khi huấn luyện xong, khi cần đưa gợi ý thật, mô hình bỏ hết Nhánh B — chỉ dùng embedding "sạch" từ Nhánh A để **Suy luận Top-K** (Khối 13): tính điểm toàn bộ catalog, loại bỏ item đã xem, trả về Top-20.

---

# DATA FLOW

```
Ma trận tương tác thô (rating ≥ 4)
        │
        ▼
  [Khối 1] Đồ thị hai phía
        │
        ▼
 ┌───────────────────────────────────────┐
 │  E^(0) — MỘT bảng embedding DUY NHẤT,  │◄─────────────────────────────┐
 │  dùng chung cho CẢ 2 nhánh bên dưới    │                              │
 └───────────────────────────────────────┘                              │
        │                                                               │
   ┌────┴────────────────┐                                              │
   ▼                      ▼                                             │
[Khối 3] Nhánh A      [Khối 8] Augmentation                              │
   │                      │                                              │
   ▼                      ▼                                              │
[Khối 2] Backbone ×1  [Khối 9] Nhánh B → [Khối 2] Backbone ×2             │
   │                      │                                              │
   ▼                      ▼                                              │
user_emb, item_emb    2 bộ embedding  ◄─ 3 bộ embedding này KHÔNG BAO GIỜ  │
   │                      │             trộn/concat với nhau (xem "Câu    │
   │                      │             hỏi quan trọng" đầu tài liệu)     │
   ▼                      ▼                                              │
[Khối 4] Scoring      [Khối 10] Contrastive Loss                         │
   │                      │                                              │
   ▼                      │                                              │
[Khối 5] BPR Loss          │                                             │
[Khối 7] ILE Loss           │                                            │
   │                      │                                              │
   └──────────┬───────────┘                                              │
              ▼                                                          │
       [Khối 11] L2 Reg  (luôn cộng thêm)                                 │
              │                                                          │
              ▼                                                          │
       [Khối 12] Total Loss  ◄── ĐIỂM DUY NHẤT loss của 2 nhánh cộng lại  │
              │                                                          │
              ▼                                                          │
   Gradient (của CẢ 2 nhánh — cộng dồn vào ĐÚNG MỘT bảng E^(0) ở trên)    │
              │                                                          │
              └──────────────────────────────────────────────────────────┘
              Cập nhật E^(0) một lần duy nhất, quay lại đúng ô "E^(0)"
              ở trên cùng — đây là nơi 2 nhánh thực sự ảnh hưởng lẫn nhau
```

---

# LUỒNG TRAINING

Mỗi lần huấn luyện gồm nhiều **epoch** (tối đa 100). Mỗi epoch, dữ liệu train (563.204 cạnh) chia thành nhiều **mini-batch** (4.096 cặp/batch) — khoảng **137 batch/epoch**.

Với mỗi batch: (1) **Lấy mẫu** (Khối 6) — chọn positive + negative; (2) **Forward** (Khối 2, 3, 4, 8, 9) — lan truyền qua Backbone (3 lần nếu bật CL), tính điểm số; (3) **Loss** (Khối 5, 7, 10, 11, 12) — tính cả 4 thành phần, cộng lại; (4) **Backward** — `loss.backward()`, PyTorch tự tính đạo hàm; (5) **Update** — `optimizer.step()` (Adam), sau khi giới hạn độ lớn gradient tối đa 1.0 (gradient clipping) để tránh cập nhật quá đột ngột.

Cứ mỗi 5 epoch, mô hình tạm dừng, chạy thử trên **Validation** để đo Recall@K — cải thiện thì lưu "best checkpoint"; không cải thiện 10 lần liên tiếp (50 epoch) thì dừng sớm.

---

# LUỒNG INFERENCE

Khi triển khai thật, quy trình đơn giản hơn hẳn: (1) Nạp best checkpoint; (2) Lan truyền **1 lần duy nhất** trên đồ thị gốc (Khối 3, không có Khối 8/9/10); (3) Tính điểm toàn bộ catalog (Khối 13); (4) Che item đã xem; (5) Trả về Top-K.

**Khác biệt cốt lõi:** lúc train cần tối đa 3 lần lan truyền/bước (nuôi Contrastive Loss); lúc suy luận chỉ cần **1 lần** — vì Contrastive Learning và Augmentation chỉ có tác dụng "định hình" embedding trong lúc học, không cần thiết khi đã có embedding cuối cùng. Đây là lý do tốc độ suy luận gần như không đổi so với LightGCN gốc.

---

# POPULARITY BIAS

**Là gì?** Xu hướng hệ thống gợi ý liên tục đề xuất item vốn đã phổ biến, gần như bỏ qua item ít phổ biến (long-tail) — dù chúng đôi khi phù hợp hơn với một số người dùng cụ thể.

**Tại sao LightGCN gặp vấn đề này?** *(kiến thức nền, suy luận chung)* Mô hình học từ dữ liệu tương tác — item càng phổ biến càng nhiều dữ liệu (nhiều cạnh, nhiều lượt xuất hiện trong batch) để học biểu diễn tốt. Item long-tail ít dữ liệu, embedding thường kém chính xác hơn, dẫn tới điểm dự đoán thấp hơn một cách hệ thống — không phải vì kém phù hợp, mà vì mô hình chưa "học đủ".

**Giải quyết bằng gì, phối hợp ra sao?**
- **ILE** (Khối 7) tấn công hàm mục tiêu: buộc loss nhóm long-tail giảm xuống ngang nhóm phổ biến.
- **Degree-Aware Augmentation + Contrastive Learning** (Khối 8, 10) tấn công không gian embedding: làm yếu tín hiệu item phổ biến trong 2 view, ép mô hình chú ý nhiều hơn tới item long-tail.
- **Popularity-Aware Negative Sampling** (Khối 6) tấn công tín hiệu đối nghịch: item phổ biến bị chọn làm "ví dụ sai" thường xuyên hơn.

Ba cơ chế tác động vào **3 điểm khác nhau** của quá trình huấn luyện — không trùng lặp, có thể bổ sung cho nhau.

---

# SO SÁNH VỚI LIGHTGCN GỐC

| Khía cạnh | LightGCN gốc | Popularity-Aware LightGCN |
|---|---|---|
| Kiến trúc | 1 nhánh duy nhất | 2 nhánh song song (chính + phụ) |
| Loss | Chỉ BPR (+L2) | BPR + ILE + Contrastive + L2 |
| Đồ thị dùng để dự đoán | Đồ thị gốc | Đồ thị gốc — giống hệt (Nhánh A không đổi) |
| Contrastive Learning | Không có | Có (chỉ Nhánh B, không ảnh hưởng dự đoán) |
| Negative Sampling | Ngẫu nhiên đều | Có thể thiên vị theo độ phổ biến (β) |
| Fairness | Không xét riêng | ILE cân bằng loss head/tail có chủ đích |
| Chi phí tính toán lúc train | 1 lần lan truyền/bước | Tối đa 3 lần/bước (khi bật CL) |
| Tốc độ suy luận | Baseline | Gần như không đổi |
| Khả năng giảm Popularity Bias | Không có cơ chế riêng | 3 cơ chế phối hợp (ILE, CL+Augmentation, Neg Sampling) |

*(Số liệu thực nghiệm cụ thể về Accuracy/TailRecall/Coverage — xem `Slide_Review_and_Defense_QA.md`, không lặp lại ở đây để tránh nhầm giữa "kiến trúc" và "kết quả đo được".)*

---

# TỔNG KẾT

Popularity-Aware LightGCN mở rộng LightGCN gốc để giải quyết vấn đề rất thực tế: xu hướng chỉ đề xuất những gì đã phổ biến sẵn, bỏ quên phần lớn catalog.

Điểm thiết kế quan trọng nhất là **tách kiến trúc thành 2 nhánh song song, dùng chung một Backbone VÀ dùng chung một bảng embedding gốc `E^(0)`**. Nhánh chính (Phần III) giữ nguyên y hệt LightGCN gốc, chịu trách nhiệm tạo embedding dùng để dự đoán và suy luận thật. Nhánh phụ (Phần IV) hoàn toàn tách biệt — tạo 2 phiên bản đồ thị "nhiễu có chủ đích" (làm yếu tín hiệu item phổ biến), dùng Contrastive Learning để embedding ổn định hơn — nhưng **không bao giờ** được dùng để đưa ra gợi ý cuối cùng. Hai nhánh **không hề trộn embedding với nhau** ở bất cứ đâu trong forward pass; điểm duy nhất chúng thực sự ảnh hưởng lẫn nhau là lúc lan truyền ngược, khi gradient của cả BPR+ILE (Nhánh A) lẫn Contrastive Loss (Nhánh B) cùng cộng dồn vào đúng MỘT bảng `E^(0)` dùng chung đó — đây là lý do Nhánh B có thể bị bỏ hoàn toàn lúc suy luận mà không mất đi những gì nó đã "dạy" được.

Ba cơ chế cụ thể trực tiếp chống popularity bias: **ILE** cân bằng loss ngay ở hàm mục tiêu; **Degree-Aware Augmentation + Contrastive Learning** làm yếu tín hiệu phổ biến khi học biểu diễn; **Popularity-Aware Negative Sampling** chọn item phổ biến làm ví dụ sai thường xuyên hơn. Ba cơ chế tác động vào ba điểm khác nhau trong vòng đời huấn luyện, được thiết kế để bổ sung nhau.

Điểm được nhấn mạnh nhất: mọi kỹ thuật bổ sung (Nhánh phụ, Contrastive Learning, Augmentation) **chỉ tồn tại lúc huấn luyện**. Khi triển khai thật, hệ thống quay lại đúng quy trình suy luận đơn giản của LightGCN gốc — các cải tiến giảm bias không phải trả giá bằng tốc độ phục vụ người dùng thật, chỉ đánh đổi bằng thời gian huấn luyện dài hơn (do lan truyền đồ thị nhiều lần hơn mỗi bước).

Về bản chất, đây là minh hoạ cho một nguyên lý thiết kế khá phổ biến trong Machine Learning hiện đại *(kiến thức nền)*: dùng kỹ thuật tự-giám-sát và hàm mục tiêu phụ trợ để định hình biểu diễn tốt hơn lúc huấn luyện, trong khi vẫn giữ một đường suy luận đơn giản, nhanh, đã được kiểm chứng — thay vì nhồi nhét mọi độ phức tạp vào ngay bước dự đoán cuối cùng.
