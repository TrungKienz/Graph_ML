# Nhận xét của giảng viên hướng dẫn & Bộ câu hỏi bảo vệ — Popularity-Aware LightGCN

*Vai trò: giảng viên phản biện, đọc kỹ slide + đối chiếu code/số liệu thật trước buổi bảo vệ. Phiên bản này đã cập nhật sau khi các fix ở mục 3 được áp dụng vào slide.*

---

## 1. Đánh giá tổng quan (sau khi đã sửa)

**Điểm mạnh:**
- Luồng trình bày đúng chuẩn: Overview → từng thành phần → training/protocol → setup → kết quả → phân tích/kết luận.
- Slide 5 (overview) đơn giản, đúng vai trò "bản đồ" — không nhồi chi tiết.
- Slide 6–9 minh hoạ cơ chế bằng hình vẽ before/after, đồ thị mini, sơ đồ siamese, khớp đúng công thức với code.
- Slide 11 và 13 có bảng số liệu thật, mũi tên ↑/↓, và mục "Honest limitations" — không nhiều đồ án dám tự phê bình ngay trong slide.
- Slide 12 vẽ đúng từ số liệu thật.
- **Đã sửa**: nhãn component (2), phạm vi "leakage-free" ở Slide 9, chú thích quy tắc chọn operating point ở Slide 10, chú thích baseline 1-run ở Slide 12, chú thích schematic ở Slide 6, và bổ sung 2 hạn chế quan trọng nhất vào Slide 13.

**Vẫn cần lưu ý khi bảo vệ (không phải lỗi slide, mà là giới hạn thật của đồ án — xem Phần 4):** dù slide đã diễn đạt chính xác hơn, các câu hỏi khó vẫn sẽ xoay quanh đúng những điểm đó. Học thuộc phần trả lời ở mục 4, đừng chỉ đọc slide.

---

## 2. Rà soát nhanh từng slide (sau khi sửa)

| Slide | Trạng thái | Ghi chú |
|---|---|---|
| 5 — Overview | ✅ Đã sửa | Nhãn (2) đổi thành "builds views for contrastive learning" — chính xác với `aug_main=False` |
| 6 — ILE | ✅ Đã sửa | Thêm caption "Schematic illustration... not measured data" |
| 7 — Aug + CL | ✅ Đã sửa | Thêm "Scope note" ngay trên slide về `aug_main=False` |
| 8 — Neg Sampling | ✅ Ổn | Không cần sửa |
| 9 — Objective & Protocol | ✅ Đã sửa | Thêm dòng "Scope of leakage-free" phân biệt rõ 2 cấp độ |
| 10 — Setup | ✅ Đã sửa | Thêm bullet "Operating-point rule" + caveat TEST-based |
| 11 — Main Results | ✅ Ổn | Baseline ở bảng này ĐÃ là 3-seed (không cần sửa — xem Q6) |
| 12 — Frontier | ✅ Đã sửa | Thêm "Note" phân biệt baseline 1-run vs 3-seed |
| 13 — Analysis & Conclusion | ✅ Đã sửa | Bảng thu gọn, bổ sung 2 limitation quan trọng nhất |

---

## 3. Các sửa đổi đã áp dụng (tóm tắt)

1. Slide 5 & 7: nhãn component (2) → "builds views for contrastive learning" / "Scope note" giải thích `aug_main=False`.
2. Slide 9: thêm dòng phân biệt phạm vi "leakage-free" (chọn checkpoint trong 1 run vs. chọn siêu tham số qua nhiều run).
3. Slide 10: thêm bullet "Operating-point rule" nêu rõ tiêu chí + caveat đang tính trên TEST.
4. Slide 12: thêm "Note" về baseline MostPopular/BPR-MF chỉ 1 run.
5. Slide 6: thêm caption "schematic, not measured data".
6. Slide 13: gộp + bổ sung 4 "Honest limitations" (thêm 2 điểm mới: aug_main=False, chọn hyperparam trên TEST).

---

## 4. Bộ câu hỏi bảo vệ — đầy đủ, theo nhóm chủ đề

> Dùng phần này để luyện tập trả lời miệng. Mỗi câu đều có: **(a)** vì sao hội đồng có thể hỏi, **(b)** câu trả lời nên nói, **(c)** nếu bị hỏi xoáy sâu hơn thì nói gì tiếp.

### Nhóm A — Tính trung thực & phạm vi tuyên bố

**Q1. Component (2) "Degree-Aware Graph Augmentation" — nó có thực sự làm thay đổi đồ thị dùng để xếp hạng cuối cùng không?**

- *(a) Vì sao hỏi:* đọc code sẽ thấy `aug_main=False` trong mọi cấu hình báo cáo.
- *(b) Trả lời:* Không, trong 5 cấu hình cuối (baseline, accuracy, BEST, high-tail, fairness), `aug_main=False`. Đồ thị dùng để `propagate()` và tính điểm cuối luôn là đồ thị train gốc, không dropout. Degree-aware dropout ở đây chỉ tạo 2 "view" tạm thời, dùng riêng cho loss InfoNCE; sau khi tính xong `L_CL`, các view này không ảnh hưởng đến embedding cuối dùng để suy luận.
- *(c) Nếu hỏi xoáy "vậy tại sao vẫn gọi là Graph Augmentation":* Vì cơ chế (dropout theo degree) đúng là một dạng graph augmentation theo nghĩa kỹ thuật (giống SGL), chỉ khác là phạm vi áp dụng của chúng tôi giới hạn ở việc tạo view cho contrastive loss, không áp dụng lên graph chính — tôi đã sửa lại nhãn trên slide để phản ánh đúng phạm vi này. Cấu hình `aug_main=True` (áp dụng lên graph chính, không CL) có tồn tại như một ablation riêng trong `train_all_popaware.py` nhưng chưa chạy 3-seed.

**Q2. "TEST scored exactly once, no leakage" — nhưng bộ siêu tham số (λ, β) của 4 operating point được chọn dựa trên số liệu nào?**

- *(a) Vì sao hỏi:* `train_sweep_popaware.py` in ra "SWEEP RESULTS (test)" rồi chọn theo frontier trên chính các số đó.
- *(b) Trả lời:* Câu "TEST scored exactly once" đúng ở cấp *một lần train cụ thể*: mỗi run chọn checkpoint theo VAL, và gọi TEST đúng 1 lần ở cuối để lấy số liệu của run đó. Nhưng ở cấp *chọn cấu hình* (chọn λ nào là "BEST"), quy trình sweep hiện tính frontier (max Coverage@20, ràng buộc Recall@20 không giảm quá ngưỡng) dựa trên số liệu TEST của 24 điểm lưới. Đây là một dạng dò tập test qua nhiều lần thử — nhẹ hơn train trực tiếp trên test, nhưng không phải zero-leakage tuyệt đối.
- *(c) Nếu hỏi "vậy sửa thế nào cho đúng":* Chạy lại sweep với tiêu chí frontier tính trên VAL, chỉ dùng TEST một lần duy nhất sau khi đã chốt cấu hình. Tôi đã disclose rõ điều này trên Slide 9/10/13 thay vì che giấu.

**Q3. Biểu đồ cột "head vs tail loss" ở Slide 6 có phải số đo thật không?**

- *(b) Trả lời:* Không, đây là sơ đồ minh hoạ khái niệm (đã ghi chú "schematic" ngay trên slide), giải thích cơ chế của công thức `L_ILE=(ℓ̄_head−ℓ̄_tail)²`. Số liệu thật chỉ ở Slide 11–12.

**Q4. Bảng checklist "6/6 tiêu chí đạt" ở Slide 13 — các tiêu chí này có phải được định nghĩa *sau khi* thấy kết quả để chắc chắn khớp không (circular reasoning)?**

- *(a) Vì sao hỏi:* đây là câu hỏi kinh điển khi một nhóm tự chấm "đạt kỳ vọng" — hội đồng luôn nghi ngờ tiêu chí bị nắn theo kết quả.
- *(b) Trả lời:* 6 tiêu chí này được đặt ra **trước khi** chạy 3-seed cuối, dựa trên yêu cầu gốc của đề bài (trích từ trao đổi nhóm: "TailRecall chấp nhận được, không quá thấp cũng không quá cao"; giảm bias nhưng Recall/NDCG không giảm quá 5%). 5/6 tiêu chí có ngưỡng định lượng rõ ràng (tăng/giảm bao nhiêu %) đặt ra từ đầu. Tiêu chí thứ 6 ("chấp nhận được, không quá cực đoan") là tiêu chí định tính duy nhất, và tôi thừa nhận đây là tiêu chí "mềm" nhất, dễ bị nghi ngờ nhất — xem thêm Q22 (mục cherry-picking) để có câu trả lời đầy đủ hơn về điểm này.

---

### Nhóm B — Độ nghiêm ngặt thống kê

**Q5. n=3 seed là mẫu rất nhỏ. "Khoảng mean±std không chồng lấn" có phải là bằng chứng thống kê chính thức không?**

- *(b) Trả lời:* Không, đây là tiêu chí trực quan (heuristic), không thay thế được kiểm định giả thuyết chính thức (t-test, Wilcoxon). Với n=3, ước lượng std không ổn định. Chúng tôi chọn 3 seed vì giới hạn thời gian GPU (mỗi seed × 5 cấu hình × ~1h). Đây là đánh đổi thực dụng, không phải bằng chứng thống kê chặt chẽ — cần nói rõ giới hạn này nếu được hỏi, không nên khẳng định "có ý nghĩa thống kê" theo nghĩa thống kê học chính thức.
- *(c) Nếu hỏi "vậy phải làm gì để chặt chẽ hơn":* Chạy tối thiểu 5 seed, dùng paired t-test hoặc Wilcoxon signed-rank test trên từng cặp (baseline, PopAware) theo seed tương ứng, báo cáo p-value thay vì chỉ nhìn mean±std.

**Q6. Baseline LightGCN trong bảng Slide 11 có cùng số seed với PopAware không?**

- *(b) Trả lời:* Có — điều này hay bị hiểu nhầm nên tôi xin làm rõ: bảng Slide 11 lấy baseline từ `train_final_seeds.py` (cấu hình "baseline" chạy đúng 3 seed {42,0,1}, ra 0.1287±0.0005), **không phải** từ file `main_results.csv` (1 run). File 1-run chỉ được dùng làm điểm tham chiếu bên ngoài ở Slide 12 (biểu đồ frontier, để định vị MostPopular/BPR-MF) — tôi đã ghi chú rõ điều này ngay trên Slide 12.

**Q7. Tại sao chọn đúng seed {42, 0, 1}? Có ý nghĩa gì đặc biệt không, hay có thể đổi seed khác ra kết quả khác hẳn?**

- *(b) Trả lời:* Không có ý nghĩa đặc biệt — 42 là seed mặc định quy ước trong `config.py` (SEED=42, tham chiếu đùa vui "Hitchhiker's Guide" phổ biến trong ML), 0 và 1 là hai seed bổ sung đơn giản để có n=3. Việc seed không chồng lấn mean±std qua 3 lựa chọn "tuỳ ý" như vậy phần nào củng cố rằng cải thiện không phải do may mắn chọn đúng 1 seed đẹp, nhưng như đã nói ở Q5, đây vẫn là mẫu nhỏ.

**Q8. Độ lệch chuẩn báo cáo tính theo công thức nào — chia cho n hay n-1?**

- *(a) Vì sao hỏi:* kiểm tra hiểu biết kỹ thuật cơ bản về std.
- *(b) Trả lời:* Chia cho n (population std, `ddof=0` trong `pandas.Series.std`), không phải n-1 (sample std). Với n=3, std(ddof=0) nhỏ hơn std(ddof=1) một chút (hệ số √(2/3)≈0.816 lần) — nghĩa là các khoảng mean±std báo cáo trong slide **hẹp hơn** một chút so với nếu dùng ddof=1. Đây là điểm cần lưu ý: nếu dùng ddof=1 (thường chuẩn hơn khi coi 3 seed là mẫu ngẫu nhiên từ một phân phối lớn hơn), khoảng tin cậy sẽ rộng hơn, và một số kết luận "không chồng lấn" cần được kiểm tra lại cẩn thận hơn.

**Q9. Bạn quét 24 điểm hyperparameter rồi chọn ra vài điểm "đẹp nhất". Có rủi ro multiple-comparison / false discovery không (thử càng nhiều càng dễ tình cờ có điểm đẹp)?**

- *(b) Trả lời:* Có, đây là rủi ro thật (đã nêu ở Q2). Với 24 điểm sweep, xác suất ít nhất một điểm "đẹp" xuất hiện do nhiễu ngẫu nhiên tăng lên so với chỉ thử 1 cấu hình. Chúng tôi giảm thiểu phần nào bằng cách: (1) mỗi điểm sweep tuy 1-seed nhưng cùng seed 42 cho tất cả 24 điểm (so sánh công bằng nội bộ), và (2) sau khi chọn ra 4 operating point từ sweep, chúng tôi **chạy lại độc lập 3-seed mới** cho đúng 5 cấu hình đó (không phải chỉ lấy lại số từ sweep) — nghĩa là số liệu Slide 11 là một phép kiểm chứng độc lập, không đơn thuần là con số tốt nhất trong 24 điểm đã thấy. Đây là điểm mạnh giảm nhẹ lo ngại multiple-comparison, dù không loại bỏ hoàn toàn nghi ngờ ở bước chọn λ ban đầu.

---

### Nhóm C — So sánh công bằng / vị trí trong literature

**Q10. Tại sao không so sánh với các phương pháp debiasing chuẩn trong literature (IPS, DICE, MACR, causal debiasing/PD)?**

- *(b) Trả lời:* Đây là giới hạn thật của phạm vi đồ án. Nhóm có khám phá hướng causal debiasing (Popularity Deconfounding) ở giai đoạn đầu nhưng quyết định dừng để tập trung vào phương pháp chính theo đúng yêu cầu đề bài. So sánh hiện tại (LightGCN, BPR-MF, MostPopular) chứng minh cải thiện so với backbone của chính nó và so với một baseline mạnh (BPR-MF), nhưng chưa định vị được so với state-of-the-art debiasing. Hướng mở rộng hợp lý: thêm ít nhất 1 baseline debiasing chuẩn (IPS dễ triển khai nhất) để so sánh trực tiếp.

**Q11. Bốn "component" trong slide — có ablation nào 3-seed tách riêng từng cái để chứng minh mỗi cái đóng góp độc lập không?**

- *(b) Trả lời:* Có ablation tách riêng (`train_all_popaware.py`: baseline/ile/degreeaug/degreeaug_cl/full) nhưng chạy 1 seed, dùng trong giai đoạn phát triển, không phải số liệu 3-seed trong Slide 11. Bốn operating point báo cáo chính thức đều bật đồng thời ILE+CL, chỉ khác cường độ λ/β — chứng minh hiệu quả của **tổ hợp**, chưa tách bạch hoàn toàn từng phần với độ tin cậy 3-seed. Bằng chứng gián tiếp về cơ chế: trong sweep, tăng λ_ILE một mình kéo TailRecall tăng đơn điệu; tăng λ_CL một mình kéo Coverage tăng đơn điệu — dấu hiệu mỗi thành phần có tác động thật.

**Q12. Recall@20 tăng từ 0.1287 lên 0.1338 (+4%) — con số này có ý nghĩa thực tế không, hay nằm trong biên độ nhiễu bình thường giữa các lần chạy LightGCN?**

- *(a) Vì sao hỏi:* +4% nghe nhỏ, cần chứng minh nó không phải "nhiễu vặt".
- *(b) Trả lời:* Với baseline 3-seed có std=0.0005 (rất nhỏ, ~0.4% của giá trị trung bình) và BEST có mean=0.1338±0.0030, khoảng cách giữa 2 mean (0.1287 vs 0.1338 = 0.0051) lớn hơn nhiều lần tổng 2 std cộng lại (0.0005+0.0030=0.0035) — nghĩa là chênh lệch vượt ngoài dao động ngẫu nhiên quan sát được giữa các seed. Về ý nghĩa thực tế: +4% Recall trong recommender system thường được coi là cải thiện đáng chú ý (nhiều paper SOTA công bố cải thiện 2-5% đã được xem là đóng góp có giá trị), đặc biệt khi đi kèm cải thiện lớn ở các trục fairness (TailRecall ×6.5).

**Q13. Contrastive Learning trong phương pháp của bạn lấy ý tưởng gần giống SGL (Self-supervised Graph Learning, Wu et al. 2021). Vậy đóng góp mới nằm ở đâu, khác gì so với việc áp dụng thẳng SGL lên LightGCN?**

- *(b) Trả lời:* Đúng là cơ chế contrastive view + edge dropout lấy cảm hứng từ SGL. Khác biệt chính: (1) SGL nguyên bản dùng dropout **ngẫu nhiên đều** (uniform) để tạo view, còn ở đây dropout được thiết kế **theo degree** (degree-aware), tức là có chủ đích thiên về giảm bias thay vì chỉ là regularizer chung chung; (2) SGL không có ILE (cân bằng loss head/tail) và không có popularity-aware negative sampling — hai thành phần này là phần bổ sung so với SGL; (3) mục tiêu của SGL là cải thiện accuracy/robustness nói chung, còn tổ hợp ở đây được thiết kế và đánh giá cụ thể cho bài toán giảm popularity bias, với bộ metric fairness riêng (TailRecall/Coverage/ARP/Exposure). Đóng góp không phải là phát minh một cơ chế hoàn toàn mới, mà là **tổ hợp có chủ đích 4 cơ chế bổ sung nhau** (mỗi cái tác động một điểm khác nhau trong vòng đời huấn luyện) cho đúng bài toán long-tail, cùng một pipeline thực nghiệm nghiêm ngặt (leakage-free, 3-seed) để đo đúng đánh đổi accuracy/fairness.

---

### Nhóm D — Thiết kế phương pháp / lựa chọn kỹ thuật

**Q14. Ngưỡng percentile 50/80 để chia head/middle/tail là cố định. Kết quả có nhạy với ngưỡng này không?**

- *(b) Trả lời:* Ngưỡng 50/80 (tail <P50, middle P50–P80, head >P80, tức head = top 20%) theo quy ước phổ biến trong literature long-tail recommendation. Chúng tôi **chưa sweep độ nhạy** của chính ngưỡng này (chỉ sweep λ_ILE, λ_CL, layers, β) — đây là một trục chưa kiểm tra, ghi nhận là hướng mở rộng.

**Q15. Rejection sampling cho negative (tối đa 20 lần) — nếu sau 20 lần vẫn trùng positive thì sao?**

- *(b) Trả lời:* Không có đảm bảo tuyệt đối — sau 20 lần vẫn giữ giá trị cuối cùng dù có thể trùng positive của user. Với β nhỏ (0–0.5) và catalog ~3500 item, xác suất này cực thấp (mỗi user chỉ tương tác với phần rất nhỏ catalog), nên ảnh hưởng thực nghiệm không đáng kể, nhưng về lý thuyết không phải zero. Có thể thay bằng loại trừ trực tiếp (set difference) nếu cần chặt chẽ tuyệt đối.

**Q16. Tại sao dùng BPR loss (pairwise) mà không dùng sampled softmax hoặc BCE (listwise/pointwise)?**

- *(b) Trả lời:* BPR là loss chuẩn của LightGCN gốc (Rendle et al. 2009; He et al. 2020 - LightGCN paper), giữ nguyên để đảm bảo so sánh công bằng với backbone — thay đổi loss nền sẽ làm nhiễu việc đánh giá đóng góp của 4 thành phần bổ sung (không biết cải thiện đến từ loss mới hay từ ILE/CL/NegSampling). Đây là lựa chọn có chủ đích để cô lập biến số cần đo, không phải bỏ sót.

**Q17. Vì sao dropout đối xứng (quyết định 1 lần trên mỗi cạnh vô hướng, mirror 2 chiều) mà không dropout độc lập từng chiều?**

- *(b) Trả lời:* Đồ thị LightGCN về bản chất là vô hướng (user-item tương tác 2 chiều để lan truyền). Nếu dropout độc lập từng chiều (user→item và item→user có thể khác nhau), ma trận kề sẽ **không còn đối xứng**, phá vỡ giả định `D^-1/2 A D^-1/2` chuẩn của LightGCN (chuẩn hoá symmetric normalize dựa trên giả định đồ thị vô hướng) — có thể dẫn đến hành vi lan truyền không kiểm soát được hoặc vi phạm tính chất toán học của phép chuẩn hoá. Dropout đối xứng đảm bảo view augmented vẫn là một đồ thị vô hướng hợp lệ.

**Q18. Tại sao chọn num_layers=2 làm điểm chính, trong khi sweep cho thấy K=3 đôi khi tốt hơn cho full method?**

- *(b) Trả lời:* Số liệu sweep cho thấy với **full method** (đủ 4 thành phần), K=3 nhỉnh hơn K=2 một chút ở Recall@20 (+0.0009) và Coverage@20 (+0.010) trung bình trên 12 điểm khớp — chênh lệch nhỏ, không lớn tới mức bắt buộc phải đổi. K=2 được chọn làm điểm chính vì: (1) chi phí tính toán thấp hơn (ít hơn 1 lớp lan truyền × 2 view cho CL = tiết kiệm đáng kể thời gian train), (2) với backbone thuần (không ILE/CL), K=2 rõ ràng tốt hơn K=3, nên chọn K=2 làm điểm chuẩn giữ nhất quán giữa baseline và phương pháp đề xuất. Đây là đánh đổi thực dụng (compute vs. cải thiện marginal), có ghi nhận rõ trong tài liệu đầy đủ, không phải bỏ sót.

**Q19. weight_decay cố định 1e-4, không nằm trong lưới sweep — vì sao?**

- *(b) Trả lời:* weight_decay là hyperparameter chuẩn của LightGCN gốc (giữ nguyên theo giá trị mặc định phổ biến trong literature LightGCN/NGCF), không phải tham số đặc trưng của phương pháp đề xuất (khác với λ_ILE, λ_CL, β là các tham số **mới do phương pháp này sinh ra**). Việc giới hạn phạm vi sweep vào các tham số mới giúp lưới tìm kiếm (24 điểm) khả thi về mặt thời gian GPU; sweep cả weight_decay sẽ nhân số điểm cần chạy lên nhiều lần.

**Q20. Lưới λ_ILE {0.1,0.5,1.0} và λ_CL {0.1,0.5} không đối xứng (3 giá trị vs 2 giá trị) — vì sao?**

- *(b) Trả lời:* Đây là lựa chọn thực dụng để giữ tổng số điểm sweep (24 = 2 layers × 3 λ_ILE × 2 λ_CL × 2 β) trong ngân sách thời gian GPU cho phép. λ_ILE được cho nhiều mức hơn vì đây là thành phần cốt lõi giải quyết trực tiếp mục tiêu chính (cân bằng loss head/tail), còn λ_CL đóng vai trò regularizer bổ trợ.

**Q21. Nhiệt độ τ trong InfoNCE cố định = 0.2, không sweep — CL có nhạy với τ không?**

- *(b) Trả lời:* Chưa kiểm tra độ nhạy của τ — đây là giá trị mặc định phổ biến trong literature contrastive learning (SimCLR, SGL đều dùng τ trong khoảng 0.1–0.2). Đây là một trục chưa sweep, ghi nhận như hướng mở rộng nếu được hỏi.

---

### Nhóm E — Metric & định nghĩa (đọc kỹ code `src/metrics.py`)

**Q22. Recall@K trong đồ án này — vì mỗi user chỉ có đúng 1 item test (leave-one-out), về bản chất đây có phải là Hit Rate@K (HR@K) không? Tại sao gọi là "Recall"?**

- *(a) Vì sao hỏi:* kiểm tra hiểu biết chính xác về công thức đo — đây là câu hỏi rất hay bị hỏi khi đọc kỹ `recall_at_k()`.
- *(b) Trả lời:* Chính xác, về mặt công thức, với đúng 1 item liên quan/user thì Recall@K = HR@K (đều là "test item có nằm trong top-K không", trung bình trên toàn bộ user). Cách gọi "Recall@K" theo đúng quy ước của paper gốc LightGCN/NCF (He et al.), nơi thuật ngữ "Recall@K" được dùng theo nghĩa này cho leave-one-out protocol — tôi giữ theo đúng quy ước của backbone gốc để có thể so sánh trực tiếp con số với các bài báo tham chiếu, dù về bản chất toán học nó trùng với Hit Rate@K.

**Q23. ARP@20 (Average Recommendation Popularity) — tính trên độ phổ biến thô (raw degree) hay có biến đổi gì không?**

- *(b) Trả lời:* Tính trên `log(1 + degree)`, không phải raw degree (`average_recommendation_popularity` dùng `torch.log1p`). Dùng log để nén phạm vi giá trị (một số item có degree lên tới hàng nghìn, log giúp tránh vài item cực phổ biến chi phối toàn bộ trung bình một cách quá mức) — đây là quy ước chuẩn trong literature ARP (Abdollahpouri et al.). Giá trị ARP báo cáo (~6.1–6.9) là thang log, không phải số lượt tương tác trung bình theo nghĩa đen.

**Q24. Coverage@20 = 0.538 nghĩa là gì cụ thể — tính trên toàn hệ thống hay trung bình từng user?**

- *(b) Trả lời:* Coverage@20 là **catalog coverage toàn hệ thống**: tỉ lệ item xuất hiện trong **ít nhất 1** danh sách top-20 của **bất kỳ** user nào, chia cho tổng số item trong catalog (`torch.unique(topk).numel() / num_items`). Coverage=0.538 nghĩa là 53.8% trong ~3500 item từng được gợi ý cho ít nhất một user nào đó trong top-20; 46.2% item còn lại **không bao giờ** được gợi ý cho bất kỳ ai. Đây không phải trung bình per-user (per-user sẽ luôn là 20/3500 rất nhỏ, không có ý nghĩa).

**Q25. TailRecall@20 chỉ tính trên các user có item test thuộc nhóm tail — vậy có bỏ sót user có test item là head/middle không? Số lượng user tail có đủ lớn để ổn định không?**

- *(b) Trả lời:* Đúng, `tail_recall_at_k()` chỉ lọc và tính trung bình trên tập con user mà `item_group[test_item] == 0` (tail); user có test item head/middle bị loại khỏi phép tính này (nhưng vẫn được tính trong Recall@20/NDCG@20 tổng thể). Với MovieLens-1M có phân phối lệch mạnh (theo định nghĩa tail = 50% item ít phổ biến nhất, nhưng số **lượt tương tác** rơi vào nhóm tail luôn ít hơn nhiều so với 50%), số user-test thuộc nhóm tail chắc chắn nhỏ hơn tổng số user, khiến TailRecall có phương sai cao hơn Recall tổng thể — đây là lý do tự nhiên khiến TailRecall dao động seed-to-seed lớn hơn (std tương đối lớn hơn so với Recall@20), điều này **đã thể hiện đúng** trong bảng Slide 11 (TailRecall có std tương đối cao nhất trong các metric).

**Q26. Tại sao không dùng các chỉ số công bằng phổ biến khác như Gini coefficient hoặc entropy trên phân phối gợi ý, thay vì chỉ Coverage/Exposure/ARP?**

- *(b) Trả lời:* Coverage, ARP, và Exposure-by-group là bộ metric phổ biến và được trích dẫn nhiều trong literature về popularity bias trong RecSys (Abdollahpouri et al., nhiều paper long-tail khác), đủ để mô tả đa chiều: Coverage (đa dạng danh mục), ARP (độ phổ biến trung bình được gợi ý), Exposure (tỉ lệ slot dành cho mỗi nhóm). Gini/entropy là lựa chọn hợp lệ khác nhưng chưa được tính trong đồ án này — đây là bổ sung khả thi nếu cần thêm góc nhìn định lượng về "độ công bằng phân phối".

---

### Nhóm F — Dữ liệu & tiền xử lý

**Q27. Vì sao chọn ngưỡng rating ≥ 4 làm positive, không phải ≥ 3 hoặc dùng toàn bộ rating có trọng số?**

- *(b) Trả lời:* rating≥4 là ngưỡng phổ biến trong các paper implicit-feedback trên MovieLens (bao gồm chính paper LightGCN gốc và NCF), giúp lọc ra tương tác thể hiện sở thích rõ ràng (rating 4-5/5) thay vì trung tính. Dùng ngưỡng này để nhất quán với cách backbone gốc được đánh giá trong literature, dễ so sánh chéo.

**Q28. Việc chia train/val/test là ngẫu nhiên hay theo thời gian? Vì sao cách chia ảnh hưởng đến kết quả?**

- *(b) Trả lời:* Chia theo **thời gian** (temporal leave-one-out): với mỗi user, sắp xếp tương tác theo timestamp, lấy tương tác **gần nhất** làm test, **kế gần nhất** làm val, còn lại làm train (`src/data.py: split_per_user`). Đây là protocol chuẩn (giống NCF/LightGCN), mô phỏng đúng bài toán thực tế "dự đoán tương tác tiếp theo dựa trên lịch sử quá khứ" — nghiêm ngặt hơn chia ngẫu nhiên (random split), vì random split có thể vô tình để lộ thông tin từ tương lai vào train (leakage theo thời gian), khiến kết quả bị đánh giá quá lạc quan.

**Q29. Item mới hoàn toàn không có tương tác trong train (cold-start, degree=0) được xử lý ra sao trong các cơ chế dựa trên degree (ILE, dropout, negative sampling)?**

- *(b) Trả lời:* Với `item_degree=0`: nhóm popularity sẽ rơi vào tail (dưới mọi percentile dương); `compute_degree_aware_dropout_probs` xử lý riêng trường hợp `max_degree==0` (trả về p_min đồng loạt) và dùng `log(1+deg)` nên deg=0 vẫn cho giá trị hợp lệ (log(1)=0) chứ không lỗi chia-cho-0; `build_neg_probs` dùng `deg^β`, nếu β>0 thì deg=0 → trọng số sampling = 0 (item cold-start gần như không bao giờ được chọn làm negative). Tuy nhiên MovieLens-1M là dataset "đóng" (không có luồng item mới liên tục như hệ thống thực), nên hiện tượng cold-start thực tế trong tập dữ liệu này rất hạn chế — đây là giới hạn về tính đại diện của dataset, không phải lỗi cơ chế.

---

### Nhóm G — Tính mới / đóng góp khoa học

**Q30. Về cơ bản đây có phải là ghép 3-4 kỹ thuật đã biết (loss balancing kiểu ILE, contrastive learning kiểu SGL, negative sampling theo degree) lại với nhau, không có gì thực sự mới?**

- *(b) Trả lời:* Đúng là từng cơ chế riêng lẻ không phải phát minh hoàn toàn mới — đóng góp nằm ở: (1) **tổ hợp có chủ đích** 4 cơ chế tác động vào 4 điểm khác nhau của vòng đời huấn luyện (hàm mục tiêu / cấu trúc đồ thị-view / không gian embedding / tín hiệu đối nghịch) thay vì chọn ngẫu nhiên nhiều regularizer; (2) sửa một lỗi dấu quan trọng trong công thức ILE ban đầu (bản đầu dùng `head_loss - tail_loss` không bình phương, gây phản tác dụng) — quá trình chẩn đoán và sửa lỗi này bản thân là một đóng góp thực nghiệm; (3) xây dựng pipeline đánh giá nghiêm ngặt (leakage-free, checkpoint/resume, 3-seed, sweep có kiểm soát) áp dụng nhất quán cho toàn bộ ablation — nhiều đồ án sinh viên không làm tới mức này; (4) phân tích frontier tường minh giữa accuracy và fairness, cho phép chọn operating point theo nhu cầu thay vì chỉ báo cáo 1 con số duy nhất.

**Q31. Nếu chỉ cần re-rank sau khi có kết quả (post-hoc calibration/re-ranking, không cần train lại model) cũng có thể tăng Coverage tương tự với chi phí thấp hơn nhiều — tại sao phải tốn công train lại toàn bộ model?**

- *(b) Trả lời:* Đúng là re-ranking (ví dụ: MMR, calibrated re-ranking) là một hướng cạnh tranh, chi phí thấp hơn vì không cần train lại. Tuy nhiên re-ranking chỉ tác động ở bước cuối (sắp xếp lại top-K đã có), không thay đổi được **embedding gốc** — nếu embedding gốc đã học lệch mạnh về phía item phổ biến (như LightGCN baseline cho thấy), re-ranking chỉ "che" triệu chứng ở đầu ra chứ không sửa gốc rễ biểu diễn. Phương pháp đề xuất tác động vào quá trình học biểu diễn, có tiềm năng cải thiện chất lượng gợi ý tail-item một cách bền vững hơn (không chỉ là latency thêm bước rerank). Đây là so sánh chưa được thực nghiệm trực tiếp trong đồ án — nếu hỏi sâu, cần thừa nhận: **chưa so sánh thực nghiệm với một baseline re-ranking**, đây là hướng mở rộng hợp lý.

---

### Nhóm H — Thực tiễn / triển khai / tác động

**Q32. Contrastive Learning cần lan truyền qua 2 view mỗi bước — chi phí tính toán tăng bao nhiêu so với LightGCN thuần? Có số liệu thời gian train cụ thể không?**

- *(b) Trả lời:* Về lý thuyết, mỗi bước cần thêm 2 lần lan truyền (2 view) so với 1 lần của LightGCN thuần → tăng chi phí forward pass mỗi epoch. Chúng tôi **chưa đo và báo cáo con số cụ thể** (giây/epoch, GPU-memory) trong tài liệu hiện tại — đây là khoảng trống cần bổ sung. *(Chuẩn bị trước khi bảo vệ: lấy 1-2 con số thời gian thật từ `logs/popaware/*.log` — ví dụ so sánh thời gian/epoch giữa cấu hình baseline và full — để trả lời chủ động nếu bị hỏi trực tiếp thay vì nói "chưa đo".)*

**Q33. Phương pháp chỉ thử trên MovieLens-1M — có bằng chứng tổng quát hoá sang dataset khác (thưa hơn, catalog lớn hơn) không?**

- *(b) Trả lời:* Chưa. Đây là giới hạn về phạm vi đã nêu rõ. Kiểm chứng trên dataset thứ 2 (Gowalla/Amazon, mật độ thưa hơn) là hướng phát triển được đề xuất nhưng chưa có số liệu.

**Q34. Việc cố ý đẩy mạnh gợi ý item tail có luôn tốt cho người dùng không? Nếu item tail ít phù hợp hơn (relevance thấp hơn) thì việc tăng TailRecall có làm giảm trải nghiệm/hài lòng thực tế của user không?**

- *(a) Vì sao hỏi:* câu hỏi về đánh đổi đạo đức/kinh doanh — rất hay gặp trong phản biện fairness-in-RecSys.
- *(b) Trả lời:* Đây là căng thẳng thật (accuracy-fairness trade-off), và chính vì vậy đồ án không chỉ báo cáo 1 điểm vận hành mà cung cấp **một frontier** (accuracy/BEST/high-tail/fairness) để người vận hành hệ thống chọn theo mục tiêu kinh doanh cụ thể — ví dụ ưu tiên accuracy (config "accuracy") nếu quan tâm trải nghiệm ngắn hạn, hoặc ưu tiên fairness nếu mục tiêu là đa dạng hoá danh mục dài hạn (khám phá nội dung mới, hỗ trợ nhà sản xuất nội dung nhỏ). Quan trọng: ở điểm "BEST", Recall/NDCG được xác nhận **tăng nhẹ chứ không giảm** so với baseline — nghĩa là ít nhất tại điểm vận hành được đề xuất chính, không có bằng chứng đánh đổi giảm relevance để đổi lấy fairness; đây là lý do BEST được chọn làm điểm khuyến nghị chính thay vì "fairness". Tuy nhiên, việc "relevance đo bằng Recall/NDCG" có phản ánh đúng "hài lòng thực tế" hay không thì cần một nghiên cứu người dùng thật (online A/B test) — đồ án hiện tại chỉ dừng ở offline evaluation, đây là giới hạn cần thừa nhận.

**Q35. Đánh giá full-ranking (chấm điểm toàn bộ ~3500 item cho mỗi user) khả thi ở quy mô nhỏ. Với catalog hàng triệu item trong thực tế, cách tiếp cận này có còn khả thi không?**

- *(b) Trả lời:* Full-ranking (tính điểm toàn bộ catalog cho mỗi user rồi lấy top-K) có độ phức tạp O(num_users × num_items), khả thi ở quy mô ML-1M (~6000 user × ~3500 item) nhưng không khả thi trực tiếp ở quy mô hàng triệu item. Trong triển khai thực tế, bước này thường được thay bằng **truy vấn xấp xỉ láng giềng gần nhất (ANN)** trên embedding (FAISS, ScaNN...) để rút ngắn thời gian truy vấn xuống sub-linear. Phương pháp đề xuất không thay đổi bản chất bước inference (vẫn là dot-product giữa embedding user/item), nên về nguyên tắc tương thích với ANN retrieval như LightGCN gốc — nhưng điều này chưa được thử nghiệm/đo trong đồ án.

---

### Nhóm I — Câu hỏi tình huống / bẫy logic (rất nên luyện trước)

**Q36. Nếu tôi nói "cải thiện Recall của bạn chỉ +4%, quá nhỏ để quan tâm" — bạn phản biện thế nào?**

- *(b) Trả lời:* Ba luận điểm: (1) +4% Recall đi kèm với TailRecall tăng ×6.5 và Coverage +35% — nghĩa là đạt được cải thiện fairness rất lớn **mà không phải đánh đổi** accuracy (thực tế còn tăng nhẹ), đây mới là điểm đáng chú ý, không phải bản thân con số +4% đứng một mình; (2) chênh lệch +4% vượt xa dao động seed-to-seed quan sát được (baseline std chỉ ~0.4%), nên đây là tín hiệu thật, không phải nhiễu; (3) trong nhiều benchmark recommender-system đã bão hoà (mature benchmark như MovieLens với backbone đã mạnh như LightGCN), cải thiện 2-5% Recall thường được xem là có ý nghĩa và được chấp nhận công bố trong literature.

**Q37. Kết quả nào, nếu quan sát được, sẽ khiến bạn phải kết luận là phương pháp KHÔNG hoạt động?**

- *(a) Vì sao hỏi:* câu hỏi kiểm tra tư duy khoa học (falsifiability) — một câu hỏi "triết học phương pháp luận" hay gặp ở hội đồng khó tính.
- *(b) Trả lời:* Nếu tăng λ_ILE/λ_CL mà các metric fairness (TailRecall, Coverage) **không** thay đổi theo hướng dự đoán (không tăng đơn điệu), hoặc nếu accuracy (Recall/NDCG) giảm mạnh (>10-20%) để đổi lấy một chút cải thiện fairness, hoặc nếu khoảng mean±std giữa baseline và PopAware chồng lấn nhau qua 3 seed — bất kỳ điều nào trong số này sẽ là bằng chứng phương pháp không hoạt động như kỳ vọng. Trên thực tế, chúng tôi **đã từng quan sát đúng kịch bản thất bại này** ở phiên bản ILE đầu tiên (trước khi sửa lỗi dấu): tăng cường độ ILE khi đó khiến MỌI trục cùng tệ đi (kể cả TailRecall cũng giảm) — đây chính là dấu hiệu kinh điển của một phương pháp bị lỗi/không hoạt động, và là bằng chứng cho thấy chúng tôi có tiêu chí rõ ràng để nhận biết thất bại, không chỉ paper-hoá mọi kết quả thành công.

**Q38. Nếu đặt λ_ILE hoặc λ_CL rất lớn (ví dụ 100), điều gì sẽ xảy ra? Model có suy biến (degenerate) không?**

- *(b) Trả lời:* Về lý thuyết: λ_ILE cực lớn sẽ khiến model ưu tiên tuyệt đối việc cân bằng loss head/tail, có thể hi sinh hoàn toàn khả năng dự đoán chính xác (model có thể học cách làm cho loss head cũng tệ đi để "bằng" loss tail, thay vì kéo tail tốt lên — vì công thức chỉ phạt *độ chênh lệch*, không ép buộc chiều cải thiện). λ_CL cực lớn có thể khiến model chỉ tối ưu InfoNCE (đẩy mọi embedding về phân bố đều/uniform trên hypersphere) mà bỏ qua tín hiệu BPR, dẫn tới sụp giảm accuracy nghiêm trọng. Chúng tôi **chưa thử nghiệm các giá trị cực trị này** (lưới sweep dừng ở 1.0 cho λ_ILE và 0.5 cho λ_CL) — đây là thí nghiệm bổ sung hợp lý để chứng minh hiểu biết đầy đủ về vùng hoạt động an toàn của siêu tham số, nên chuẩn bị tinh thần thừa nhận đây là điểm chưa kiểm tra nếu bị hỏi.

**Q39. Nếu chỉ được chọn MỘT con số duy nhất để thuyết phục hội đồng phương pháp này đáng giá, bạn chọn con số nào và vì sao?**

- *(b) Trả lời:* TailRecall@20 tăng từ 0.0050 lên 0.0324 (×6.5) tại điểm BEST, **đồng thời** Recall@20 vẫn tăng nhẹ (không đánh đổi) — đây là con số duy nhất thể hiện rõ nhất "đạt được mục tiêu kép" (giảm bias + không hi sinh accuracy) mà đề bài yêu cầu, và có khoảng tin cậy 3-seed không chồng lấn baseline.

---

### Nhóm J — Kỹ thuật triển khai & hành vi siêu tham số (đào sâu code)

**Q40. Slide 8 nói β (negative sampling) đánh đổi TailRecall lấy Coverage tại BEST vs high-tail — vậy β có tác động "đơn điệu" (càng tăng càng tốt cho fairness) giống λ_ILE và λ_CL không?**

- *(a) Vì sao hỏi:* đây là câu hỏi tôi (giảng viên) tự kiểm tra lại toàn bộ 12 cặp khớp trong sweep (giữ L, λ_ILE, λ_CL cố định, chỉ đổi β 0→0.5) và phát hiện: **λ_ILE→TailRecall monotonic ở cả 8/8 bộ ba giá trị**; **λ_CL→Coverage monotonic ở cả 12/12 cặp**; nhưng **β→TailRecall và β→Coverage đều KHÔNG đơn điệu** (mixed: 6 lên/6 xuống cho TailRecall; hướng tăng/giảm Coverage phụ thuộc vào λ_CL đang ở mức nào). Đây là phát hiện quan trọng khiến tôi phải sửa lại bullet trên Slide 8 cho chính xác.
- *(b) Trả lời:* Không, β là tham số có hành vi phức tạp nhất trong 3 tham số mới. Không giống λ_ILE (monotonic rõ ràng với TailRecall trên toàn bộ lưới) và λ_CL (monotonic rõ ràng với Coverage trên toàn bộ lưới), tác động của β lên TailRecall/Coverage phụ thuộc vào việc λ_CL đang ở mức nào — cho thấy β **tương tác** với λ_CL thay vì tác động độc lập, cộng dồn tuyến tính. Bằng chứng chắc chắn duy nhất và nhất quán về β mà tôi có thể bảo vệ trước hội đồng là so sánh trực tiếp 3-seed BEST (β=0.5) vs high-tail (β=0), giữ nguyên λ_ILE=1.0, λ_CL=0.1: β=0.5 đánh đổi một phần TailRecall (0.040→0.032) để lấy Coverage cao hơn (0.499→0.538). Tôi tránh khẳng định "β luôn cải thiện fairness" vì số liệu không ủng hộ điều đó một cách tổng quát.
- *(c) Nếu hỏi "vậy tại sao vẫn dùng β trong công thức nếu nó không ổn định":* Vì tại **đúng vùng vận hành được chọn** (λ_ILE=1.0, λ_CL=0.1 — vùng của BEST/high-tail), β vẫn cho một đánh đổi có ý nghĩa và dùng được (dịch chuyển điểm vận hành theo trục Coverage). Việc nó không đơn điệu trên *toàn bộ* lưới 24 điểm là một quan sát khoa học trung thực cần nêu ra, không phải lý do để loại bỏ tham số — nó cho thấy không gian siêu tham số có tương tác phức tạp hơn one giả định ban đầu, và là hướng phân tích sâu hơn (ví dụ interaction plot giữa β và λ_CL) nếu tiếp tục nghiên cứu.

**Q41. Code có rất nhiều đoạn `torch.clamp`, thêm epsilon, và comment "CRITICAL FIX" trong `config.py`/`ile_losses.py`. Đây có phải dấu hiệu bạn từng gặp lỗi NaN/Inf hoặc bất ổn số học trong lúc train không? Nguyên nhân là gì?**

- *(a) Vì sao hỏi:* các guard này (clamp score_diff về [-50,50], clamp sigmoid về [1e-8,1-1e-8], kiểm tra `torch.isfinite`) không xuất hiện ngẫu nhiên — chúng luôn là dấu vết của một lần debug thực sự.
- *(b) Trả lời:* Đúng, các guard này được thêm sau khi gặp hiện tượng NaN/Inf trong quá trình phát triển ILE loss — nguyên nhân gốc là `-log(sigmoid(x))` có thể tràn số khi `x` (chênh lệch điểm số pos-neg) rất âm (sigmoid tiến về 0, log(0) = -inf). Việc clamp `score_diff` vào [-50,50] trước khi đưa vào sigmoid, và clamp chính sigmoid output vào [1e-8, 1-1e-8] trước khi lấy log, là kỹ thuật chuẩn để đảm bảo ổn định số học (numerical stability) mà không đổi giá trị loss một cách có ý nghĩa (50 đã là vùng bão hoà của sigmoid). Việc kiểm tra `torch.isfinite` cuối mỗi hàm loss và trả về 0 nếu phát hiện NaN là một lớp phòng thủ bổ sung, đảm bảo một batch bất thường không làm hỏng toàn bộ quá trình train nhiều giờ.

**Q42. LightGCN dùng một bảng embedding chung cho cả user và item (item được đánh index bằng cách offset thêm `num_users`) thay vì 2 bảng embedding riêng. Tại sao thiết kế vậy, có rủi ro gì không?**

- *(b) Trả lời:* Đây là thiết kế chuẩn của LightGCN gốc, không phải tuỳ biến riêng: dùng một bảng embedding duy nhất `nn.Embedding(num_users+num_items, dim)` giúp việc lan truyền qua ma trận kề `A_norm` (kích thước (num_users+num_items)×(num_users+num_items)) trở thành một phép nhân ma trận-thưa duy nhất, đơn giản hơn nhiều so với việc phải tách riêng cập nhật user-embedding và item-embedding qua từng bước lan truyền (như cách tiếp cận của NGCF). Rủi ro về mặt lý thuyết: user và item chia sẻ cùng một không gian tham số (cùng chiều, cùng phân phối khởi tạo `std=0.1`) dù về ngữ nghĩa là 2 loại thực thể khác nhau — nhưng đây chính xác là thiết kế đã được kiểm chứng rộng rãi trong literature (LightGCN, NGCF), không phải điểm yếu riêng của đồ án này.

**Q43. Tại sao dùng lan truyền trung bình có trọng số cố định (symmetric-normalized mean) mà không dùng cơ chế attention (như GAT) để mô hình tự học trọng số quan trọng giữa các neighbor?**

- *(b) Trả lời:* Đây là lựa chọn thiết kế cốt lõi của LightGCN (He et al. 2020), xuất phát từ phát hiện thực nghiệm của chính paper đó: phần lớn độ phức tạp của GCN gốc (feature transform + non-linear activation + attention) **không đóng góp** vào chất lượng collaborative filtering, thậm chí có thể gây overfitting; loại bỏ hết các thành phần này và chỉ giữ lan truyền tuyến tính + trung bình theo lớp giúp LightGCN vừa đơn giản hơn vừa cho kết quả tốt hơn NGCF/GAT-based baseline trên nhiều benchmark. Vì mục tiêu của đồ án là bổ sung 4 cơ chế giảm bias LÊN TRÊN một backbone đã được công nhận, chúng tôi giữ nguyên LightGCN thay vì đổi sang attention-based GNN để không làm nhiễu việc đánh giá đóng góp của 4 thành phần mới (nếu đổi backbone, không biết cải thiện đến từ backbone mới hay từ 4 thành phần).

**Q44. `torch.backends.cudnn.deterministic = True` được bật để đảm bảo tái lập — điều này có đánh đổi về tốc độ huấn luyện không?**

- *(b) Trả lời:* Có — bật `cudnn.deterministic=True` và tắt `cudnn.benchmark` (trong `set_seed()`) thường khiến một số kernel cuDNN chạy chậm hơn so với chế độ mặc định (cuDNN tự động chọn thuật toán nhanh nhất nhưng không đảm bảo tái lập bit-for-bit giữa các lần chạy). Đây là đánh đổi có chủ đích: ưu tiên khả năng tái lập (một yêu cầu quan trọng của đồ án, đã nêu ở Slide 9) hơn là tốc độ tối đa — hợp lý cho một đồ án nghiên cứu cần so sánh công bằng giữa các cấu hình, dù trong môi trường production thực tế có thể cân nhắc đánh đổi ngược lại.

---

## 5. Kịch bản thuyết trình từng Slide (tiếng Việt)

> Cách dùng: đây là bản thảo lời nói (talk-track) cho từng slide 5–13, viết theo đúng nội dung/số liệu/công thức thực tế đang có trên slide (đối chiếu với bản dump text ở mục 2). Đọc to thành tiếng vài lần để quen nhịp, sau đó chuyển sang nói bằng ý riêng — đừng học thuộc lòng từng chữ, hội đồng dễ nhận ra giọng "đọc học thuộc". Thời lượng gợi ý mỗi slide: 60–90 giây (trừ Slide 11 khoảng 90–120 giây vì nhiều số liệu). Tổng thời lượng phần này: khoảng 11–13 phút.

### Slide 5 — Method Overview (~60-75s)

> "Kính thưa quý thầy cô, sau phần đặt vấn đề, em xin trình bày phương pháp đề xuất. Slide này là bản đồ tổng quan của toàn bộ phương pháp Popularity-Aware LightGCN.
>
> Ở góc trên bên trái là đồ thị hai phía user–item — hình minh hoạ user là chấm xanh, item là chấm xám, riêng item màu đỏ to hơn là item phổ biến hơn, tức có degree cao hơn trong tập train. Đồ thị này chỉ được xây dựng từ các cạnh tương tác trong tập TRAIN.
>
> Đồ thị được đưa vào LightGCN Backbone — mạng lan truyền đồ thị K lớp, dùng chuẩn hoá đối xứng, lấy trung bình embedding qua các lớp. Đây là backbone gốc, em giữ nguyên không đổi.
>
> Từ backbone này, em bổ sung 4 thành phần — đánh số (1) đến (4) — và quan trọng là **bốn thành phần này bật/tắt độc lập với nhau** để phục vụ ablation: (1) Item Loss Equalization tác động lên hàm mục tiêu; (2) Degree-Aware Graph Augmentation tạo view phụ trợ cho contrastive learning; (3) Contrastive Learning kiểu InfoNCE tác động lên không gian embedding; (4) Popularity-aware Negative Sampling tác động lên tín hiệu học đối nghịch.
>
> Bốn thành phần hợp thành một hàm mục tiêu chung — Joint Objective — em sẽ trình bày công thức cụ thể ở slide 9. Và toàn bộ quá trình huấn luyện tuân theo một protocol chống rò rỉ dữ liệu: chọn model theo tập validation, tập test chỉ chấm điểm đúng một lần ở cuối.
>
> Slide này chỉ là bản đồ tổng quan — em sẽ đi vào chi tiết từng thành phần ở 4 slide tiếp theo."

### Slide 6 — Item Loss Equalization (~75-90s)

> "Bắt đầu với thành phần thứ nhất: Item Loss Equalization, viết tắt ILE.
>
> Bên trái là sơ đồ khi KHÔNG có ILE, tức λ_ILE = 0. Loss trung bình của nhóm tail — thanh màu đỏ, cao hơn — luôn cao hơn hẳn loss của nhóm head — thanh be, thấp hơn. Nghĩa là mô hình học nhóm head tốt hơn hẳn nhóm tail, dẫn tới gợi ý bị thiên lệch nặng về phía item phổ biến.
>
> Bên phải, khi BẬT ILE, hai thanh loss head và tail được kéo về gần bằng nhau.
>
> Cơ chế nằm ở công thức phía dưới: loss BPR từng cặp là âm log-sigmoid của hiệu điểm số dương trừ điểm số âm — công thức chuẩn. Và ILE penalty là bình phương của hiệu số giữa loss trung bình nhóm head và loss trung bình nhóm tail.
>
> Ba điểm em muốn nhấn mạnh: Một, phân nhóm head/middle/tail dựa trên percentile degree trong tập train — tail dưới P50, middle P50 đến P80, head trên P80 tức 20% phổ biến nhất. Hai, công thức chỉ tính khi batch có cả head lẫn tail. Ba — quan trọng nhất — vì có bình phương, giá trị luôn không âm, không bị lật dấu, nên tối thiểu hoá nó sẽ kéo loss tail xuống gần loss head, chứ không đẩy ngược lại.
>
> Em cũng xin lưu ý, biểu đồ cột này chỉ là sơ đồ minh hoạ cơ chế, không phải số liệu đo thực tế — số liệu thật ở slide 11."

### Slide 7 — Degree-Aware Augmentation + Contrastive Learning (~90-105s)

> "Tiếp theo là thành phần 2 và 3, trình bày chung một slide vì liên quan chặt chẽ với nhau.
>
> Bên trái là Degree-Aware Graph Augmentation: xác suất một cạnh bị dropout khi tạo view augmented tỉ lệ thuận với độ phổ biến của item — item càng phổ biến càng dễ bị drop. Trong hình, cạnh nối tới item màu đỏ bị đánh dấu 'dropped'. Công thức: xác suất drop bằng p_min cộng (p_max trừ p_min) nhân tỉ lệ log(1+degree) trên log(1+degree lớn nhất).
>
> Bên phải là Contrastive Learning kiểu InfoNCE. Từ đồ thị gốc, em tạo 2 view đã dropout độc lập — View 1 và View 2 — lan truyền qua CÙNG một encoder LightGCN. Với cùng một node, embedding z1 (view 1) và z2 (view 2) được kéo lại gần nhau — mũi tên xanh 'pull' — trong khi bị đẩy xa các node khác trong cùng batch — mũi tên xám đứt nét 'push'.
>
> Và đây là điểm em muốn nhấn mạnh, đã ghi rõ ngay trên slide: trong TẤT CẢ kết quả báo cáo, dropout theo degree này CHỈ dùng để tạo 2 view cho Contrastive Learning — đồ thị chính dùng để tính điểm xếp hạng cuối cùng KHÔNG bị thay đổi. Em chủ động nói rõ điều này để tránh gây hiểu lầm khi thầy cô đọc tên 'Graph Augmentation'."

### Slide 8 — Popularity-aware Negative Sampling (~75-90s)

> "Thành phần thứ 4: Popularity-aware Negative Sampling.
>
> Trong BPR truyền thống, negative được chọn ngẫu nhiên đều. Ở đây, em đổi phân phối lấy mẫu negative tỉ lệ thuận với degree mũ β — item càng phổ biến càng dễ bị chọn làm negative, bị đẩy điểm số xuống mạnh hơn. Biểu đồ cột: item xếp theo độ phổ biến giảm dần, cột tím đậm bên trái có trọng số lấy mẫu cao hơn.
>
> Ba điểm: positive vẫn lấy đều từ train; β=0 tương đương lấy mẫu đều, an toàn, không đổi hành vi gốc; thực nghiệm dùng β thuộc {0, 0.5}.
>
> Bằng chứng thực nghiệm cụ thể: so sánh BEST và high-tail — hai cấu hình có λ_ILE, λ_CL giống hệt nhau, chỉ khác β. Khi β tăng từ 0 lên 0.5, TailRecall giảm nhẹ từ 0.040 xuống 0.032, nhưng Coverage tăng từ 0.499 lên 0.538. Điều này cho thấy β không đơn thuần 'càng tăng càng tốt cho mọi chỉ số' — nó dịch chuyển điểm vận hành theo hướng tăng đa dạng danh mục, chứ không cải thiện đồng loạt mọi trục công bằng."

### Slide 9 — Joint Objective & Leakage-free Protocol (~90-105s)

> "Sau khi trình bày cả 4 thành phần, slide này tổng hợp thành một hàm mục tiêu và mô tả quy trình huấn luyện.
>
> Công thức trên: L tổng bằng loss BPR gốc, cộng λ_ILE nhân loss ILE, cộng λ_CL nhân loss Contrastive, cộng weight-decay nhân chuẩn L2 bình phương của tham số. Bốn thành phần bật/tắt độc lập qua các cờ use_ile, aug_main, use_cl, và hệ số β.
>
> Bên trái là Leakage-free Evaluation Protocol: đồ thị lan truyền chỉ dùng cạnh train; model chọn theo Recall@K trên validation bằng early-stopping; tập test chỉ chấm điểm đúng một lần, ở cuối cùng.
>
> Bên phải là hạ tầng tái lập: checkpoint lưu hai dạng — latest để resume, best theo validation; mỗi epoch ghi log đầy đủ ra file và CSV lịch sử.
>
> Em xin nói rõ một điểm quan trọng về phạm vi chữ 'leakage-free': điều này đúng cho việc CHỌN CHECKPOINT trong một lần train cụ thể. Còn việc chọn BỘ SIÊU THAM SỐ — chọn λ và β nào cho 4 operating point — hiện dùng một quy tắc frontier tính trên số liệu tập test qua toàn bộ lưới sweep. Em sẽ nói rõ hơn ở slide sau, và nêu đây như một hạn chế cần cải thiện ở slide 13."

### Slide phụ — Sơ đồ chi tiết đầy đủ "Popularity-Aware LightGCN (Our Proposed Model)" — bản ĐÃ SỬA, "Two Parallel Propagation Branches" (~4-4.5 phút)

> ✅ Đây là script cho **bản đã sửa** (subtitle "Two Parallel Propagation Branches for Accuracy and Popularity-Bias Mitigation"), thay thế script cũ mô tả bản gốc còn lỗi `l̄_item`. Dùng bản này nếu trình bày như 1 slide "kiến trúc chi tiết" độc lập, hoặc khi giảng viên yêu cầu xem lại toàn cảnh sau slide 6-9.

> "Sơ đồ này thể hiện toàn bộ phương pháp đề xuất trong một hình. Điểm mấu chốt nằm ngay ở phụ đề: **'Two Parallel Propagation Branches'** — hai nhánh lan truyền song song, không phải một chuỗi tuần tự duy nhất. Đây là nguyên tắc giải thích được gần như mọi mũi tên trên hình.
>
> **Khối ① — Original Bipartite Graph:** dữ liệu vào là ma trận tương tác ẩn user-item (1=đã tương tác, 0=chưa quan sát), dựng thành đồ thị hai phía chỉ từ cạnh TRAIN — 6.034 user, 3.533 item, 563.204 cạnh. Chú thích nhỏ dưới hình: mũi tên trên hình chỉ minh hoạ chiều tương tác, còn cạnh dùng để lan truyền thật là vô hướng, hai chiều. Khung Popularity Groups chia 3.533 item theo degree: Tail 1.755 (50% thấp nhất), Middle 1.071 (30% kế tiếp), Head 707 (20% cao nhất).
>
> **Nhánh A — Main path, đồ thị gốc không dropout:** đồ thị gốc qua đúng 1 lần LightGCN Backbone — công thức chuẩn hoá đối xứng E^(k+1)=Ã·E^(k), Ã=D^(-1/2)AD^(-1/2), rồi lấy trung bình cộng các lớp ra 'Original embeddings'. Từ đó, khối ④ Scoring tính s(u,i)=⟨e_u,e_i⟩ cho cặp positive i⁺ và negative j⁻ — chú thích rõ 'Scores use original embeddings only'.
>
> **Nhánh B — Auxiliary path, chỉ phục vụ Contrastive Learning:** khối ② Degree-Aware Augmentation tạo 2 view bị dropout — xác suất drop cạnh tăng theo độ phổ biến item, công thức p_i=p_min+(p_max-p_min)·log(1+deg_i)/log(1+deg_max), khoảng [0.1,0.4] — và quan trọng: 2 view này được lấy mẫu độc lập lại ở MỖI bước huấn luyện, không cố định. Khối ③ LightGCN Backbone chạy riêng 2 lần trên 2 view này, ra 2 bộ embedding, đi THẲNG vào Contrastive Loss — không qua khối ④ Scoring. Tổng cộng mỗi bước có 3 lần lan truyền: 1 cho nhánh A, 2 cho nhánh B.
>
> **Khối ⑤ — Training Objectives, 4 thành phần loss:** Một, BPR Loss chuẩn. Hai, Item Loss Equalization: L_ILE=(l̄_head−l̄_tail)², bình phương hiệu loss trung bình nhóm head và tail, kéo tail xuống gần head. Ba, Contrastive Loss: tổng 2 InfoNCE riêng cho user và item, và đây là InfoNCE MỘT CHIỀU — view 1 làm query, view 2 làm key, không phải bản đối xứng 2 chiều. Bốn, L2 Regularization: tính trên đúng embedding gốc (ego, layer 0) của user/positive/negative-item trong batch, chia cho kích thước batch — không phải toàn bộ tham số mô hình. Tất cả cộng lại thành Total Loss: L=L_BPR+λ_ILE·L_ILE+λ_CL·L_CL+wd·L_reg.
>
> **Khối ⑥ — Popularity-aware Negative Sampling:** xác suất chọn negative tỉ lệ với degree mũ β — β=0 là lấy mẫu đều, β>0 thiên vị item phổ biến. Mũi tên cam đi từ đây vào BPR Loss, đúng thứ tự thực thi: lấy mẫu negative trước, rồi mới tính điểm và loss.
>
> **Khối ⑦ — Top-K Inference, phần em nhấn mạnh nhất:** khi suy luận thật, tính điểm bằng đúng đồ thị GỐC không dropout — dùng lại embedding của Nhánh A, không liên quan gì đến 2 view augment ở Nhánh B. Che các item đã thấy trong train (và cả val khi đang đánh giá test), xếp hạng phần còn lại, trả về đúng Top-20.
>
> Hai khung phụ cuối: Key Hyperparameters tách rõ lưới sweep thật (λ_ILE∈{0.1,0.5,1.0}, λ_CL∈{0.1,0.5}, β∈{0,0.5}) với cấu hình baseline riêng (λ_ILE=λ_CL=0); và khung Key idea tóm tắt lại 3 ý cốt lõi: loss chính dùng đồ thị sạch, view augment chỉ phục vụ contrastive learning, ILE cân bằng head-tail còn negative sampling kiểm soát exposure."

*(Lưu ý khi trả lời Q&A: khối ⑦ tự nói rõ "using the original (no-dropout) graph" — bằng chứng trực quan tốt nhất cho câu trả lời Q1/Q40 về việc Degree-Aware Augmentation không chạm vào đồ thị suy luận cuối cùng. Cấu trúc "hai nhánh song song" trên hình cũng chính là câu trả lời trực quan cho câu hỏi "CL có đi qua Scoring không" — nhìn hình là thấy ngay câu trả lời là KHÔNG.)*

#### Ultra-review: đối chiếu từng công thức trong hình 7-panel với đúng dòng code

> ✅ **Cập nhật: bản vẽ lại đã được xác nhận đúng.** Toàn bộ các điểm A-E trong `Architecture_Diagram_Fix_Instructions.md` đã được áp dụng vào bản vẽ mới (subtitle "Two Parallel Propagation Branches..."), đối chiếu 1-1 không còn lỗi bắt buộc. Nội dung rà soát chi tiết dưới đây giữ lại làm hồ sơ tham khảo/lịch sử.

> Rà soát ở mức tối đa: mở trực tiếp `src/losses.py`, `src/popaware_training.py` (dòng 96-101, 193-201, 254-271), `src/ile_losses.py` để đối chiếu từng ký hiệu. Kết quả chia 3 nhóm: **Đúng khớp 100%**, **Sai cần sửa**, **Đúng nhưng cần nói chính xác hơn nếu bị hỏi xoáy**.

**✅ Đúng khớp 100% với code:**

| Công thức trên hình | Đối chiếu code |
|---|---|
| `|U|=6.034, |I|=3.533, Train=563.204 edges` | Khớp `preprocess_data/`: `val_edges`/`test_edges` shape (6034,2) → 6034 user; `item_degree` shape (3533,) → 3533 item; `train_edges` shape (563204,2) |
| `p_i = p_min+(p_max-p_min)·log(1+deg_i)/log(1+deg_max)`, khoảng `[0.1,0.4]` | Khớp `config.py` (`DROPOUT_P_MIN=0.1, DROPOUT_P_MAX=0.4`) và `src/ile_losses.py: compute_degree_aware_dropout_probs()` |
| `E^(k+1)=Ã E^(k)`, `Ã=D^(-1/2)AD^(-1/2)`, layer-wise mean `1/(K+1)Σ E^(k)` | Khớp `src/models.py: propagate()`, `_normalized_adjacency()` |
| `s(u,i)=⟨e_u,e_i⟩` | Khớp `src/models.py: full_sort_scores()`, `bpr_forward()` |
| `L_BPR = -1/|B| Σ log σ(s_ui+-s_uj-)` | Khớp **chính xác từng ký tự** với `src/losses.py: bpr_loss()` — `per_triplet = -logsigmoid(pos-neg)`, reduction="mean" |
| `L_CL = L_NCE^user + L_NCE^item` | Khớp **chính xác** `src/popaware_training.py` dòng 269-270: `cl = info_nce(u1[...],u2[...],tau) + info_nce(i1[...],i2[...],tau)` — tổng 2 InfoNCE riêng biệt cho user và item, **chính xác hơn** bản đơn giản hoá trong pptx slide 7 |
| `P(neg=i) ∝ deg_i^β`, β=0 uniform / β>0 popularity-biased | Khớp `src/neg_sampling.py: build_neg_probs()` |
| Inference dùng "the original (no-dropout) graph" | Khớp cơ chế `aug_main=False` trong mọi kết quả báo cáo (xem Q1) |
| Mask "train (and val when evaluating test)" | Khớp **chính xác** comment trong code, `src/popaware_training.py` dòng 199-201: `# TEST: target = test items, mask = train+val` |
| `K (recommend) = 20` | Khớp `config.PRIMARY_K = 20` |
| `β ∈ {0, 0.5}` | Khớp đúng lưới sweep trong `train_sweep_popaware.py` |

**❌ Sai, cần sửa trước khi dùng:**

1. **Ô ⑤② — `L_ILE = (l̄_item − l̄_tail)²` phải là `(l̄_head − l̄_tail)²`.** Bằng chứng: `src/ile_losses.py` đặt tên biến `head_loss`, `tail_loss`; và chính docstring trong `src/popaware_training.py` dòng 165 viết rõ `[+ λ_ILE·(head−tail)²]`. "item" ghép với "tail" không tạo thành cặp đối lập có nghĩa (3 nhóm là head/middle/tail) — đây gần như chắc chắn là lỗi gõ nhầm head→item khi vẽ hình, cần sửa lại trước khi trình bày.

**⚠️ Đúng nhưng cần trả lời chính xác hơn nếu bị hỏi xoáy sâu:**

2. **`λ_reg` vs `wd`:** code gọi hệ số này là `config.WEIGHT_DECAY` (dòng 256: `total = loss + config.WEIGHT_DECAY * reg`), không phải "λ_reg". Cùng vai trò, khác tên — nên đổi nhãn trên hình thành `wd` để nhất quán với các tài liệu khác (pptx đang dùng "wd").
3. **`L_reg = ‖θ‖²` là công thức rút gọn, chưa chính xác tuyệt đối.** Theo `src/losses.py: l2_regularization()`, công thức thật là: lấy TỔNG bình phương chuẩn L2 của **đúng 3 tensor embedding gốc (ego, layer-0) của batch hiện tại** — `u0` (user), `p0` (positive item), `n0` (negative item) — rồi **chia cho batch_size**. Tức `L_reg = (‖u0‖² + ‖p0‖² + ‖n0‖²) / |B|`, không phải chuẩn L2 của TOÀN BỘ tham số mô hình θ. Nếu bị hỏi "θ là gì", trả lời đúng phạm vi này, đừng nói "toàn bộ tham số".
4. **`L_NCE^user`, `L_NCE^item` là InfoNCE MỘT CHIỀU, không phải bản đối xứng 2 chiều.** Theo `src/popaware_training.py` dòng 96-101, hàm `info_nce(z1,z2,tau)` chỉ tính `cross_entropy((z1@z2ᵀ)/τ, diag)` — một chiều duy nhất (z1 làm query, z2 làm key), KHÔNG cộng thêm chiều ngược lại `cross_entropy((z2@z1ᵀ)/τ, diag)` như một số biến thể SimCLR/SGL đối xứng khác vẫn làm. Hình không tuyên bố sai điều này (không ghi "symmetric"), nhưng nếu giảng viên yêu cầu viết ra công thức đầy đủ của `L_NCE^user`, cần viết đúng là 1 chiều, không phải trung bình 2 chiều.
5. **Khung "Key Hyperparameters": giá trị `0` trong `λ_ILE∈{0,0.1,0.5,1.0}` và `λ_CL∈{0,0.1,0.5}` không đến từ lưới sweep.** Lưới sweep thật (`train_sweep_popaware.py`) chỉ chạy `λ_ILE∈{0.1,0.5,1.0}` và `λ_CL∈{0.1,0.5}` (không có 0); giá trị "0" chỉ xuất hiện gián tiếp qua cấu hình `baseline` (tương đương tắt hẳn use_ile/use_cl). Khung ghi "(example)" nên về cơ bản không sai, nhưng nếu bị hỏi "0 có trong lưới sweep không", câu trả lời chính xác là "không, 0 chỉ có ở baseline riêng".
6. **✅ ĐÃ XÁC NHẬN — Khung "Popularity Groups" (1.755 / 1.071 / 707 item) đúng 100%.** Đã chạy trực tiếp `Counter(item_popularity_group.tolist())` trên `preprocess_data/item_popularity_group.pt`, kết quả `{0: 1755, 1: 1071, 2: 707}` (0=tail, 1=middle, 2=head), tổng 3.533 — khớp chính xác tuyệt đối với số trên hình. Không cần sửa gì ở khung này.

**⚠️ Rà soát thêm lượt 2 — logic mũi tên/kết nối giữa các khối (chưa soi ở lượt 1):**

7. **Contrastive Loss (nhánh ③) thực chất KHÔNG đi qua khối ④ "Scoring".** Luồng mũi tên trên hình vẽ liền mạch ①→②→③→④→⑤, dễ khiến người xem hiểu rằng cả 4 loss trong khối ⑤ đều dùng chung điểm số dot-product `s(u,i)` từ khối ④. Thực tế, theo code: `L_BPR` và `L_ILE` đúng là dùng điểm số `s(u,i)` (dot product, khối ④); nhưng `L_CL` (`info_nce`) lại lấy **thẳng embedding thô** từ khối ③ (2 view), chuẩn hoá rồi nhân ma trận trực tiếp — **không đi qua bước "Scoring" dạng dot-product** của khối ④. Nếu bị hỏi "mũi tên từ Scoring có áp dụng cho InfoNCE không", câu trả lời chính xác là: không, nhánh ③ nhận embedding trực tiếp từ khối ③, chỉ có nhánh ①② mới thực sự đi qua khối ④.
8. **Chiều mũi tên nét đứt giữa khối ⑤ (Total Loss) và khối ⑥ (Negative Sampling) nên hiểu là "⑥ diễn ra TRƯỚC, làm đầu vào cho ⑤"**, không phải chiều ngược lại. Về mặt thực thi, phải lấy mẫu negative (khối ⑥) trước, rồi mới tính được điểm số và loss (khối ⑤) cho đúng cặp positive/negative đó — nét đứt hai chiều trên hình có thể hiểu là "có phụ thuộc qua lại" một cách hợp lý, nhưng nếu bị yêu cầu mô tả đúng thứ tự thực thi, nên nói rõ: lấy mẫu negative → tính điểm → tính loss, chứ không phải loss sinh ra rồi mới quay lại lấy mẫu negative.
9. **Khối ① vẽ đồ thị với mũi tên một chiều (user → item)**, trong khi đồ thị `edge_index_train` thực tế dùng để lan truyền trong code là **hai chiều** (`shape (2, 1.126.408)` ≈ gấp đôi 563.204 cạnh train, vì mỗi cạnh được lưu cả 2 hướng user↔item). Cách vẽ mũi tên 1 chiều trong hình chỉ nhằm minh hoạ "ai đã tương tác với ai" (ngữ nghĩa dữ liệu gốc), không phải cấu trúc đồ thị dùng để lan truyền — không sai, nhưng nên sẵn sàng giải thích phân biệt này nếu bị hỏi "đồ thị này có hướng hay vô hướng".

**Cập nhật tóm lại:** 9/9 khối nội dung + 3 điểm luồng mũi tên đã rà soát. Tổng cộng: 1 lỗi cần sửa (mục 1), 8 điểm cần nắm chắc phần diễn giải chính xác hơn nếu bị hỏi xoáy (mục 2-9). Không phát hiện thêm lỗi công thức hay lỗi logic nghiêm trọng nào khác.

#### Checklist hành động — dùng để vẽ lại hình cho chắc chắn chính xác

> Sắp theo mức ưu tiên. Mức 1 **bắt buộc sửa** (sai thật). Mức 2 **nên sửa** (để hình tự nó đã đủ chính xác, không cần giải thích thêm bằng lời khi trình bày). Mức 3 **tuỳ chọn** (footnote nhỏ cho người đọc kỹ tính, không sửa cũng không sai). Mức 4 là việc cần tự làm trước khi in số liệu.

**Mức 1 — Bắt buộc sửa (sai thật):**

1. Ô ⑤② đổi `L_ILE = (l̄_item − l̄_tail)²` → **`L_ILE = (l̄_head − l̄_tail)²`**. Đổi luôn `l̄_g = mean_{i∈g}[...]` giữ nguyên, chỉ đổi mác nhóm `item`→`head`.

**Mức 2 — Nên sửa (để hình tự đủ chính xác):**

2. Đổi tên hệ số L2 trong Total Loss: `λ_reg` → **`wd`** (khớp tên biến thật `config.WEIGHT_DECAY`), cho khớp với các tài liệu khác.
3. Thêm chú thích nhỏ dưới `L_reg = ‖θ‖²`: ghi rõ **"θ = ego embedding (layer-0) của user/positive-item/negative-item trong batch hiện tại, chia cho batch size"** — thay vì để người đọc hiểu nhầm là toàn bộ tham số mô hình.
4. **[NÂNG MỨC — quan trọng hơn đánh giá lần trước]** Toàn bộ luồng ①→②→③→④ trên hình ngụ ý CHỈ CÓ MỘT đường lan truyền (qua 2 view augment) rồi mới ra Scoring/BPR/ILE. Đọc lại đúng code (`popaware_training.py` dòng 244-272) cho thấy thực tế có **2 nhánh lan truyền hoàn toàn tách biệt, chạy song song mỗi bước huấn luyện**:
   - **Nhánh A (chính, cho BPR/ILE/Inference):** đồ thị GỐC (`edge_index`, không dropout vì `aug_main=False`) → 1 lần `propagate()` → embedding → Scoring `s(u,i)` → BPR Loss + ILE Loss. Đây cũng chính là embedding dùng ở khối ⑦ Inference.
   - **Nhánh B (phụ, chỉ cho CL):** đồ thị gốc → tạo **2 view MỚI, lấy mẫu độc lập lại từ đầu mỗi bước** (không phải 2 view cố định) → **2 lần `propagate()` riêng** (`u1,i1=propagate(v1)`; `u2,i2=propagate(v2)`) → thẳng vào InfoNCE, không qua Scoring.
   
   Tức mỗi bước huấn luyện (khi bật CL) có **tổng cộng 3 lần lan truyền** (1 cho nhánh A + 2 cho nhánh B), không phải 1-2 lần như hình đang vẽ theo một chuỗi duy nhất. Cần vẽ lại thành **2 nhánh rẽ riêng ngay sau khối ①**, không phải một chuỗi nối tiếp — đây là điểm cấu trúc quan trọng nhất cần sửa, không chỉ là "đổi 1 mũi tên" như tôi mô tả ở lượt review trước.
5. Đổi mũi tên nét đứt giữa ⑤ Total Loss và ⑥ Negative Sampling thành **một chiều, từ ⑥ → ⑤** (negative sampling luôn diễn ra trước khi tính điểm/loss), thay vì mũi tên hai đầu gây hiểu nhầm về thứ tự.

**Mức 3 — Tuỳ chọn (footnote cho chặt chẽ, không bắt buộc):**

6. Thêm ghi chú nhỏ cạnh `L_NCE^user`, `L_NCE^item`: "(InfoNCE một chiều: view1 làm query, view2 làm key)" — để rõ đây không phải bản đối xứng 2 chiều.
7. Sửa khung "Key Hyperparameters" thành: `λ_ILE ∈ {0.1, 0.5, 1.0}` và `λ_CL ∈ {0.1, 0.5}` là **lưới sweep**; ghi thêm dòng nhỏ "baseline: λ_ILE=λ_CL=0" riêng, thay vì gộp "0" chung vào một tập hợp gây hiểu nhầm là nằm trong lưới sweep.
8. Thêm chú thích nhỏ dưới đồ thị ở khối ①: "(mũi tên minh hoạ chiều tương tác; cạnh dùng để lan truyền trong huấn luyện là vô hướng/2 chiều)".

**Mức 4 — Đã hoàn tất, không còn việc tồn đọng:**

9. ✅ Đã chạy `Counter(item_popularity_group.tolist())` trên `preprocess_data/item_popularity_group.pt` → `{0: 1755, 1: 1071, 2: 707}`, tổng 3.533 — khớp chính xác 100% với hình. Không cần làm gì thêm ở mục này.

**Những gì ĐÃ đúng, giữ nguyên không cần đổi:** toàn bộ khối ①②③④⑥⑦, công thức `L_BPR`, số liệu `|U|`, `|I|`, `Train edges`, `K=20`, `β∈{0,0.5}`, và câu "using the original (no-dropout) graph" + "mask train (and val when evaluating test)" ở khối ⑦ — đây là 2 câu chính xác nhất và quan trọng nhất trên toàn hình, đừng chỉnh sửa gì ở đó.

**Tóm lại:** hình đạt độ chính xác cao (9/10 khối đúng hoàn toàn), có 1 lỗi chắc chắn cần sửa (mục 1), và 5 điểm nên nắm chắc phần "trả lời chính xác hơn" để không bị đuối lý nếu giảng viên hỏi xoáy vào từng ký hiệu.

### Slide 10 — Experimental Setup (~75-90s)

> "Chuyển sang phần thực nghiệm. Slide này trình bày thiết lập thí nghiệm.
>
> Dataset: MovieLens-1M, ngưỡng rating từ 4 trở lên là positive, chia validation/test theo leave-one-out theo thời gian cho từng user.
>
> Backbone: embedding dimension 64, num_layers = 2 làm điểm vận hành chính.
>
> Em quét lưới 24 điểm siêu tham số: num_layers {2,3}, λ_ILE {0.1, 0.5, 1.0}, λ_CL {0.1, 0.5}, β {0, 0.5}.
>
> Quy tắc chọn operating point: tối đa Coverage@20 với ràng buộc Recall@20 không giảm quá một ngưỡng so với baseline — và em lưu ý ngay, quy tắc này hiện tính trên số liệu TEST của lưới sweep, đây là điểm em đã nêu như một hạn chế, sẽ giải trình kỹ hơn nếu thầy cô hỏi.
>
> Để đảm bảo độ tin cậy thống kê, mỗi cấu hình cuối được huấn luyện lại độc lập qua 3 seed — 42, 0, 1 — báo cáo trung bình cộng độ lệch chuẩn.
>
> Bảng dưới liệt kê 5 cấu hình chốt: baseline LightGCN, và 4 điểm PopAware — accuracy, BEST, high-tail, fairness — cùng λ_ILE, λ_CL, β tương ứng."

### Slide 11 — Main Results (~100-120s)

> "Đây là bảng kết quả chính — trọng tâm của phần thực nghiệm.
>
> Bảng có 6 chỉ số ở K=20: Recall, NDCG càng cao càng tốt cho độ chính xác; TailRecall, Coverage càng cao càng tốt cho công bằng; ARP, HeadExposure càng thấp càng tốt, tức càng ít thiên lệch về phía item phổ biến. Mỗi giá trị là trung bình cộng độ lệch chuẩn qua 3 seed.
>
> So với baseline, tại PopAware-BEST: TailRecall tăng gấp 6,5 lần, từ 0.0050 lên 0.0324; Coverage tăng 35%; ARP giảm 6,3%; HeadExposure giảm 7,9%; và quan trọng nhất, Recall vẫn TĂNG 4%, không hề giảm.
>
> Hai điểm em nhấn mạnh ở bullet dưới bảng: Một, khoảng mean cộng trừ độ lệch chuẩn giữa baseline và mọi cấu hình PopAware đều KHÔNG chồng lấn nhau trên các chỉ số công bằng — cải thiện vượt ra ngoài nhiễu ngẫu nhiên giữa các seed. Hai, ba trong bốn cấu hình PopAware — accuracy, BEST, high-tail — vừa giảm bias vừa CẢI THIỆN accuracy; chỉ riêng cấu hình 'fairness' mới thực sự đánh đổi accuracy để lấy debias mạnh nhất."

### Slide 12 — Frontier: Accuracy vs. Fairness (~60-75s)

> "Slide này trực quan hoá đánh đổi giữa accuracy và fairness bằng biểu đồ frontier.
>
> Trục hoành là Coverage@20 — càng sang phải, bias càng thấp. Trục tung là Recall@20 — càng lên cao, độ chính xác càng cao. Mỗi điểm là một cấu hình đã huấn luyện thật, không phải minh hoạ định tính.
>
> Bốn điểm PopAware nối bằng đường nét đứt, tạo thành một frontier có thể lựa chọn theo mục tiêu — khác với BPR-MF chỉ là MỘT điểm cố định. Cụm PopAware nằm bên phải LightGCN gốc, tức Recall cao hơn; cấu hình 'fairness' đánh đổi accuracy nhiều nhất để đạt Coverage và HeadExposure tốt nhất.
>
> Em ghi chú ngay trên slide: MostPopular và BPR-MF chỉ có 1 lần chạy, không phải trung bình 3-seed như LightGCN baseline và các điểm PopAware — cần thận trọng hơn khi so sánh với hai điểm tham chiếu này."

### Slide 13 — Analysis & Conclusion (~90-105s)

> "Slide cuối: đối chiếu xem phương pháp đã đạt mục tiêu ban đầu chưa.
>
> Bảng trên liệt kê 6 tiêu chí đặt ra từ đầu, đối chiếu kết quả đo được tại BEST so với baseline. Cả 6 đều đạt: TailRecall tăng ×6.5; Coverage tăng 35%; ARP giảm 6.3%; HeadExposure giảm 7.9%; Recall/NDCG không giảm mà còn tăng nhẹ; và mức tăng TailRecall được coi là hợp lý, không quá cực đoan so với high-tail (tăng ×8.0).
>
> Khung xanh lá bên dưới tóm lại: 6 trên 6 tiêu chí đạt tại PopAware-BEST, xác nhận có ý nghĩa qua 3 seed, không phải một lần chạy may mắn.
>
> Cuối cùng, em chủ động trình bày 'Honest limitations': Một, Degree-Aware Augmentation trong mọi kết quả báo cáo chỉ tác động lên 2 view của Contrastive Learning, hiệu quả độc lập lên đồ thị chính chưa kiểm chứng với 3 seed. Hai, Coverage tại điểm cân bằng (0.538) vẫn thấp hơn BPR-MF (0.628). Ba, giá trị tuyệt đối TailRecall còn nhỏ (khoảng 0.03) — nên hiểu là 'giảm đáng kể bias', không phải 'đã giải quyết hoàn toàn long-tail'. Bốn, bộ siêu tham số của các operating point được chọn qua quy tắc frontier tính trên số liệu test, chưa tách bạch hoàn toàn với validation — em chủ động công bố điều này để đảm bảo tính trung thực khoa học.
>
> Em xin dừng phần trình bày tại đây, sẵn sàng nhận câu hỏi từ quý thầy cô."

---

## 6. Danh sách hành động trước khi bảo vệ

1. Đọc lại Q2, Q6, Q11, Q18, Q22, Q23, Q24, Q40 — đây là các câu đòi hỏi hiểu đúng chi tiết code/công thức, dễ bị hỏi xoáy nếu hội đồng đã đọc code.
2. Chuẩn bị sẵn 1-2 con số thời gian huấn luyện thật (từ log) để trả lời Q32 chủ động thay vì nói "chưa đo".
3. Thuộc câu trả lời Q36–Q39 (nhóm bẫy logic) — đây là nhóm câu hỏi kiểm tra tư duy khoa học, không chỉ kiểm tra thuộc số liệu.
4. Q40 là câu QUAN TRỌNG NHẤT mới phát hiện: đừng lỡ miệng nói "β cải thiện fairness một cách đơn điệu" — số liệu sweep không ủng hộ câu đó. Chỉ khẳng định đúng phạm vi đã kiểm chứng (so sánh BEST vs high-tail).
5. Tập đọc to kịch bản ở mục 5 ít nhất 2-3 lần, canh thời gian bằng đồng hồ thật — nếu vượt quá thời lượng cho phép của buổi bảo vệ, cắt bớt ở Slide 6-9 (phần cơ chế) trước, giữ nguyên Slide 11 (số liệu chính) và phần "Honest limitations" ở Slide 13.
6. Nếu còn thời gian trước khi bảo vệ: cân nhắc chạy thêm 1 thí nghiệm nhỏ (ví dụ λ_ILE=5 để xem có suy biến không) để có câu trả lời thực nghiệm thay vì chỉ lý thuyết cho Q38.
7. Nếu giảng viên yêu cầu xem code trực tiếp: mở sẵn mục 7 bên dưới trên một màn hình/tab riêng, đã tra cứu trước file/dòng cần trỏ tới cho từng chủ đề hay bị hỏi.
8. Nếu giảng viên hỏi "giải thích từ đầu đến cuối nó chạy như thế nào" (không phải hỏi về kiến trúc mà hỏi về cơ chế vận hành/luồng dữ liệu) — dùng đúng mạch 10 bước ở mục 8, không lẫn với phần kiến trúc 4 thành phần ở mục 5 (hai góc nhìn bổ sung cho nhau: mục 5 nói "có gì", mục 8 nói "nó chạy ra sao").

---

## 7. Cấu trúc thư mục & mã nguồn — chuẩn bị nếu giảng viên yêu cầu xem code

> Mục tiêu của phần này: khi giảng viên nói "cho xem code chỗ X", bạn mở đúng file/hàm trong vài giây, không lúng túng dò cả project. Cấu trúc dưới đây **chỉ liệt kê phần thật sự thuộc pipeline chính** — phần "rác phát triển" (debug/test script tạm) được gom riêng ở mục 7.4 để bạn biết mà bỏ qua nếu giảng viên tình cờ thấy và hỏi.

### 7.1 Cây thư mục (phần cốt lõi)

```text
TestSSH/
├── src/                              # code lõi, tái sử dụng bởi mọi script train/eval
│   ├── config.py                     # hằng số & siêu tham số duy nhất (SEED, EMBEDDING_DIM, LR, TAU, DROPOUT_P_MIN/MAX...)
│   ├── data.py                       # tiền xử lý gốc: lọc rating≥4, split_per_user() — leave-one-out theo thời gian
│   ├── data_loader.py                # DataProcessor: load tensor đã cache từ preprocess_data/
│   ├── metrics.py                    # 10 metric + evaluate_full_ranking() — ĐỌC KỸ nếu bị hỏi định nghĩa metric
│   ├── baselines.py                  # MostPopular
│   ├── models.py                     # BPRMF, LightGCNRecommender (propagate(), _normalized_adjacency())
│   ├── losses.py                     # bpr_loss, l2_regularization
│   ├── train.py                      # vòng lặp train BPR-MF/LightGCN gốc (dùng cho baseline notebook)
│   ├── ile_losses.py                 # ile_loss() — công thức ILE đã sửa lỗi dấu; compute_degree_aware_dropout_probs()
│   ├── neg_sampling.py                # build_neg_probs(), sample_bpr_batch_popaware() — negative sampling theo degree^β
│   └── popaware_training.py          # ★ FILE QUAN TRỌNG NHẤT: train_popaware_lightgcn() — vòng lặp huấn luyện hợp nhất
│                                      #   (symmetric_edge_dropout, info_nce, checkpoint/resume/log, VAL/TEST protocol)
│
├── preprocess_data/                  # tensor đã tiền xử lý (train/val/test edges, item_degree, item_popularity_group)
│   └── README.md                     # mô tả schema từng file .pt
│
├── train_all_popaware.py             # ablation TÁCH RIÊNG từng thành phần (baseline/ile/degreeaug/degreeaug_cl/full), 1 seed
├── train_sweep_popaware.py           # quét 24 điểm lưới + chọn operating point theo frontier (chính là file ở Q2/Q9/Q40)
├── train_final_seeds.py              # chạy 5 cấu hình chốt × 3 seed → SỐ LIỆU CHÍNH của Slide 11
├── evaluate_test_full.py             # nạp lại model đã lưu, tính đủ 10 metric (không train lại)
├── run_on_gpu.sh                     # gửi job lên cụm GPU A100 qua Slurm (xem 7.3)
│
├── results/
│   ├── metrics/main_results.csv      # baseline MostPopular/BPR-MF/LightGCN — 1 run (dẫn trong Slide 12 "Note")
│   ├── popaware_sweep_20260715_155058.csv        # 24 điểm sweep — nguồn số liệu cho Q9/Q40
│   ├── popaware_final_runs_20260715_174317.csv   # 15 run (5 cấu hình × 3 seed) — số liệu THÔ trước khi lấy mean±std
│   ├── popaware_final_meanstd_20260715_174317.csv # mean±std cuối — ĐÚNG BẢN NGUỒN của bảng Slide 11
│   └── popaware/history_*.csv        # loss/metric theo từng epoch của mỗi run — dùng nếu bị hỏi "cho xem đường cong loss"
│
├── checkpoints/popaware/*_best.pt    # trọng số model tốt nhất theo VAL (mỗi run 1 file _best + 1 file _latest)
├── logs/popaware/*.log               # log đầy đủ từng epoch, có timestamp — dùng để trả lời Q32 (thời gian train/epoch)
├── models/final_model_*.pt           # model cuối, tương thích evaluate_test_full.py
│
├── README.md                         # tổng quan project, cách setup & chạy
├── PopAware_LightGCN_Documentation.md# báo cáo kỹ thuật đầy đủ (tiếng Việt)
├── report_method_section.tex         # phần phương pháp viết bằng LaTeX (tiếng Anh)
├── slides.md                         # bản nháp markdown của slide 5-13
├── PopAware_Slides_05-13_EN.pptx     # ★ FILE SLIDE DÙNG ĐỂ BẢO VỆ
└── Slide_Review_and_Defense_QA.md    # chính file này
```

### 7.2 Tra cứu nhanh: "Nếu giảng viên hỏi... → mở file..."

| Giảng viên hỏi / yêu cầu | Mở file (hàm/dòng) |
|---|---|
| "Công thức LightGCN implement thế nào?" | `src/models.py` → class `LightGCNRecommender`, hàm `propagate()`, `_normalized_adjacency()` |
| "Công thức ILE, sửa lỗi dấu ở đâu?" | `src/ile_losses.py` → hàm `ile_loss()` (dòng ~114-119 có comment giải thích rõ bản cũ vs bản đã sửa) |
| "Dropout theo degree tính thế nào?" | `src/ile_losses.py` → `compute_degree_aware_dropout_probs()`; áp dụng đối xứng ở `src/popaware_training.py` → `symmetric_edge_dropout()` |
| "InfoNCE / Contrastive Learning code ở đâu?" | `src/popaware_training.py` → hàm `info_nce()` (đây là bản ĐANG DÙNG; hàm `contrastive_loss()` trong `ile_losses.py` là bản cũ, không dùng trong pipeline chính — nếu giảng viên mở nhầm file đó, giải thích rõ điều này) |
| "Negative sampling theo popularity code ở đâu?" | `src/neg_sampling.py` → `build_neg_probs()`, `sample_bpr_batch_popaware()` |
| "Cách chia train/val/test?" | `src/data.py` → hàm `split_per_user()` (sort theo timestamp, lấy 2 dòng cuối làm val/test) |
| "Định nghĩa Recall/NDCG/TailRecall/Coverage/ARP/Exposure?" | `src/metrics.py` → `recall_at_k()`, `ndcg_at_k()`, `tail_recall_at_k()`, `catalog_coverage_at_k()`, `average_recommendation_popularity()`, `exposure_by_group()` |
| "Toàn bộ vòng lặp train (checkpoint, resume, log, leakage-free) ở đâu?" | `src/popaware_training.py` → hàm `train_popaware_lightgcn()` — file quan trọng nhất, nên mở sẵn |
| "Siêu tham số cụ thể (LR, batch size, weight decay...) là bao nhiêu?" | `src/config.py` |
| "Cho xem kết quả sweep 24 điểm thô" | `results/popaware_sweep_20260715_155058.csv` |
| "Cho xem số liệu 3-seed trước khi tính mean±std" | `results/popaware_final_runs_20260715_174317.csv` |
| "Số liệu bảng Slide 11 lấy chính xác từ đâu?" | `results/popaware_final_meanstd_20260715_174317.csv` |
| "Cho xem log huấn luyện thật / thời gian mỗi epoch" | `logs/popaware/final_best_L2_s42.log` (hoặc bất kỳ file log nào trong thư mục) |
| "Cho xem đường cong loss/metric qua các epoch" | `results/popaware/history_final_best_L2_s42.csv` |
| "Chạy thử lại một cấu hình ngay bây giờ được không?" | Xem lệnh ở mục 7.3 bên dưới |
| "Ablation tách riêng từng thành phần ở đâu?" | `train_all_popaware.py` (chạy 1 seed, xem 7.1) |

### 7.3 Lệnh chạy nếu cần demo trực tiếp

Toàn bộ training chạy trên cụm GPU A100 từ xa qua `run_on_gpu.sh`, không chạy trực tiếp trên máy laptop:

```bash
./run_on_gpu.sh --status                         # xem job đang chạy/chờ trên cụm
./run_on_gpu.sh --log                             # xem log job gần nhất
./run_on_gpu.sh evaluate_test_full.py             # chạy nhanh, tính lại 10 metric từ model đã lưu (không cần Slurm)
./run_on_gpu.sh train_final_seeds.py --seeds 42 --epochs 5   # demo nhanh 1 seed, 5 epoch (không phải số liệu thật)
```

Nếu giảng viên chỉ muốn xem code chạy được (không cần đúng số liệu), dùng lệnh demo nhanh ở trên (`--epochs 5`) để có kết quả trong vài phút thay vì chờ full training.

### 7.4 Các file KHÔNG cần quan tâm (rác phát triển, không thuộc pipeline chính)

Ở thư mục gốc còn nhiều file như `debug_*.py`, `fix_*.py`, `test_*.py`, `*_convert.py`, `comprehensive_fix.py`, `minimal_test.py`, `hello.py`, `hello2.py`, `gpu_hello.py`, `simple_test.py`... — đây là các script một-lần dùng để chẩn đoán lỗi cụ thể trong quá trình phát triển (lỗi glob `**` trên Python cũ, lỗi `weights_only` của PyTorch≥2.6, lỗi kiểu dữ liệu tensor khi convert từ parquet). Nếu giảng viên tình cờ mở phải một trong số này và hỏi, câu trả lời an toàn:

> "Đây là script debug tạm thời trong quá trình phát triển để xử lý một lỗi kỹ thuật cụ thể, không thuộc pipeline huấn luyện/đánh giá chính. Pipeline chính nằm ở 4 file `train_all_popaware.py`, `train_sweep_popaware.py`, `train_final_seeds.py`, `evaluate_test_full.py`, cùng toàn bộ code trong `src/`."

Tương tự, trong `src/` có `ile_training.py`, `run_ile_experiments.py`, `run_augmentation_experiments.py`, `pd_debias.py`, `pd_training.py` — đây là các phiên bản/cách tiếp cận **thử trước rồi bị thay thế** (ile_training.py là vòng lặp train ILE-only trước khi có `popaware_training.py` hợp nhất; `pd_debias.py`/`pd_training.py` thuộc hướng causal debiasing "Popularity Deconfounding" đã khám phá rồi dừng lại — xem Q10). Không cần xoá, nhưng cũng không nên chủ động show trừ khi giảng viên hỏi trực tiếp về lịch sử phát triển.

---

## 8. Giải thích Step-by-Step: Pipeline hoạt động như thế nào (dễ hiểu nhất)

> Mục tiêu phần này: nếu giảng viên hỏi "giải thích cho tôi từ lúc load data đến lúc ra kết quả, nó chạy như thế nào" — đây là câu trả lời tuần tự, dùng ví von đơn giản, kèm đúng tên file/hàm để bạn có thể vừa nói vừa trỏ vào code. Xuyên suốt phần này dùng một ví von duy nhất: **hãy tưởng tượng mỗi user và mỗi item là một CHẤM ĐIỂM trên một tấm bản đồ 64 chiều** — việc huấn luyện chính là quá trình DI CHUYỂN các chấm điểm đó sao cho user nằm gần các item họ thích.

### Bước 0 — Tiền xử lý dữ liệu (chạy đúng 1 lần, trước khi train)

Đọc MovieLens-1M, giữ lại rating ≥ 4 làm "positive". Đánh số lại user/item thành chỉ số liên tục 0..N. Với mỗi user, sắp xếp tương tác theo thời gian, cắt ra 1 tương tác gần nhất làm test, 1 tương tác kế đó làm validation, phần còn lại làm train. Tính degree của từng item (bao nhiêu lượt xuất hiện trong train) rồi xếp vào nhóm tail/middle/head theo percentile. Lưu hết thành file `.pt` để không phải làm lại mỗi lần train.
📁 *File: `src/data.py` (hàm `split_per_user`) → sinh ra dữ liệu trong `preprocess_data/`.*

### Bước 1 — Load dữ liệu khi bắt đầu train

Khi một script train (ví dụ `train_final_seeds.py`) chạy, việc đầu tiên là gọi `DataProcessor()` để đọc lại các file `.pt` đã cache ở Bước 0: danh sách cạnh train/val/test, đồ thị 2 chiều `edge_index_train`, degree từng item, nhóm popularity từng item. Đồng thời dựng sẵn một bảng tra cứu "mỗi user đã tương tác với những item nào trong train" — dùng để loại trừ khi lấy negative sau này.
📁 *File: `src/data_loader.py` → class `DataProcessor`.*

### Bước 2 — Khởi tạo Embedding: tấm bản đồ ban đầu, còn "trắng"

Mỗi user và mỗi item được gán một vector 64 số thực (`EMBEDDING_DIM=64`), khởi tạo **hoàn toàn ngẫu nhiên** (phân phối chuẩn, độ lệch nhỏ std=0.1). Toàn bộ user + item dùng CHUNG một bảng embedding duy nhất (`nn.Embedding(num_users + num_items, 64)`) — item chỉ đơn giản được đánh số tiếp theo sau user (item thứ *i* có index = `num_users + i`).

Ở bước này, các "chấm điểm" trên bản đồ 64 chiều nằm rải rác ngẫu nhiên, **chưa mang ý nghĩa gì cả** — mô hình chưa biết user nào thích item nào. Toàn bộ quá trình học phía sau là để dịch chuyển các chấm điểm này.
📁 *File: `src/models.py` → `LightGCNRecommender.__init__` (dòng khởi tạo `self.embedding`).*

### Bước 3 — Lan truyền qua đồ thị: "hỏi ý kiến hàng xóm"

Thay vì dùng thẳng embedding gốc (Bước 2), LightGCN cho mỗi user/item cập nhật vị trí của mình bằng cách **hỏi ý kiến hàng xóm trực tiếp trên đồ thị**: embedding lớp kế tiếp của 1 node = trung bình có trọng số của embedding các hàng xóm ở lớp trước (trọng số là `1/√(deg_i × deg_j)` — chuẩn hoá đối xứng, để user/item có nhiều tương tác không "lấn át" người ít tương tác).

Lặp lại đúng `num_layers` lần (K=2 ở cấu hình chính) — sau lần 1, user "biết" thông tin của các item mình từng tương tác; sau lần 2, user còn "biết" gián tiếp cả những user khác có cùng sở thích (hàng xóm của hàng xóm). Cuối cùng, embedding dùng để suy luận là **trung bình cộng của cả 3 lớp** (lớp 0 gốc + lớp 1 + lớp 2) — gọi là "layer-wise mean pooling".
📁 *File: `src/models.py` → hàm `propagate()`.*

### Bước 4 — Tính điểm số dự đoán: "đo độ hợp gu"

Điểm số dự đoán giữa 1 user và 1 item = **tích vô hướng (dot product)** giữa 2 vector embedding cuối cùng (sau khi đã lan truyền ở Bước 3). Hai chấm điểm càng "hướng cùng chiều và gần nhau" trên bản đồ 64 chiều thì tích vô hướng càng lớn → mô hình dự đoán user càng thích item đó.
📁 *File: `src/models.py` → `full_sort_scores()` (khi đánh giá) hoặc trực tiếp trong `bpr_forward()` (khi train).*

### Bước 5 — Lấy một batch để học: chọn "câu hỏi luyện tập"

Mỗi bước huấn luyện, lấy ngẫu nhiên 4096 cặp (user, item positive) thật từ train. Với mỗi cặp, chọn thêm đúng 1 "item negative" — một item mà (nhiều khả năng) user không thích — theo 1 trong 2 cách: chọn đều ngẫu nhiên trong toàn catalog (β=0), hoặc thiên vị chọn item phổ biến hơn (β>0, xác suất chọn ∝ degree^β). Có kiểm tra để đảm bảo negative không trùng với item user đã thực sự tương tác trong train (thử lại tối đa 20 lần nếu trùng).
📁 *File: `src/neg_sampling.py` → hàm `sample_bpr_batch_popaware()`.*

### Bước 6 — Tính Loss: "chấm điểm xem mô hình đang sai đến đâu"

Đây là bước mô hình tự "biết mình sai ở đâu":
- **BPR loss** (luôn có): muốn điểm(user, positive) lớn hơn điểm(user, negative) càng nhiều càng tốt → công thức `-log(sigmoid(điểm_dương - điểm_âm))`. Nếu mô hình đoán đúng (điểm dương đã lớn hơn nhiều), loss nhỏ; nếu đoán sai/ngược, loss lớn.
- **+ ILE loss** (nếu bật): phạt thêm nếu loss trung bình của các item nhóm tail trong batch này cao hơn hẳn loss trung bình nhóm head.
- **+ Contrastive loss** (nếu bật): kéo embedding của cùng 1 node ở 2 "view" đồ thị đã dropout lại gần nhau.
- **+ L2 regularization**: phạt nhẹ nếu embedding có giá trị quá lớn, để tránh học thuộc lòng (overfitting).

Tất cả cộng lại theo đúng công thức Joint Objective ở Slide 9: `L = L_BPR + λ_ILE·L_ILE + λ_CL·L_CL + wd·‖Θ‖²`.
📁 *File: `src/losses.py` (BPR, L2), `src/ile_losses.py` (ILE), `src/popaware_training.py` (hàm `info_nce`, và vòng lặp ghép tất cả loss lại).*

### Bước 7 — Cập nhật Embedding: "học từ sai lầm"

Từ giá trị loss ở Bước 6, PyTorch tự động tính đạo hàm ngược (backpropagation) để biết **mỗi con số trong bảng embedding cần tăng hay giảm bao nhiêu** để loss nhỏ đi. Thuật toán Adam (learning rate = 0.001) dùng đạo hàm đó để cập nhật toàn bộ bảng embedding.

Sau bước này, các "chấm điểm" trên bản đồ 64 chiều đã dịch chuyển một chút: user dịch **lại gần** positive item, dịch **ra xa** negative item. Lặp lại hàng nghìn lần, các chấm điểm dần dần sắp xếp lại có ý nghĩa: user nằm gần cụm item họ thực sự thích.
📁 *File: `src/popaware_training.py` → vòng lặp chính trong `train_popaware_lightgcn()` (đoạn `loss.backward()` + `optimizer.step()`).*

### Bước 8 — Lặp lại qua nhiều epoch, thỉnh thoảng "kiểm tra bài"

Một epoch = duyệt hết toàn bộ dữ liệu train một lượt (chia thành nhiều batch 4096 cặp). Train tối đa 100 epoch. Cứ mỗi 5 epoch (`eval_every=5`), mô hình tạm dừng cập nhật, lan truyền lại toàn bộ đồ thị, và **chấm điểm thử trên tập VALIDATION** (chưa đụng đến TEST) bằng Recall@K.
📁 *File: `src/popaware_training.py` → hàm `_evaluate()`, gọi định kỳ trong vòng lặp epoch.*

### Bước 9 — Early stopping + lưu checkpoint: "biết dừng đúng lúc"

Nếu Recall@K trên VAL ở lần kiểm tra này **cao hơn** kỷ lục trước đó → lưu ngay bộ embedding hiện tại thành file `..._best.pt` (đây chính là checkpoint sẽ dùng để chấm điểm cuối cùng). Nếu sau 10 lần kiểm tra liên tiếp (`patience=10`) mà VAL không cải thiện thêm → dừng train sớm, tránh việc mô hình "học vẹt" (overfit) dữ liệu train quá mức. Dù dừng sớm hay không, mỗi lần train đều lưu thêm 1 file `..._latest.pt` để nếu máy bị ngắt giữa chừng thì resume lại đúng chỗ đang dừng.
📁 *File: `src/popaware_training.py` → khối `if improved:` (lưu best) và điều kiện đếm `patience`.*

### Bước 10 — Đánh giá cuối cùng trên TEST: chỉ làm đúng 1 lần

Sau khi dừng train, nạp lại đúng bộ embedding tốt nhất theo VAL (`..._best.pt`). Tính điểm số cho **toàn bộ** cặp (user, item) trong catalog — gọi là "full-ranking". Với mỗi user, loại bỏ những item họ đã thấy trong train, lấy ra Top-20 item điểm cao nhất còn lại. So Top-20 này với đúng 1 item thật user tương tác trong TEST → tính ra cả 10 chỉ số (Recall, NDCG, TailRecall, Coverage, ARP, 3 Exposure...) trong đúng MỘT lần gọi duy nhất.
📁 *File: `src/metrics.py` → hàm `evaluate_full_ranking()`, gọi 1 lần duy nhất ở cuối `train_popaware_lightgcn()`.*

### Tóm tắt bằng một câu

> Bắt đầu với các chấm điểm ngẫu nhiên trên bản đồ 64 chiều (Bước 2) → cho mỗi chấm "hỏi hàng xóm" để mang thông tin đồ thị (Bước 3) → so đo hàng nghìn lần xem chấm nào nên gần chấm nào (Bước 5-7) → dừng đúng lúc dựa trên tập validation, không phải test (Bước 8-9) → cuối cùng mới hé lộ kết quả trên test đúng một lần (Bước 10).

### Sơ đồ tóm tắt vòng lặp (để vẽ nhanh lên bảng nếu cần)

```text
[Load data]  →  [Khởi tạo embedding ngẫu nhiên]
     │
     ▼
┌─────────────── Lặp lại mỗi epoch (tối đa 100 lần) ───────────────┐
│  [Lan truyền đồ thị K lớp] → [Tính điểm user-item]                │
│  [Lấy batch: positive thật + negative lấy mẫu]                    │
│  [Tính Loss = BPR (+ILE +CL +L2)] → [Backprop] → [Adam cập nhật]  │
│  (mỗi 5 epoch) → [Chấm điểm trên VAL] → cải thiện? lưu best        │
│                                        không cải thiện 10 lần? dừng│
└────────────────────────────────────────────────────────────────────┘
     │
     ▼
[Nạp lại best checkpoint] → [Chấm điểm TEST đúng 1 lần] → [10 metric]
```
