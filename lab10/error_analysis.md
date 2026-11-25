# LAB 10 — Phân tích thực nghiệm thật có AI hỗ trợ

 Mọi số dưới đây lấy từ full run **`8faacbbd16e7d738dcf71bbdd8f51d12ffe7b805c1b43f8ea62df17775b0772c`**, không từ smoke test. Model phân tích là **Transformer_384, epoch2**, được chọn bằng internal validation trước khi suy luận heldout. Bản prediction gốc được giữ nguyên tại [prediction.md](prediction.md), hash `e50b5d4c8b0dbf5bad9892b893610a34c0d6c163602c1601e4e878adcb8e2deb`; phần đối chiếu nằm trong báo cáo này.

Nguồn kiểm tra: [results.csv](results.csv), [predictions.json](.runtime/full_run/predictions.json), [length_runs.json](.runtime/full_run/length_runs.json), [error_counts.json](.runtime/full_run/error_counts.json), [run_manifest.json](.runtime/full_run/run_manifest.json). EM/F1 trong bảng là **%**; delta là **điểm phần trăm**. Offset ký tự theo Python Unicode `[start,end)`, end token ở mô hình là inclusive.

## 14. Experiment 1: Context Length

Toàn bộ 87.599 câu hỏi train gốc được chia theo title thành **79.214 train / 8.385 validation**; **10.570 public dev** là heldout nội bộ, không phải official hidden test. Seed42, dữ liệu/split, tokenizer, khởi tạo backbone, question cap48, stride64, batch32, BF16 CUDA, lr3e-5, AdamW và epochs2 đều giữ cố định. DistilBERT có3 special tokens cho cặp, nên cấu hình128 còn ít nhất77 context token, lớn hơn stride64. Không dùng stride128 cho cấu hình128.

| Max length | Validation EM | Validation F1 | Heldout EM | Heldout F1 | Training time (s) | Best epoch |
|---|---|---|---|---|---|---|
| 128 | 65.391 | 78.601 | 75.080 | 83.154 | 763.595 | 2 |
| 256 | 65.617 | 79.470 | 75.620 | 84.097 | 627.861 | 2 |
| 384 | 66.213 | 79.960 | 76.632 | 84.724 | 884.245 | 2 |

384 đạt F1 validation cao nhất (79,960%) nên được chọn trước heldout; heldout sau đó cũng tăng từ83,154% ở128 lên84,724% ở384. Trong ba điểm đo này, tăng length cải thiện EM/F1, nhưng không đủ kết luận “luôn tốt hơn” cho mọi length/model/dataset. 256 nhanh nhất trong lần chạy này; chi phí không tăng đơn giản tuyến tính theo chiều dài vì số window giảm rất mạnh.

| Length | Train features | Validation features | Optimizer steps/2 epochs |
|---|---|---|---|
| 128 | 192549 | 20352 | 12036 |
| 256 | 87362 | 9125 | 5462 |
| 384 | 80071 | 8440 | 5006 |

Giữ cùng số epoch trên câu hỏi gốc không đồng nghĩa cùng số optimizer steps, vì overflow thay đổi số feature. Đây là giới hạn so sánh đã ghi rõ, không phải thay ngầm dữ liệu. Thời gian tính cả training, validation inference/decoding và ghi epoch checkpoint; loại preprocessing và bản sao best cuối. Không gộp thời gian tải dữ liệu hay suy luận heldout vào cột này.

## 17. Experiment 2: Pretrained QA Model

Backbone DistilBERT được fine-tune cùng QA head; start/end logits được học bằng trung bình hai cross-entropy. Lựa chọn epoch dùng validation F1, rồi EM, rồi loss thấp hơn khi hòa; cả ba length chọn epoch2. Đây là best trong **hai epoch thực sự đã chạy**, không khẳng định đã tìm được optimum toàn cục. [Learning curves](learning_curves.png) chứa train/validation loss, validation EM/F1 và vạch epoch được chọn.

| Length | Epoch | Train loss | Validation loss | Validation EM | Validation F1 |
|---|---|---|---|---|---|
| 128 | 1 | 1.100 | 0.806 | 64.281 | 77.706 |
| 128 | 2 | 0.665 | 0.842 | 65.391 | 78.601 |
| 256 | 1 | 1.549 | 1.029 | 63.769 | 77.638 |
| 256 | 2 | 0.944 | 0.959 | 65.617 | 79.470 |
| 384 | 1 | 1.614 | 1.045 | 63.733 | 77.665 |
| 384 | 2 | 0.985 | 0.978 | 66.213 | 79.960 |

Ở128, validation loss tăng0,806→0,842 trong khi F1 tăng77,706→78,601: loss xác suất trên feature và F1 span sau decode đo những khía cạnh khác nhau. Vì vậy không chọn lại epoch1 chỉ vì loss thấp hơn; protocol đã chốt chọn F1. Loss giữa các length cũng không so ngang đơn giản vì cơ cấu window có/không chứa gold khác nhau. Không suy ra chắc chắn overfitting chỉ từ hai điểm loss này.

## 20. Experiment 3: EM vs F1

| Model | n | EM | F1 |
|---|---|---|---|
| Baseline | 10570 | 0.218 | 18.883 |
| Transformer_384 | 10570 | 76.632 | 84.724 |

Baseline trả cả câu có keyword overlap cao nên có thể chứa gold nhưng dư nhiều token; EM rất thấp, F1 còn18,883%. Transformer tăng rõ trên cùng10.570 câu hỏi. Normalization chuyển lowercase, bỏ dấu câu ASCII và articles a/an/the, rồi chuẩn hóa whitespace liên tiếp thành một khoảng trắng; F1 dùng multiset token, lấy điểm cao nhất trên mọi gold reference. Vì vậy số từ lặp được đếm đúng số lần, không dùng set để tính F1.

Có **1.507** ví dụ thật EM=0 nhưng F1>0. Năm ví dụ dưới lấy lại từ các case boundary ở mục24 để không lặp toàn bộ context. EM đều bằng0; phần từ trùng giữ F1 dương:

| ID | Gold được so sánh | Prediction | EM | F1 | Vì sao |
|---|---|---|---|---|---|
| 56beace93aeaaa14008c91e1 | Santa Clara | Santa Clara, California | 0 | 80.000 | Giữ Santa Clara nhưng thêm California: overlap2 từ trên prediction3 và gold2, F1=4/5. |
| 56bead5a3aeaaa14008c91ec | four | one of four | 0 | 50.000 | Giữ four nhưng thêm one/of: overlap1 trên prediction3 và gold1, F1=1/2. |
| 56e761d037bdd419002c3f14 | passion | their passion | 0 | 66.667 | Giữ passion nhưng thêm their: overlap1 trên prediction2 và gold1, F1=2/3. |
| 5726e985dd62a815002e94da | full independent prescribing authority | full independent prescribing | 0 | 85.714 | Thiếu authority: giữ3 trong4 từ gold, F1=6/7. |
| 56bf3c633aeaaa14008c9581 | Mercedes-Benz Superdome | Mercedes-Benz | 0 | 66.667 | Thiếu Superdome: giữ Mercedes-Benz (sau normalization thành mercedesbenz), overlap1 trên1/2 từ, F1=2/3. |

## 22. Experiment 4: Question/Context Matching

Lexical recall = `|content_words(question) ∩ content_words(text)| / |content_words(question)|`, lowercase alphanumeric word sets, bỏ stopword đã liệt kê trong notebook. Ngưỡng high≥0,5, low<0,5 được chốt trước xem điểm. Answer-overlap dùng giá trị lớn nhất trên các gold. Đây là phân nhóm mô tả, không phải thí nghiệm nhân quả hay nhãn semantic similarity.

| View | Group | Model | n | EM | F1 |
|---|---|---|---|---|---|
| context_overlap | low | Baseline | 1605 | 0.125 | 15.723 |
| context_overlap | high | Baseline | 8965 | 0.234 | 19.449 |
| answer_overlap | low | Baseline | 10404 | 0.038 | 18.324 |
| answer_overlap | high | Baseline | 166 | 11.446 | 53.943 |
| context_overlap | low | Transformer_384 | 1605 | 69.034 | 78.180 |
| context_overlap | high | Transformer_384 | 8965 | 77.992 | 85.895 |
| answer_overlap | low | Transformer_384 | 10404 | 76.807 | 84.798 |
| answer_overlap | high | Transformer_384 | 166 | 65.663 | 80.047 |

Transformer ở context-overlap high tốt hơn low (F1 85,895 so78,180), nhưng answer-overlap high chỉ80,047 so84,798 ở low; nhóm answer-high chỉ166 mẫu so10.404 mẫu low. Do đó “có nhiều từ giống nhau thì luôn đúng hơn” không được dữ liệu này ủng hộ. Bốn giao nhóm có support low/low=1.605, low/high=0, high/low=8.799, high/high=166: answer nằm trong context làm hai biến liên quan cấu trúc, không phải hai trục độc lập. Không gán điểm0 cho nhóm rỗng. Loại câu hỏi, độ dài, topic và support có thể gây khác biệt.

Ba ví dụ sau cho thấy model có thể trả đúng với ít từ trùng theo quy ước trên; đây là đọc quan hệ trong context, không phải bằng chứng đã đo được “hiểu nghĩa” một cách độc lập:

### Lexical thấp nhưng EM đúng — `56bf10f43aeaaa14008c9501`

**Question:** What month, day and year did Super Bowl 50 take place? 

**Gold alternatives (giữ cả các reference trùng):** #0: "February 7, 2016" [334,350); #1: "February 7, 2016" [334,350); #2: "February 7, 2016" [334,350). So sánh boundary dùng gold #0.

**Prediction:** "February 7, 2016" `[334,350)`; EM=100.000%, F1=100.000%.

**Context gốc `[279,405)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
n their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Sa
```

**Phân tích quan sát:** Lexical question/context=0.375. Câu hỏi dùng “month, day and year ... take place”; context dùng “was played on February 7, 2016”. Prediction lấy đúng kiểu thông tin thời gian dù cách diễn đạt không trùng hoàn toàn.

### Lexical thấp nhưng EM đúng — `56bf159b3aeaaa14008c9507`

**Question:** What was the win/loss ratio in 2015 for the Carolina Panthers during their regular season?

**Gold alternatives (giữ cả các reference trùng):** #0: "15–1" [48,52); #1: "15–1" [48,52); #2: "15–1" [48,52). So sánh boundary dùng gold #0.

**Prediction:** "15–1" `[48,52)`; EM=100.000%, F1=100.000%.

**Context gốc `[0,107)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
The Panthers finished the regular season with a 15–1 record, and quarterback Cam Newton was named the NFL M
```

**Phân tích quan sát:** Lexical question/context=0.400. Câu hỏi dùng “win/loss ratio”; context biểu diễn bằng “15–1 record”. Model lấy đúng con số thay vì cần xuất hiện nguyên cụm win/loss ratio.

### Lexical thấp nhưng EM đúng — `56d98a59dc89441400fdb52b`

**Question:** Who did the Panthers beat to become the NFC champs?

**Gold alternatives (giữ cả các reference trùng):** #0: "Arizona Cardinals" [152,169); #1: "the Arizona Cardinals" [148,169); #2: "Arizona Cardinals" [152,169). So sánh boundary dùng gold #0.

**Prediction:** "Arizona Cardinals" `[152,169)`; EM=100.000%, F1=100.000%.

**Context gốc `[97,224)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
 the NFL Most Valuable Player (MVP). They defeated the Arizona Cardinals 49–15 in the NFC Championship Game and advanced to the
```

**Phân tích quan sát:** Lexical question/context=0.400. Câu hỏi dùng “beat ... champs”, context dùng “defeated ... NFC Championship Game”. Prediction Arizona Cardinals đúng; đây là sự tương ứng giữa beat và defeated cùng vai đối tượng.

Ngược lại, case `56be4db0acb8001400a502ef` ở mục24 vẫn chọn đội thua dù cả hai tên đội và quan hệ defeated cùng xuất hiện: overlap cao không bảo đảm model xác định đúng vai. Chỉ từ một prediction không thể kết luận attention đã tập trung sai token nào.

### Kiểm chứng paired prediction 3/4: thêm nhiễu và đảo câu

Nguồn [perturbation_results.json](.runtime/full_run/perturbation_results.json). Tập chung có **10.438/10.570** câu hỏi; **132** câu bị loại vì ít nhất một accepted gold cắt qua biên câu regex. Regex phân đoạn sau dấu `. ! ?` và whitespace có thể chia cả viết tắt; đây là operational sentence reversal, không phải sentence parser ngôn ngữ học. Đảo các lát nguyên văn rồi nối newline, cập nhật mọi offset; prefix cố định dài206 ký tự và cộng206 vào mọi gold start. Các phép biến đổi được assert bảo toàn gold text. Không retrain hoặc chọn lại checkpoint bằng biến thể.

| Variant | n | EM | F1 | ΔEM | ΔF1 | F1 tăng | F1 giảm | Text prediction đổi |
|---|---|---|---|---|---|---|---|---|
| original | 10438 | 76.566 | 84.700 | 0.000 | 0.000 | 0 | 0 | 0 |
| noise_prefix | 10438 | 75.541 | 83.774 | -1.025 | -0.925 | 594 | 738 | 1921 |
| sentence_reverse | 10438 | 74.737 | 82.739 | -1.830 | -1.960 | 623 | 883 | 2128 |

So sánh dùng original trên cùng10.438 mẫu, không lấy84,724% của toàn10.570 làm mốc. Nhiễu làm F1 giảm0,925 điểm; đảo câu giảm1,960 điểm. Đồng thời vẫn có594/623 mẫu F1 tăng: hai phép biến đổi không làm mọi mẫu xấu đi. Ví dụ `56beace93aeaaa14008c91e1` đổi từ Santa Clara, California (F1=0,8) sang Denver (F1=0) khi thêm prefix, gold mới bắt đầu609. Ví dụ `56beb0f43aeaaa14008c921f` đổi từ đáp án đúng2010 sang1985 khi đảo câu (F1 giảm1); gold2010 đã được ánh xạ sang528. Đây là thay đổi quan sát được; không quy nguyên nhân riêng cho vị trí, discourse hay window vì phép biến đổi đồng thời ảnh hưởng nhiều yếu tố.

### Đối chiếu bốn prediction đã khóa

1. **Prediction1:** được hỗ trợ ở mức mô tả: Transformer hơn baseline rõ, context-overlap high tốt hơn low, nhưng có nhầm đối tượng cạnh tranh. Không kết luận lexical overlap là nguyên nhân duy nhất.
2. **Prediction2:** được hỗ trợ một phần: nhóm7+ có EM62,661%, thấp hơn nhóm1 (79,735%); F1 cũng giảm nhưng ít hơn. Nhóm2–3 tốt hơn nhóm1 nên xu hướng không đơn điệu, đúng với caveat đã dự đoán.
3. **Prediction3:** được hỗ trợ trên phép nhiễu cụ thể: F1 giảm nhẹ0,925 điểm và1.921 prediction text đổi dù gold được giữ. Không suy rộng sang mọi kiểu distractor.
4. **Prediction4:** được hỗ trợ: đảo câu không bất biến,2.128 prediction text đổi, F1 giảm1,960 điểm; nhiều mẫu vẫn đúng. Prediction ban đầu không chốt rằng đảo câu giảm nhiều hơn nhiễu, nên kết quả đó là quan sát mới chứ không viết ngược vào dự đoán.

## 23. Experiment 5: Answer Length

Độ dài = số subword của **gold reference đầu tiên**, không có special token, dùng cùng DistilBERT tokenizer ở mọi run. EM/F1 vẫn xét tất cả gold. Các nhóm cộng đủ10.570 mẫu, không lấy độ dài prediction để phân nhóm.

| Gold subwords | n | Baseline EM | Baseline F1 | Transformer EM | Transformer F1 |
|---|---|---|---|---|---|
| 1 | 2186 | 0.000 | 10.470 | 79.735 | 84.856 |
| 2-3 | 3988 | 0.100 | 15.562 | 80.090 | 86.322 |
| 4-6 | 2840 | 0.070 | 20.911 | 77.042 | 85.844 |
| 7+ | 1556 | 1.093 | 35.516 | 62.661 | 78.394 |

Nhóm7+ khó hơn theo điểm quan sát, nhưng nhóm2–3 đạt cao nhất; chưa tách riêng tác động của độ dài khỏi độ khó câu hỏi/topic. Có9 câu có gold đầu tiên dài hơn decoder cap30 subword. Không kết luận cả9 không thể EM vì các accepted gold khác có thể ngắn hơn. Baseline F1 tăng theo độ dài gold một phần vì baseline lấy cả câu nên có nhiều overlap với đáp án dài; không vì vậy mà baseline suy luận tốt hơn. Xem [answer_length_analysis.png](answer_length_analysis.png).

## 24. Error Analysis

Quy ước boundary: chọn **một cặp gold chung** có ít endpoint sai nhất; hòa thì tổng khoảng cách ký tự nhỏ nhất, rồi thứ tự annotation. `wrong_start`/`wrong_end` so với cặp đó; `wrong_span` ở đây là **cả hai biên sai**. `span_mismatch` nghĩa là không khớp bất kỳ cặp gold nào. Không ghép start của gold này với end của gold khác. Các metric text vẫn dùng mọi reference độc lập theo SQuAD.

Bốn bảng quota dưới tương ứng20 lượt ví dụ nhưng **15 ID độc nhất**: năm partial-match dùng lại các case boundary đã có context. Năm wrong-start đều end đúng; năm wrong-end đều start đúng; năm wrong-span đều sai cả hai. Các ví dụ minh họa được chọn theo khả năng đọc/giải thích, không phải mẫu ngẫu nhiên để ước lượng tỷ lệ nguyên nhân.

### Nhóm Wrong start, end đúng — 5 ví dụ thật

| ID | Gold index | Prediction span | F1 |
|---|---|---|---|
| 56bead5a3aeaaa14008c91ec | 0 | [568,579) | 50.000 |
| 56e761d037bdd419002c3f14 | 0 | [372,385) | 66.667 |
| 5725e44238643c19005ace38 | 0 | [400,433) | 85.714 |
| 57273455f1498d1400e8f48e | 0 | [458,520) | 82.353 |
| 57291fb4af94a219006aa0b3 | 1 | [417,475) | 94.118 |

### Wrong start, end đúng 1 — `56bead5a3aeaaa14008c91ec`

**Question:** How many teams have been in the Super Bowl eight times?

**Gold alternatives (giữ cả các reference trùng):** #0: "four" [575,579); #1: "four" [575,579); #2: "four" [575,579). So sánh boundary dùng gold #0.

**Prediction:** "one of four" `[568,579)`; EM=0.000%, F1=50.000%.

**Context gốc `[513,634)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
e Patriots, Dallas Cowboys, and Pittsburgh Steelers as one of four teams that have made eight appearances in the Super Bo
```

**Phân tích quan sát:** Câu hỏi cần số lượng đội, gold chỉ là “four”. Prediction lấy thêm “one of” ở bên trái nhưng kết thúc đúng ký tự 579. Đây là mở rộng biên start, không phải thiếu thông tin: cùng lát context đã có cụm “one of four teams”.

### Wrong start, end đúng 2 — `56e761d037bdd419002c3f14`

**Question:** What gets transferred to students who are receptive to the teacher?

**Gold alternatives (giữ cả các reference trùng):** #0: "passion" [378,385); #1: "passion" [378,385); #2: "passion" [378,385). So sánh boundary dùng gold #0.

**Prediction:** "their passion" `[372,385)`; EM=0.000%, F1=66.667%.

**Context gốc `[317,440)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
isposition towards the course content tend to transfer their passion to receptive students. These teachers do not teach by 
```

**Phân tích quan sát:** Câu hỏi hỏi thứ được truyền sang học sinh; cả hai span cùng chứa “passion”. Model giữ thêm sở hữu từ “their”, nên start sớm 6 ký tự và end đúng. Nội dung chính có mặt; lỗi quan sát được là mức độ tối giản của boundary.

### Wrong start, end đúng 3 — `5725e44238643c19005ace38`

**Question:** What damaged the color TV camera that Apollo 12 had taken into space?

**Gold alternatives (giữ cả các reference trùng):** #0: "the Sun" [426,433); #1: "Sun" [430,433); #2: "the Sun" [426,433); #3: "the Sun" [426,433); #4: "pointed into the Sun." [413,434). So sánh boundary dùng gold #0.

**Prediction:** "accidentally pointed into the Sun" `[400,433)`; EM=0.000%, F1=85.714%.

**Context gốc `[345,488)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
rface color television camera, but it was damaged when accidentally pointed into the Sun. They made two EVAs totaling 7 hours and 45 minutes. O
```

**Phân tích quan sát:** Prediction bao gồm cả mô tả hành động “accidentally pointed into” trước “the Sun”, trong khi gold được chọn chỉ giữ tác nhân Sun. Mọi reference được chấp nhận vẫn được tính EM/F1; reference cuối “pointed into the Sun.” giải thích F1 khá cao. Không thể suy ra truncation từ trường hợp này.

### Wrong start, end đúng 4 — `57273455f1498d1400e8f48e`

**Question:** Where did the Chinese Nationalists move the mausoleum away from advancing Chinese Communist forces? 

**Gold alternatives (giữ cả các reference trùng):** #0: "Kumbum Monastery or Ta'er Shi near Xining" [479,520); #1: "Kumbum Monastery" [479,495); #2: "Tibetan monastery of Kumbum Monastery" [458,495). So sánh boundary dùng gold #0.

**Prediction:** "Tibetan monastery of Kumbum Monastery or Ta'er Shi near Xining" `[458,520)`; EM=0.000%, F1=82.353%.

**Context gốc `[403,575)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
ers moved it another 200 km farther west to the famous Tibetan monastery of Kumbum Monastery or Ta'er Shi near Xining, which soon fell under Communist control. In early 195
```

**Phân tích quan sát:** Prediction bắt đầu ở “Tibetan monastery of”, sớm hơn reference được chọn “Kumbum Monastery or Ta'er Shi near Xining”, và cùng kết thúc tại 520. Các gold khác chấp nhận tên ngắn hơn nhưng không chấp nhận toàn bộ span model chọn; đây là một đáp án dài mang thêm mô tả loại địa điểm.

### Wrong start, end đúng 5 — `57291fb4af94a219006aa0b3`

**Question:** What did the Kenyan business people hope for when meeting with the Chinese?

**Gold alternatives (giữ cả các reference trùng):** #0: "support from China for a planned $2.5 billion railway from the southern Kenyan port of Mombasa to neighboring Uganda" [422,538); #1: "support from China for a planned $2.5 billion railway" [422,475); #2: "support from China for a planned $2.5 billion railway" [422,475). So sánh boundary dùng gold #1.

**Prediction:** "gain support from China for a planned $2.5 billion railway" `[417,475)`; EM=0.000%, F1=94.118%.

**Context gốc `[362,530)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
panied by 60 Kenyan business people [and hoped to] ... gain support from China for a planned $2.5 billion railway from the southern Kenyan port of Mombasa to neighborin
```

**Phân tích quan sát:** Gold ngắn nói “support from China ... railway”; prediction chỉ thêm động từ “gain” phía trước. Câu hỏi “hope for” khiến cả cụm động từ hợp ngữ cảnh, nhưng span annotation ngắn hơn. F1 cao phản ánh phần lớn token trùng, không phải prediction đúng EM.

### Nhóm Wrong end, start đúng — 5 ví dụ thật

| ID | Gold index | Prediction span | F1 |
|---|---|---|---|
| 56beace93aeaaa14008c91e1 | 0 | [403,426) | 80.000 |
| 56e7611500c9c71400d77055 | 0 | [137,166) | 40.000 |
| 5726e985dd62a815002e94da | 0 | [132,160) | 85.714 |
| 572940973f37b319004781a5 | 0 | [109,130) | 50.000 |
| 56bf3c633aeaaa14008c9581 | 0 | [69,82) | 66.667 |

### Wrong end, start đúng 1 — `56beace93aeaaa14008c91e1`

**Question:** What city did Super Bowl 50 take place in?

**Gold alternatives (giữ cả các reference trùng):** #0: "Santa Clara" [403,414); #1: "Santa Clara" [403,414); #2: "Santa Clara" [403,414). So sánh boundary dùng gold #0.

**Prediction:** "Santa Clara, California" `[403,426)`; EM=0.000%, F1=80.000%.

**Context gốc `[348,481)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
16, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasize
```

**Phân tích quan sát:** Question yêu cầu city; gold Santa Clara, prediction gồm cả California. Start đúng403 nhưng end426 vượt gold414. Đây là mở rộng địa danh theo cấp bang trong cùng câu, EM mất dù hai từ city đều được giữ.

### Wrong end, start đúng 2 — `56e7611500c9c71400d77055`

**Question:** In addition to schools, where else is popularly based authority effective?

**Gold alternatives (giữ cả các reference trùng):** #0: "governments" [137,148); #1: "governments" [137,148); #2: "governments" [137,148). So sánh boundary dùng gold #0.

**Prediction:** "governments and schools alike" `[137,166)`; EM=0.000%, F1=40.000%.

**Context gốc `[82,221)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
 order more effectively than dictatorial authority for governments and schools alike. They also claim that in these schools the preservatio
```

**Phân tích quan sát:** Question đã loại schools bằng cụm “In addition to schools”. Prediction vẫn lấy cả “governments and schools alike”, thay vì chỉ governments. Start đúng137; end quá dài18 ký tự. Quan hệ loại trừ của câu hỏi chưa được thể hiện đúng ở span chọn.

### Wrong end, start đúng 3 — `5726e985dd62a815002e94da`

**Question:** What type of authority are ambulatory care pharmacists given in the U.S. federal health care system?

**Gold alternatives (giữ cả các reference trùng):** #0: "full independent prescribing authority" [132,170); #1: "full independent prescribing authority" [132,170); #2: "full independent prescribing authority" [132,170). So sánh boundary dùng gold #0.

**Prediction:** "full independent prescribing" `[132,160)`; EM=0.000%, F1=85.714%.

**Context gốc `[77,225)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
ervice, and NIH) ambulatory care pharmacists are given full independent prescribing authority. In some states such North Carolina and New Mexico the
```

**Phân tích quan sát:** Gold là cụm danh từ “full independent prescribing authority”. Prediction dừng ở prescribing, bỏ danh từ cuối authority dù start132 đúng. Đây là partial answer với end sớm10 ký tự, không phải sai thực thể hoàn toàn.

### Wrong end, start đúng 4 — `572940973f37b319004781a5`

**Question:** What did the IPCC say was mistaken?

**Gold alternatives (giữ cả các reference trùng):** #0: "the date" [109,117); #1: "the date" [109,117); #2: "the date" [109,117). So sánh boundary dùng gold #0.

**Prediction:** "the date is incorrect" `[109,130)`; EM=0.000%, F1=50.000%.

**Context gốc `[54,185)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
for policymakers. The IPCC has since acknowledged that the date is incorrect, while reaffirming that the conclusion in the final su
```

**Phân tích quan sát:** Question hỏi điều gì bị nhầm; gold “the date”, model kéo dài sang vị ngữ “is incorrect”. Start109 đúng; end130 thay vì117. Model lấy thêm khẳng định về đối tượng, nên boundary không đúng dù giữ được danh từ date.

### Wrong end, start đúng 5 — `56bf3c633aeaaa14008c9581`

**Question:** What was the name of New Orleans' superdome at the time that Super Bowl 50 took place?

**Gold alternatives (giữ cả các reference trùng):** #0: "Mercedes-Benz Superdome" [69,92); #1: "Mercedes-Benz Superdome" [69,92); #2: "Mercedes-Benz Superdome" [69,92). So sánh boundary dùng gold #0.

**Prediction:** "Mercedes-Benz" `[69,82)`; EM=0.000%, F1=66.667%.

**Context gốc `[14,147)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
ntually narrowed the bids to three sites: New Orleans' Mercedes-Benz Superdome, Miami's Sun Life Stadium, and the San Francisco Bay A
```

**Phân tích quan sát:** Question hỏi tên superdome; model chỉ lấy Mercedes-Benz và bỏ Superdome. Start69 đúng, end82 sớm hơn gold92. Cụm tên riêng bị cắt mất thành tố, tạo F1 dương nhưng không exact match.

### Nhóm Wrong span, cả hai biên sai — 5 ví dụ thật

| ID | Gold index | Prediction span | F1 |
|---|---|---|---|
| 56be4db0acb8001400a502ef | 0 | [249,266) | 0.000 |
| 56e751fb00c9c71400d76faa | 0 | [247,255) | 0.000 |
| 570d4606b3d812140066d61a | 0 | [209,220) | 50.000 |
| 570d4a6bfed7b91900d45e14 | 0 | [151,162) | 40.000 |
| 570d4e6cb3d812140066d671 | 0 | [277,303) | 0.000 |

### Wrong span, cả hai biên sai 1 — `56be4db0acb8001400a502ef`

**Question:** Which NFL team won Super Bowl 50?

**Gold alternatives (giữ cả các reference trùng):** #0: "Denver Broncos" [177,191); #1: "Denver Broncos" [177,191); #2: "Denver Broncos" [177,191). So sánh boundary dùng gold #0.

**Prediction:** "Carolina Panthers" `[249,266)`; EM=0.000%, F1=0.000%.

**Context gốc `[122,321)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
eason. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game w
```

**Phân tích quan sát:** Cùng câu ghi “Denver Broncos defeated ... Carolina Panthers”. Model chọn đội đứng sau defeated thay vì đội thắng đứng trước. Đây là nhầm vai trong quan hệ thắng–thua, có đối tượng cạnh tranh ngay trong context; không khẳng định nguyên nhân nội tại của attention.

### Wrong span, cả hai biên sai 2 — `56e751fb00c9c71400d76faa`

**Question:** What is a typical name for learning that is not inside of the classroom?

**Gold alternatives (giữ cả các reference trùng):** #0: "field trips" [555,566); #1: "field trips" [555,566); #2: "field trips" [555,566). So sánh boundary dùng gold #0.

**Prediction:** "pedagogy" `[247,255)`; EM=0.000%, F1=0.000%.

**Context gốc `[192,621)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
ills. Different ways to teach are often referred to as pedagogy. When deciding what teaching method to use teachers consider students' background knowledge, environment, and their learning goals as well as standardized curricula as determined by the relevant authority. Many times, teachers assist in learning outside of the classroom by accompanying students on field trips. The increasing use of technology, specifically the ri
```

**Phân tích quan sát:** Question hỏi tên hoạt động học ngoài lớp; câu gần gold ghi “learning outside ... on field trips”. Prediction pedagogy xuất hiện ở câu khác nói tên cách dạy học. Hai span đều thuộc chủ đề giáo dục nhưng trả lời hai quan hệ khác nhau; đây là chọn nhầm thông tin có liên quan chủ đề.

### Wrong span, cả hai biên sai 3 — `570d4606b3d812140066d61a`

**Question:** How many dairy cows are there in Australia?

**Gold alternatives (giữ cả các reference trùng):** #0: "3 million" [87,96); #1: "3 million" [87,96); #2: "3 million" [87,96). So sánh boundary dùng gold #0.

**Prediction:** "2.4 million" `[209,220)`; EM=0.000%, F1=50.000%.

**Context gốc `[32,275)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
farming in Australia. It is home to 60% of Australia's 3 million dairy cattle and produces nearly two-thirds of the nation's milk, almost 6.4 billion litres. The state also has 2.4 million beef cattle, with more than 2.2 million cattle and cal
```

**Phân tích quan sát:** Question hỏi dairy cows ở Australia; context phân biệt 3 million dairy cattle và 2.4 million beef cattle. Model chọn số của beef cattle. Cả hai boundary đều khác gold; từ million vẫn trùng nên F1 không nhất thiết bằng0 mặc dù loại gia súc bị nhầm.

### Wrong span, cả hai biên sai 4 — `570d4a6bfed7b91900d45e14`

**Question:** What was the name of the eastern half of the colony of 1788?

**Gold alternatives (giữ cả các reference trùng):** #0: "New South Wales" [110,125); #1: "New South Wales" [110,125); #2: "New South Wales" [110,125). So sánh boundary dùng gold #0.

**Prediction:** "New Holland" `[151,162)`; EM=0.000%, F1=40.000%.

**Context gốc `[55,217)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
1788, Australia was divided into an eastern half named New South Wales and a western half named New Holland, under the administration of the colonial government i
```

**Phân tích quan sát:** Context đối chiếu eastern half=New South Wales và western half=New Holland. Question hỏi phía đông nhưng model chọn tên phía tây. Đây là ví dụ cặp thực thể cạnh tranh và sai quan hệ hướng, quan sát trực tiếp được từ câu đối chiếu.

### Wrong span, cả hai biên sai 5 — `570d4e6cb3d812140066d671`

**Question:** How are Victorian cabinet members chosen?

**Gold alternatives (giữ cả các reference trùng):** #0: "elected" [266,273); #1: "elected" [266,273); #2: "elected" [266,273). So sánh boundary dùng gold #0.

**Prediction:** "either house of parliament" `[277,303)`; EM=0.000%, F1=0.000%.

**Context gốc `[211,358)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
 political agenda. Cabinet consists of representatives elected to either house of parliament. It is responsible for managing areas of government th
```

**Phân tích quan sát:** Question hỏi cách chọn cabinet members; context dùng “representatives elected to either house of parliament”. Gold elected là cơ chế chọn; prediction either house of parliament lại trả lời nơi/cơ quan. Đây là nhầm loại thông tin cần trả lời, cả start và end đều sai.

Các nhãn nguyên nhân có bằng chứng trong ví dụ gồm boundary quá rộng/hẹp, partial answer, nhầm đối tượng cạnh tranh và nhầm loại thông tin. Không gán “insufficient context”, “truncation” hay “tokenization” làm nguyên nhân cho các case này khi chưa có bằng chứng feature/coverage tương ứng. Các cửa sổ có thể không chứa gold được gán CLS khi training, nhưng inference đã ghép mọi window; việc answer không nằm ở một window riêng không chứng minh bị mất khỏi toàn bộ document.

### Một lỗi vị trí có text hoàn toàn đúng

### Token/chuỗi lặp — `5729f5a03f37b31900478607`

**Question:** What type of T cells help with both innnate and adaptive immunity?

**Gold alternatives (giữ cả các reference trùng):** #0: "γδ T cells" [395,405); #1: "γδ" [395,397); #2: "γδ T cells" [395,405). So sánh boundary dùng gold #0.

**Prediction:** "γδ T cells" `[477,487)`; EM=100.000%, F1=100.000%.

**Context gốc `[340,542)`** (chỉ số toàn cục, lát cắt có thể bắt đầu/kết thúc giữa từ):
```text
 TCRs, such as CD1d-restricted Natural Killer T cells, γδ T cells straddle the border between innate and adaptive immunity. On one hand, γδ T cells are a component of adaptive immunity as they rearrange
```

**Phân tích quan sát:** Cả gold và prediction đều là “γδ T cells” nên EM=F1=100%, nhưng context nhắc lại chuỗi này ở hai vị trí. Annotation ở395:405, model chọn lần xuất hiện477:487. Đây là positional mismatch, không nên diễn giải thành câu trả lời sai nghĩa. Nó cho thấy kiểm tra lặp token/chuỗi và tách metric text khỏi boundary là cần thiết.

## 25. Wrong Start vs Wrong End

| Start | End | n |
|---|---|---|
| Đúng | Đúng | 7703 |
| Đúng | Sai | 622 |
| Sai | Đúng | 943 |
| Sai | Sai | 1302 |

Tổng wrong-start=**2.245** (=943+1.302), wrong-end=**1.924** (=622+1.302), both-wrong=**1.302**, any-span-mismatch=**2.867** (=943+622+1.302). Các tập start/end chồng nhau, không cộng2.245+1.924 để suy số câu lỗi độc nhất. Có**397** mẫu EM=1 nhưng vị trí không khớp gold, do normalization hoặc span lặp; vì thế2.867 không bằng số câu EM=0.

Start sai nhiều hơn end321 mẫu trong run này. Các case start-only đã đọc thường giữ end nhưng mở rộng sang bổ ngữ/động từ phía trái; case end-only có cả kéo dài sang clause lẫn cắt mất danh từ cuối. Một giả thuyết phù hợp là hai biên có mức mơ hồ khác nhau ở các cụm này. Tuy nhiên15 case chọn tay không chứng minh toàn bộ943/622 trường hợp đều cùng nguyên nhân; cần annotate mẫu đại diện và kiểm soát độ dài trước khi kết luận nguyên nhân thống kê. Không suy ra model head start có chất lượng thấp hơn trong mọi ngữ cảnh chỉ từ số đếm.

## 29. AI Usage

**AI-assisted code:** pipeline tokenizer/alignment, start/end loss, metrics, decoding, experiments, visualization và các kiểm thử. **What was generated:** báo cáo thực nghiệm này, chọn case thật, giải thích boundary và đối chiếu bốn prediction; không có số liệu minh họa giả. **What was modified:** thay phiếu phân tích chờ sinh viên của lượt chuẩn bị bằng báo cáo dựa full run. **What was verified:** mọi ID trên được tìm trong predictions.json; flags/offsets/substring/metric lấy dữ liệu thật, bảng tổng hợp đối chiếu CSV và JSON cùng run_id. [analysis_manifest.json](.runtime/full_run/analysis_manifest.json) lưu ID quota để kiểm tra độc lập.

Mục29 của đề gốc yêu cầu sinh viên tự làm một số phần. Người dùng ở lượt mới đã yêu cầu AI hỗ trợ toàn bộ; điều này được công khai và **không** được trình bày như sự chấp thuận ngoại lệ từ giảng viên. Báo cáo không giả nhận người viết là sinh viên hoặc bịa trải nghiệm học tập cá nhân. Prediction gốc không sửa sau khi có kết quả.

## 30. Reproducibility

| Trường | Giá trị |
|---|---|
| Run ID | 8faacbbd16e7d738dcf71bbdd8f51d12ffe7b805c1b43f8ea62df17775b0772c |
| Model / tokenizer | distilbert-base-uncased |
| Model revision | 12040accade4e8a0f71eabdb258fecc2e7e948be |
| Dataset / revision | rajpurkar/squad / 7b6d24c440a36b6815f21b70d25016731768db1f |
| Train / validation / heldout | 79.214 / 8.385 / 10.570; title90/10; public dev không official test |
| Lengths / stride / question cap | 128,256,384 / 64 / 48 |
| Batch / precision / epochs / LR | 32 / BF16 autocast CUDA / 2 / 3e-5 |
| Optimizer / seed | AdamW (weight_decay0.01), gradient clip1.0 / 42 |
| Hardware | NVIDIA GeForce RTX 5060 Ti |
| Selected checkpoint | .runtime/full_run/checkpoints/length_384, epoch2 |
| Prediction SHA256 | e50b5d4c8b0dbf5bad9892b893610a34c0d6c163602c1601e4e878adcb8e2deb |

Thư viện/phiên bản, hardware, source hashes, split IDs/fingerprints và artifact hashes nằm trong run_manifest/split_manifest; từng checkpoint có training_metadata, experiment_config và epoch tốt nhất. Cache/checkpoint thuộc `.runtime/full_run`, giữ nguyên runtime cũ. Không hứa bit-identical giữa phần cứng/phiên bản khác. Core là SQuAD1; chưa đánh giá no-answer SQuAD2 và score raw logits không phải xác suất trả lời được. Chỉ có một seed và hai epoch, nên chênh lệch nhỏ chưa có ước lượng độ bất định nhiều seed.
