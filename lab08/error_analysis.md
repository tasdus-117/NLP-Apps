## 20. Error Analysis

### Kết quả IMDb test của model ứng dụng

| Chỉ số | Giá trị |
|---|---:|
| Accuracy | 0.879680 |
| Macro-Precision | 0.879702 |
| Macro-Recall | 0.879680 |
| Macro-F1 | 0.879678 |
| Số dự đoán đúng | 21.992 |
| Số dự đoán sai | 3.008 |

| Nhãn | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.876786 | 0.883520 | 0.880140 | 12.500 |
| positive | 0.882619 | 0.875840 | 0.879216 | 12.500 |

Confusion matrix đã lưu ở `confusion_matrix.png`, hàng là nhãn thật, cột là nhãn dự đoán:

| True / Pred | negative | positive |
|---|---:|---:|
| negative | 11.044 | 1.456 |
| positive | 1.552 | 10.948 |

Có 1.456 false positives và 1.552 false negatives nếu xem positive là lớp dương. Tổng hai loại lỗi bằng 3.008, khớp histogram và Accuracy 21.992 / 25.000 = 87,968%.

### Bảng phân tích lỗi thực tế

| Domain / ID | Trích đoạn văn bản | True | Pred | Confidence | Error type | Hypothesis |
|---|---|---|---|---:|---|---|
| IMDb #1 | “Worth the entertainment value of a rental, esp...” | negative | positive | 0.924527 | Chưa xác định | Phần mở đầu có lời khen nhưng output không lưu toàn văn. Cần đọc đủ review và đầu vào token hóa để phân biệt cảm xúc pha trộn, phủ định hay thiếu ngữ cảnh. |
| IMDb #4 | “First off let me say, If you haven't enjoyed a...” | negative | positive | 0.987808 | Chưa xác định | Đây là lỗi tự tin cao đã xác nhận; đoạn văn bị rút gọn trong bảng hiển thị nên chưa thể xác định sắc thái hoặc nguyên nhân. |
| Amazon #2 | “Batteries died within a year ...”; “The design is nice and convenient. However, after about a year, the batteries would not hold a charge.” | negative | positive | 0.7175 | context; mixed sentiment | Lời khen thiết kế và giai đoạn đầu có thể cạnh tranh với phê bình độ bền sau một năm. Cần kiểm tra đầu vào token hóa trước khi quy nguyên nhân cho truncation. |
| Amazon #9 | “Not an ultimate guide”; “I enjoyed the format and tone”; “I did not feel that she imparted any insider secrets”; “this book may be a tremendous help” | negative | positive | 0.5404 | ambiguous sentiment; context | Review phân biệt người mới với người đã nghiên cứu kỹ. Mô hình có thể chưa gắn lời khen và lời chê với đúng đối tượng sử dụng. Confidence gần 0.5 thể hiện hai lớp cạnh tranh mạnh. |
| Amazon #11 | “Not!”; “this is not him, it is a gimmick, very well orchestrated.” | negative | positive | 0.8661 | negation; context | Cụm “very well orchestrated” có bề mặt tích cực nhưng đi cùng sự phủ nhận tính xác thực. Có thể mô hình ưu tiên lời khen bề mặt hơn đánh giá tổng thể; đây là giả thuyết từ văn bản. |
| Amazon #51 | “Great Beginning...poor ending”; “breathtaking in the beginning”; “the ending, not too great.” | positive | negative | 0.9481 | mixed sentiment; label ambiguity | Khen mở đầu nhưng chê kết thúc, trong khi nhãn tổng là positive. Mô hình có thể ưu tiên phần kết tiêu cực. Không đủ bằng chứng để nói dataset gán nhãn sai. |
| Amazon #63 | “A historical gem whose title obscures its intent”; “Started laughing”; “enjoyed the book”; “worth the price” | negative | positive | 0.9710 | label ambiguity; possible irony | Nhiều cụm khen xuất hiện trong văn bản nhưng nhãn là negative. Có thể liên quan sắc thái châm biếm hoặc cách gán nhãn; chưa xác định được nguyên nhân chỉ từ một mẫu. |

### Nhận xét

Các lỗi được lưu cho thấy lời khen và lời chê trong cùng review có thể gây khó khăn. Những mẫu Amazon #2, #9 và #51 yêu cầu tổng hợp đánh giá theo thời gian hoặc đối tượng; #11 có phủ định cùng tín hiệu khen bề mặt. Tuy nhiên, đây là các mẫu được chọn để minh họa, không đủ để ước lượng tần suất từng loại lỗi trên toàn tập.

Hai lỗi IMDb được hiển thị đều có confidence trên 0.90. Amazon #51 và #63 cũng sai dù confidence lần lượt là 0.9481 và 0.9710. Confidence cao vì vậy không bảo đảm dự đoán đúng. Chưa kiểm tra token hóa từng mẫu mới nên không khẳng định phần kết luận nào đã bị cắt. Các ID, confidence và số token của bảng cũ không được dùng làm bằng chứng cho checkpoint hiện tại.

## 21. Confidence Analysis

### Cách tính

Với logits của mỗi mẫu, lấy softmax rồi chọn xác suất của lớp dự đoán:

```python
probabilities = torch.softmax(logits, dim=-1)
confidence, predictions = probabilities.max(dim=-1)
correct = predictions.eq(labels)
```

Confidence không dùng nhãn thật; correctness được tính bằng cách so sánh dự đoán với nhãn thật. Với bài toán hai lớp và chọn xác suất lớn nhất, confidence thuộc khoảng [0.5, 1]. Quy ước mô tả: High ≥ 0.90; Medium từ 0.70 đến dưới 0.90; Low dưới 0.70.

### Kết quả đã xác nhận

Histogram `confidence_analysis.png` dùng chính các dự đoán IMDb ở mục 19, cùng bins cho hai nhóm, trục Y là **số mẫu**. Có **21.992 mẫu đúng** và **3.008 mẫu sai**, tổng 25.000, khớp confusion matrix. Cả hai nhóm đều có mẫu ở vùng confidence cao; chỉ nhìn histogram không thể suy ra chính xác số mẫu theo ngưỡng 0.90 hoặc confidence trung bình.

| Domain / ID | True | Pred | Confidence | Nhóm | Kết quả |
|---|---|---|---:|---|---|
| IMDb #0 | negative | negative | 0.979951 | High | Đúng |
| IMDb #1 | negative | positive | 0.924527 | High | Sai |
| IMDb #3 | negative | negative | 0.829769 | Medium | Đúng |
| IMDb #4 | negative | positive | 0.987808 | High | Sai |
| Amazon #9 | negative | positive | 0.5404 | Low | Sai |
| Amazon #51 | positive | negative | 0.9481 | High | Sai |

```python
summary = imdb_predictions.copy()
summary["group"] = np.select(
    [summary["confidence"] >= 0.90, summary["confidence"] >= 0.70],
    ["High", "Medium"], default="Low",
)
confidence_table = summary.groupby("group", observed=True).agg(
    examples=("correct", "size"),
    correct=("correct", "sum"),
    accuracy=("correct", "mean"),
    mean_confidence=("confidence", "mean"),
).reindex(["High", "Medium", "Low"])
confidence_table["incorrect"] = confidence_table["examples"] - confidence_table["correct"]
confidence_table["confidence_minus_accuracy"] = (
    confidence_table["mean_confidence"] - confidence_table["accuracy"]
)
display(confidence_table)
```

### Diễn giải

Confidence thể hiện mức ưu tiên lớp của mô hình; correctness thể hiện mức khớp nhãn thật. Các lỗi confidence cao đã lưu đủ để bác bỏ cách hiểu “confidence cao luôn đúng”. Để kết luận mô hình quá tự tin ở mức toàn tập, cần so sánh accuracy với confidence trung bình theo bins hoặc dùng chỉ số calibration phù hợp. Không suy ra mức quá tự tin cụ thể từ số liệu của lần chạy cũ. Nếu hiệu chỉnh xác suất, cần học tham số trên validation rồi đánh giá trên test độc lập.

## 28. Error Analysis — Câu hỏi bắt buộc

### Q1. Model nào hưởng lợi nhiều hơn khi training data ít?

**Observation:** Pretrained giữ chất lượng tốt hơn khi dữ liệu ít. Lợi thế Accuracy ở 10% và 25% đều lớn hơn ở 100%; khoảng cách không giảm đều tại mọi mức.

**Evidence:** Các tập con phân tầng, lồng nhau gồm 2.000, 5.000, 10.000 và 20.000 mẫu. Hai model dùng chung tập con ở từng mức và cùng 3 epoch, seed 42, batch 16, learning rate 2e-5, max length 128. Validation 5.000 và test 25.000 mẫu được giữ cố định. Mỗi mức khởi tạo lại model, **kể cả 100%**; không tái sử dụng kết quả mục 14.

| Training data | # Examples | Pretrained Accuracy | Scratch Accuracy | Chênh lệch Accuracy (điểm %) | Pretrained macro-F1 | Scratch macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| 10% | 2.000 | 0.855760 | 0.500000 | 35.576 | 0.855710 | 0.333333 |
| 25% | 5.000 | 0.866880 | 0.500120 | 36.676 | 0.866874 | 0.333671 |
| 50% | 10.000 | 0.868720 | 0.758000 | 11.072 | 0.868615 | 0.754226 |
| 100% | 20.000 | 0.879680 | 0.798280 | 8.140 | 0.879678 | 0.796637 |

Từ 100% xuống 10%, pretrained giảm **2,392 điểm % Accuracy**, scratch giảm **29,828 điểm %**. Macro-F1 giảm tương ứng khoảng **2,397** và **46,330 điểm %**. Ở 10%, cặp Accuracy 0.5 và macro-F1 0.333333 phù hợp với việc dự đoán gần như chỉ một lớp; output không lưu confusion matrix nên chưa xác định lớp nào.

**Interpretation:** Kết quả ủng hộ Prediction 3: trọng số pretrained giúp mô hình học từ ít nhãn hơn trong cấu hình này. Tuy nhiên, giảm dữ liệu cũng làm giảm số bước optimizer khi giữ nguyên epoch. Scratch có thể chưa hội tụ và chưa được tối ưu learning rate riêng. Không suy rộng kết luận thành giới hạn tốt nhất của scratch. Ở 25%, log thời gian có chênh lệch lớn khoảng 983,65 giây giữa hai model; chưa đủ thông tin để quy cho kiến trúc hoặc pretraining, nên phần này tập trung vào Accuracy và F1.

### Q2. Nếu pretrained model tốt hơn from-scratch, điều đó chứng minh điều gì?

**Observation:** Mục 14 cho thấy pretrained tốt hơn scratch trong cùng ngân sách 3 epoch.

**Evidence:** Cùng BERT, head `Linear(768, 2)`, tổng **109.483.778 tham số**, split, tokenizer, seed, batch, LR và max length. Mỗi lần huấn luyện có head mới; không giả định hai head có trọng số khởi tạo giống hệt nhau.

| Model | Accuracy | Macro-F1 | Training time (giây) | # Parameters |
|---|---:|---:|---:|---:|
| From scratch | 0.798280 | 0.796637 | 505.956207 | 109.483.778 |
| Pretrained | 0.879680 | 0.879678 | 504.666896 | 109.483.778 |

Pretrained cao hơn **8,140 điểm % Accuracy**, khoảng **8,304 điểm % macro-F1** và thời gian ngắn hơn khoảng **1,29 giây**. Phạm vi đo gồm train, validation và lưu checkpoint, không gồm tải model/dữ liệu, token hóa, nạp lại checkpoint, test hoặc chi phí pretraining ban đầu.

**Interpretation:** Kết quả ủng hộ Prediction 1 về lợi ích chuyển giao trong điều kiện đã thử. Chênh lệch thời gian nhỏ ở một lần đo không chứng minh fine-tuning luôn nhanh hơn scratch khi cùng kiến trúc. Kết quả tốt hơn cũng không chứng minh khả năng hiểu ngôn ngữ tổng quát; model vẫn có lỗi và có thể dựa vào tín hiệu bề mặt. Log có cảnh báo attention backward không hoàn toàn deterministic, nên seed cố định chưa bảo đảm mọi lần chạy giống hệt nhau.

### Q3. Nếu frozen model gần bằng fine-tuned model, điều đó có ý nghĩa gì?

**Observation:** Đây là tình huống giả định. Kết quả thực tế chưa cho thấy hai phương pháp gần bằng nhau: Frozen Accuracy **76,172%**, fine-tuning **87,968%**.

**Evidence:** Cần so sánh cùng split, tokenizer, max length, cách chọn checkpoint và ngân sách huấn luyện, xác định mức chênh lệch chấp nhận được, rồi kiểm tra độ ổn định qua nhiều seed. Một lần chạy gần nhau chưa chứng minh hai phương pháp tương đương.

**Interpretation:** Nếu kết quả vẫn gần nhau sau các kiểm tra, biểu diễn pretrained có thể đã đủ hữu ích để head học ranh giới nhãn. Frozen khi đó có thể giảm chi phí. Tuy nhiên, cũng có thể fine-tuning chưa được tối ưu; không kết luận encoder không cần thích nghi trong mọi tác vụ.

### Q4. Nếu fine-tuning tốt hơn frozen model rất nhiều, điều đó gợi ý điều gì?

**Observation:** Đây là xu hướng thực tế ở mục 15.

| Mode | Accuracy | Macro-F1 | Training time (giây) | # Trainable |
|---|---:|---:|---:|---:|
| Frozen | 0.761720 | 0.761461 | 182.525694 | 1.538 |
| Fine-tuning | 0.879680 | 0.879678 | 504.666896 | 109.483.778 |

**Evidence:** Fine-tuning cao hơn **11,796 điểm % Accuracy**, khoảng **11,822 điểm % macro-F1**, và tốn thêm **322,14 giây**. Hai chế độ cùng pretrained checkpoint, dữ liệu và cấu hình 3 epoch. Frozen khóa toàn encoder và giữ encoder ở `eval()` khi train head. Fine-tuning tái sử dụng đúng cấu hình từ mục 14.

**Interpretation:** Encoder thích nghi với IMDb mang lại lợi ích rõ trong cấu hình này, phù hợp Prediction 2. Head vẫn học được từ biểu diễn pretrained cố định, nhưng cập nhật encoder cho phép thay đổi biểu diễn theo tác vụ. Chưa kiểm tra trực tiếp biểu diễn nên không khẳng định model đã học một quy tắc ngôn ngữ cụ thể; LR cho Frozen cũng chưa được tối ưu riêng.

### Q5. Nếu performance giảm mạnh trên domain khác, nguyên nhân có thể là gì?

**Observation:** Lần chạy này **không quan sát thấy giảm trên Domain B**. Amazon test đạt kết quả cao hơn IMDb test.

| Domain | Dataset | # Examples | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| A — phim | stanfordnlp/imdb, test | 25.000 | 0.879680 | 0.879678 |
| B — sản phẩm | fancyzhx/amazon_polarity, test | 400.000 | 0.900883 | 0.900730 |

**Evidence:** Dùng cùng model fine-tune trên IMDb, chọn bằng validation IMDb, cùng tokenizer và max length 128. Domain B dùng **toàn bộ 400.000 mẫu test chính thức**, ghép title với content, nhãn 0=negative và 1=positive, không có nhãn trung tính. Amazon cao hơn khoảng **2,120 điểm % Accuracy** và **2,105 điểm % macro-F1**. Các số Amazon được làm tròn theo bảng output; không xem các chữ số sau đó là độ chính xác đầy đủ.

**Interpretation:** Domain shift không nhất thiết làm giảm điểm trên mọi dataset. Hai tập khác nhau về nội dung, độ khó, độ dài và quy trình gán nhãn; kết quả này chưa chứng minh model đã thích nghi hoàn hảo với sản phẩm. Nếu gặp sự suy giảm ở domain khác, có thể kiểm tra từ vựng, tín hiệu cảm xúc, phân bố nhãn, truncation và nhãn mơ hồ. Các lỗi Amazon ở mục 20 cho thấy vẫn có khó khăn với cảm xúc pha trộn và phủ định, nhưng không thể quy toàn bộ khoảng cách metric cho một nguyên nhân.

### Q6. Nếu sequence length tăng nhưng performance gần như không đổi trong khi training time tăng mạnh, có nên tiếp tục tăng sequence length không?

**Observation:** Nếu lợi ích gần như bằng không trong khi chi phí tăng rõ, em sẽ ưu tiên chiều dài ngắn hơn đáp ứng yêu cầu. Tuy nhiên, kết quả thực tế hiện tại vẫn cải thiện qua cả ba mức 64, 128 và 256.

**Evidence:** Mục 17 khởi tạo lại pretrained model cho mỗi mức, giữ cùng split, batch 16, 3 epoch và padding cố định tới max length. Thời gian gồm train, validation và lưu checkpoint; bộ nhớ là **CUDA max_memory_allocated**, reset riêng từng lần, đơn vị MiB.

| Max length | Accuracy | Macro-F1 | Training time (giây) | GPU peak memory (MiB) |
|---|---:|---:|---:|---:|
| 64 | 0.832920 | 0.832794 | 314.738910 | 2172.637695 |
| 128 | 0.879680 | 0.879678 | 512.594636 | 2578.470215 |
| 256 | 0.914760 | 0.914737 | 986.185847 | 3895.986816 |

Từ 64 lên 128: Accuracy tăng **4,676 điểm %**, F1 tăng khoảng **4,688 điểm %**, thời gian tăng **197,86 giây**, peak memory tăng **405,83 MiB**. Từ 128 lên 256: Accuracy tăng **3,508 điểm %**, F1 tăng khoảng **3,506 điểm %**, thời gian tăng **473,59 giây**, bộ nhớ tăng **1317,52 MiB**. Từ 64 lên 256: Accuracy tăng **8,184 điểm %**, F1 tăng khoảng **8,194 điểm %**, thời gian tăng **671,45 giây** và bộ nhớ tăng **1723,35 MiB**. Thời gian 256 khoảng 1,92 lần 128.

**Interpretation:** Chiều dài lớn hơn có thể giữ thêm nội dung cuối review nhưng tăng chi phí attention O(T²) theo độ dài sau padding. Tổng thời gian và bộ nhớ toàn model không nhất thiết tăng đúng T² vì còn các thành phần khác. Trong ba mức đã thử, 256 có test Accuracy và F1 cao nhất, đồng thời tốn tài nguyên nhất; chưa thể kết luận đây là chiều dài tối ưu ngoài phạm vi đã thử. Việc chọn cấu hình triển khai nên dựa trên validation, nhu cầu chất lượng và tài nguyên, tránh liên tục điều chỉnh bằng test.
