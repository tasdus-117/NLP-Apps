# LAB 10 — Tính toán và lý thuyết


## 6. Calculation Exercise 1: Start / End

Chỉ số **0-based** của dãy context giả định trong đề:

| Token | The | capital | of | France | is | Paris | . |
|---|---|---|---|---|---|---|---|
| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |

- `Paris`: start = 5, end = 5.
- `capital of France`: start = 1, end = 3; gồm cả token 3.

## 7. Calculation Exercise 2: Start Loss

Dùng logarithm tự nhiên: `L_start = -ln(0.60) = 0.5108256238` nats. Tổng xác suất của bảng là 1. Với xác suất gold end là `p_end`, `L_end = -ln(p_end) = = -ln(0.60) = 0.5108256238`.

## 8. QA Loss

`L = (CE(start_logits, gold_start) + CE(end_logits, gold_end))/2`.

Softmax được tính theo chiều **vị trí token**, không theo hai nhãn start/end. Cross-entropy nhận logits trực tiếp và áp dụng log-softmax ổn định số. Cần hai prediction vì vị trí bắt đầu không quyết định độ dài đáp án; cùng start có thể kết thúc ở nhiều nơi. Hai argmax độc lập có thể cho end < start, nên decoding phải ràng buộc cặp span hợp lệ; training loss này không tự áp đặt ràng buộc đó.

Các window không chứa **toàn bộ** đáp án dùng CLS làm sentinel label cho cả start/end trong mã tham khảo. Đây là nhãn ở mức window, không biến SQuAD 1.1 thành bài toán no-answer. Khi inference core luôn tìm context span, không dùng CLS để tuyên bố document không có đáp án.

Kết quả QA loss xác định từ đề là `L = (0.5108256238 - 0.5108256238)/2 = 0`. 

## 12. Calculation Exercise 3: Character Span

Context nguyên văn: `The capital of France is Paris.`

| Đại lượng | Giá trị tham khảo |
|---|---|
| Character đầu của `Paris` (0-based) | 25 |
| Character cuối inclusive | 29 |
| Character span half-open | `[25, 30)` |
| Kiểm tra substring | `context[25:30] == "Paris"` |
| Word index khi split whitespace | 5 (word là `Paris.`) |
| Token index theo dãy **giả định mục 6** | start = end = 5 |
| Token index của question/context pair thật đã chạy | start = end = 14; token `paris` (DistilBERT uncased) |

Một word có thể gồm nhiều subword. Offsets của context token quy chiếu vào context gốc; offsets của question token quy chiếu vào question. Phải dùng `sequence_ids == 1` trước khi alignment. Nếu annotation bắt đầu/kết thúc giữa token, nhãn là token bao trùm; decode nguyên token có thể lấy thêm ký tự và làm EM giảm.

Xem [tokenizer_alignment.ipynb](tokenizer_alignment.ipynb), mục 10–16, để xem token/offset thật và kiểm thử mid-token, truncation, ngoài window, partial span, overflow. Notebook ghi `.runtime/full_run/tokenizer_alignment.json` khi chạy.

Kiểm tra character: tiền tố `The capital of France is ` dài 25 ký tự; `Paris` dài 5 nên end exclusive là 30. Dấu chấm ở vị trí 30 không thuộc đáp án. Các vị trí của `Paris` là `P:25, a:26, r:27, i:28, s:29`. Với pair tokenizer, vị trí tương ứng phải cộng question/special tokens theo kết quả thực, không cộng một hằng số cho mọi câu hỏi.

Bằng chứng token index thực tế: `.runtime/full_run/tokenizer_alignment.json`, khóa `calculation`, từ tokenizer `distilbert-base-uncased` revision `12040accade4e8a0f71eabdb258fecc2e7e948be`. Với question `What is the capital of France?`, có 7 question tokens; `[CLS]` ở 0, question ở 1–7, `[SEP]` ở 8, context bắt đầu ở 9; `Paris` là context token thứ 5 nên pair index `9+5=14`. Cách tính này áp dụng cho đúng question/tokenizer đã nêu, không phải mọi câu hỏi.

## 19. Token-level F1 for QA

Ví dụ đề: gold `Paris France`, prediction `France`. Sau normalization: gold có 2 token, prediction có 1; overlap = 1. Precision = 1/1 = 1; recall = 1/2 = 0.5; `F1 = 2×1×0.5/(1+0.5) = 2/3 ≈ 0.6667`, tức 66.67 trên thang 0–100. EM = 0 vì chuỗi normalized không bằng nhau.

Overlap dùng **multiset/Counter**, không phải set: gold `red red blue`, prediction `red blue blue` có overlap 2, không phải 3; P = R = F1 = 2/3. Với nhiều gold hợp lệ, tính EM/F1 với từng gold rồi lấy maximum **riêng cho từng metric**. Chuẩn hóa SQuAD: lowercase, bỏ ASCII punctuation, bỏ article tiếng Anh `a/an/the`, chuẩn hóa whitespace. F1 token ở đây dùng whitespace sau normalization, khác subword của Transformer.

Ví dụ bổ sung tính đầy đủ: gold `The red red blue!`, prediction `a red blue blue`. Sau normalization lần lượt là `red red blue` và `red blue blue`. Gold Counter là `{red:2, blue:1}`, prediction Counter là `{red:1, blue:2}`; giao có `min(2,1)+min(1,2)=2` token. Cả hai chuỗi dài 3 token, nên `P=2/3`, `R=2/3`, `F1=2/3≈66.67%`, còn `EM=0`. Nếu chỉ lấy set, sẽ đếm sai việc phân bố số lần lặp. Nếu golds là `["red red blue", "red blue blue"]`, cùng prediction đạt `EM=1, F1=1` nhờ reference thứ hai. Metric là điểm tương đồng token giữa hai câu trả lời, không phải accuracy của việc dự đoán nhãn token trên toàn context.

