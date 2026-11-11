## 9. Calculation Exercise 1 — MLM Loss

### Tính loss ban đầu

Dùng log tự nhiên:

$$L = -\ln P(\text{eats}) = -\ln(0.70) \approx 0.3567.$$

### Trả lời câu hỏi

1. Khi xác suất của `eats` là 0.95:
   $$L = -\ln(0.95) \approx 0.0513.$$
   Loss **giảm** từ 0.3567 xuống 0.0513 so với ban đầu.

2. Khi xác suất của `eats` là 0.01:
   $$L = -\ln(0.01) \approx 4.6052.$$
   Loss **tăng** từ 0.3567 lên 4.6052 so với ban đầu.

3. Cross-entropy phù hợp với MLM vì dự đoán token bị mask là bài toán phân loại trên toàn bộ từ vựng. Nhãn one-hot có giá trị 1 tại token đúng `eats` và 0 tại các token khác, nên:
   $$L = -\sum_{v \in V} y_v \ln P(v) = -\ln P(\text{token đúng}).$$
   Vì vậy, loss giảm khi mô hình gán xác suất cao hơn cho token đúng và tăng mạnh khi xác suất đó gần 0.

## 10. Calculation Exercise 2 — Tokenization

### Kết quả tokenization

1. **Tokenizer/checkpoint:** Bài tính dùng `BertTokenizerFast`/`AutoTokenizer.from_pretrained` với cùng checkpoint `google-bert/bert-base-uncased` được khai báo trong `pretrained_classification.py`. Câu đầu vào của bài tính là `"The cats are playing."`; dùng `padding=True`, `truncation=True`, `return_tensors="pt"`.

2. **Token sequence đầy đủ:** `["[CLS]", "the", "cats", "are", "playing", ".", "[SEP]"]`.

3. **`input_ids`:** `[101, 1996, 8870, 2024, 2652, 1012, 102]`.

4. **Special tokens thực sự xuất hiện:** `[CLS]` ở vị trí đầu tiên (index 0) và `[SEP]` ở vị trí thứ 7 (index 6). `special_tokens_mask` tương ứng là `[1, 0, 0, 0, 0, 0, 1]`; không có `[PAD]` trong chuỗi.

5. **`attention_mask`:** `[1, 1, 1, 1, 1, 1, 1]`. Đối chiếu độ dài: `len(attention_mask) = len(input_ids) = 7`. Tất cả vị trí đều được chú ý, gồm cả special tokens; không có padding vì batch chỉ có một câu.

### So sánh số lượng

6. Câu gốc có **4 từ**: `The`, `cats`, `are`, `playing` (không tính dấu chấm là một từ). Đầu ra có **7 token**, gồm **2 special tokens** và **5 subword token** sau khi loại special tokens: `["the", "cats", "are", "playing", "."]`. Dấu chấm được tokenizer tách thành một token thường nên được tính trong 5 subword token. Không từ nào bị tách thành nhiều mảnh trong đầu ra này; `The` được tokenizer uncased chuyển thành `the`. Chênh lệch là **7 = 4 từ + 1 token dấu chấm + 2 special tokens**.

Ghi chú đối chiếu: mục 9 là phép tính lý thuyết; mục 10 là ví dụ tokenization riêng. Output mới của mục 14–19 không lưu ví dụ câu này, nên không dùng nó làm bằng chứng cho kết quả phân loại mới. File `pretrained_classification.py` hiện dùng ba câu minh họa khác trong `main()`.
