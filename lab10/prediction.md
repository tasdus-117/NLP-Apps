# LAB 10 — Prediction trước thực nghiệm

## 9. Prediction Before Experiment

### Prediction 1

Model có thể chọn nhầm khi nhiều đoạn lặp từ của câu hỏi nhưng chỉ một đoạn trả lời đúng quan hệ được hỏi. Tôi dự đoán baseline overlap chịu ảnh hưởng rõ hơn Transformer vì baseline không mô hình hóa quan hệ giữa các token. Nhóm question/context overlap cao có thể tốt hơn trung bình nhờ dễ tìm thông tin, nhưng không bảo đảm đúng ở các distractor. Độ tin tưởng: vừa. Kiểm chứng bằng EM/F1/support của các nhóm high/low theo ngưỡng cố định 0.5, đồng thời đọc các ví dụ cạnh tranh thật; đây là so sánh mô tả, không đủ kết luận nhân quả.

### Prediction 2

Đáp án dài sẽ dễ sai boundary hơn, nên EM có xu hướng giảm mạnh hơn F1 khi chuyển từ nhóm 1 sang 7+ subword. Không kỳ vọng điểm giảm đơn điệu ở mọi nhóm vì loại câu hỏi và support khác nhau. Độ tin tưởng: vừa. Kiểm chứng bằng bốn nhóm 1, 2–3, 4–6, 7+ subword của tokenizer cùng model; báo cả EM/F1 và số mẫu. Kiểm tra riêng giới hạn decoder 30 token, tránh quy mọi trường hợp dài không thể decode cho khả năng học của model.

### Prediction 3

Thêm các câu không liên quan vào đầu context có thể làm giảm nhẹ EM/F1 và thay đổi một số span, vì đổi vị trí tương đối và số window; Transformer không nhất thiết bất biến dù đáp án vẫn nguyên vẹn. Sliding windows sẽ tránh mất toàn bộ thông tin do cắt đuôi đơn giản, nên không dự đoán suy giảm quá lớn chỉ vì chiều dài. Độ tin tưởng: thấp đến vừa. Kiểm chứng paired trên toàn bộ heldout có thể biến đổi: giữ question/model/config cố định, chèn cùng một prefix nhiễu định trước, cộng độ dài prefix vào mọi answer_start, kiểm tra substring chính xác, báo delta EM/F1 và số prediction thay đổi cùng ví dụ thật.

### Prediction 4

Đảo thứ tự các câu có thể ảnh hưởng một phần vì làm đổi discourse, vị trí đáp án và biên window, nhưng các đáp án nằm trọn trong một câu còn đủ tín hiệu cục bộ có thể giữ nguyên. Tôi dự đoán nhiều câu vẫn trả lời đúng song không bất biến hoàn toàn; không khẳng định mức giảm lớn hơn phép thêm nhiễu. Độ tin tưởng: thấp. Kiểm chứng paired original/reversed trên cùng tập eligible: chỉ giữ ví dụ mà mọi gold nằm trọn một câu; đảo các lát câu nguyên văn, ánh xạ lại từng character offset bằng vị trí lát, assert gold substring. Báo support và lý do loại mẫu không đủ điều kiện; không dùng tập này để chọn hyperparameter/checkpoint.

### Quy ước đã chốt trước training

Toàn bộ SQuAD1.1: train gốc chia theo title 90/10 với seed42; public dev là heldout nội bộ, không phải official hidden test. Ba max_length128/256/384 dùng cùng data, stride64, question cap48, batch32, BF16 CUDA, epochs2, lr3e-5, AdamW, decoder cap30. Chọn epoch/cấu hình bằng validation F1 (EM rồi loss khi hòa), sau đó mới xem heldout. Nếu BF16/batch32 không chạy được vì tài nguyên, phải ghi lỗi và chốt lại cấu hình chung trước full run, không âm thầm làm nhẹ từng cấu hình.

PREDICTION_READY = YES
