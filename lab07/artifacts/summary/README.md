# Dữ liệu dùng để vẽ hình

Toàn bộ ảnh nằm trong **../../images/**. Thư mục này chỉ giữ dữ liệu số và metadata, không chứa PNG.

| Hình | Dữ liệu được giữ lại | Nguồn |
|---|---|---|
| [positional_encoding.png](../../images/positional_encoding.png) | positional_encoding.npz: pe (3,4), positions, d_model | Cấu hình minh họa bài 11 |
| [attention_heads.png](../../images/attention_heads.png) | validation_attention.pt: weights (4,N,N), tokens, ID, layer, checkpoint | Checkpoint chuẩn bài 19; lưu 3 mẫu validation bài 20; hình dùng index 86 |
| [learning_curves.png](../../images/learning_curves.png) | training_history.csv: train/validation loss và F1 từng epoch | 30 dòng epoch của 10 lượt train; hình dùng cặp baseline Transformer/LSTM |
| [confusion_matrix.png](../../images/confusion_matrix.png) | confusion_matrix.csv: hàng thật, cột dự đoán | ../error_analysis/test_predictions.csv, đủ 25.000 test của checkpoint chuẩn |

[results.csv](../../results.csv) là bảng tổng hợp với 10 lượt train độc lập. Cột experiment có thể liệt kê nhiều bài khi checkpoint được dùng lại; không cộng lặp thời gian. Các metric và tokens_retained_ratio ở dạng fraction 0–1. Tỷ lệ token đo trên train, không tính PAD/CLS; trường không áp dụng được để trống.

Attention memory là lý thuyết một tensor scores (B=64, H=heads, T=max_length, float32=4 bytes), không phải GPU peak memory. Hai model bài 25 cùng có CLS, content limit=127; baseline bài 19 không CLS, content limit=128.

Thời gian giữ nguyên từ lượt huấn luyện gốc, gồm train + validation + sao chép best state + logging; không gồm test/xuất hình. Bài 19–22 không warm-up riêng; bài 25 warm-up rồi bỏ cập nhật và reset. Không dùng một lần đo làm kết luận tốc độ ổn định cho mọi cấu hình.

manifest.json giữ nguồn, checkpoint và SHA256 các output. image_locations.json ghi đường dẫn cũ/mới và SHA256 của 21 ảnh; ảnh được di chuyển nguyên vẹn, không đổi nội dung. Checkpoint và dữ liệu bằng chứng giữ nguyên trong artifacts/.
