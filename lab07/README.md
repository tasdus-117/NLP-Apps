# LAB 07 — Transformer Architecture

Bài lab phân loại cảm xúc IMDb bằng Transformer tự triển khai và LSTM. Notebook đã lưu kết quả các bài 11–12, 19–23 và 25; báo cáo phân tích lỗi và đối chiếu dự đoán nằm trong các file Markdown riêng.

## Cấu trúc thư mục

```text
lab07/
├── README.md
├── transformer_classification.ipynb  # Code, bảng và output thực nghiệm
├── transformer_block.py             # Self-attention và Transformer block tự ghép
├── positional_encoding.py           # File khung; PE thực thi đang nằm trong notebook
├── calculations.md
├── prediction.md
├── error_analysis.md
├── reflection.md
├── results.csv                      # 10 lượt train độc lập
├── images/                          # Toàn bộ ảnh của workspace
├── artifacts/
│   ├── experiments_19_22/           # Checkpoint, JSON, metric và diversity
│   ├── experiment_25/               # Checkpoint và kết quả Mean/CLS-style
│   ├── error_analysis/              # Dự đoán test và tensor attention 6 mẫu
│   └── summary/                     # Dữ liệu vẽ hình và manifest nguồn
```

Toàn bộ ảnh đã chuyển vào **`images/`**. Không có ảnh kết quả rải ở thư mục lab, `artifacts/` hoặc `tmp/pdfs/`. Các ảnh nhúng trong notebook vẫn được giữ để xem output trực tiếp.

## Bốn hình chính

| Hình | Nội dung | Dữ liệu gốc |
|---|---|---|
| [Positional encoding](images/positional_encoding.png) | PE minh họa d_model=4, pos=0,1,2 | [Ma trận PE](artifacts/summary/positional_encoding.npz) |
| [Attention heads](images/attention_heads.png) | Layer 1, 4 heads, validation index 86, loại PAD | [Tensor attention](artifacts/summary/validation_attention.pt) |
| [Learning curves](images/learning_curves.png) | Loss/F1 train và validation qua 3 epochs, Transformer/LSTM | [History từng epoch](artifacts/summary/training_history.csv) |
| [Confusion matrix](images/confusion_matrix.png) | 25.000 test, Transformer chuẩn 4 heads/T=128 | [Ma trận đếm](artifacts/summary/confusion_matrix.csv), [dự đoán từng mẫu](artifacts/error_analysis/test_predictions.csv) |

Bốn hình này là kết quả xuất từ dữ liệu thật, không còn ảnh giữ chỗ. Confusion matrix có hàng là nhãn thật, cột là nhãn dự đoán; thứ tự Negative, Positive; accuracy=0.81088.

## Các hình bổ sung

| Nội dung | File trong images/ |
|---|---|
| Số heads: F1 và thời gian | [heads_f1_time.png](images/heads_f1_time.png) |
| Diversity trên 3 câu validation | [Mẫu 1](images/attention_example_1.png), [mẫu 2](images/attention_example_2.png), [mẫu 3](images/attention_example_3.png) |
| Transformer so với LSTM | [transformer_vs_lstm_learning_curves.png](images/transformer_vs_lstm_learning_curves.png) |
| Sequence length | [length_f1_time_memory.png](images/length_f1_time_memory.png) |
| Mean so với CLS-style pooling | [pooling_comparison.png](images/pooling_comparison.png) |
| Attention của 3 mẫu đúng và 3 mẫu sai | Sáu file `error_attention_test_*.png`, được liên kết trong [error_analysis.md](error_analysis.md) |
| Ảnh trích PDF, không phải kết quả model | `pages_1.png`, `pages_9.png`, `pages_17.png`, `existing_images.png` |

## Dữ liệu và kết quả

Dùng `stanfordnlp/imdb` như LAB 05: 20.000 train, 5.000 validation, 25.000 test chính thức; seed 42. Tokenizer `lower().split()`, vocabulary chỉ fit trên train. Chi tiết split/vocabulary nằm trong [data_protocol.json](artifacts/experiments_19_22/data_protocol.json).

[results.csv](results.csv) chỉ giữ metric và cấu hình tổng hợp, **một hàng cho mỗi lượt train độc lập**. Checkpoint 4 heads/T=128 được dùng ở nhiều bài nhưng không đếm lại thành lượt train mới. Accuracy, macro Precision/Recall/F1 và tỷ lệ token giữ lại ở dạng 0–1. Thông tin không áp dụng được để trống. Hai cấu hình pooling bài 25 đều có CLS, khác baseline không CLS của bài 19.

Tensor attention, ma trận PE, loss từng epoch và nhãn/dự đoán từng mẫu được giữ riêng để có thể vẽ lại; CSV tổng hợp không thay thế chúng. Xem [nguồn dữ liệu từng hình](artifacts/summary/README.md), [manifest kiểm tra](artifacts/summary/manifest.json) và [bản đồ đường dẫn ảnh cũ/mới](artifacts/summary/image_locations.json).

Thời gian huấn luyện giữ nguyên từ lần chạy thực tế và có bao gồm validation. Bài 19–22 không warm-up riêng; bài 25 warm-up rồi reset trọng số/optimizer/seed trước khoảng đo. Đây là kết quả một seed trên RTX 5060 Ti, không phải kết luận tốc độ cho mọi phần cứng.

## Notebook và báo cáo

- [Notebook](transformer_classification.ipynb): phần khung đầu có chỉ dẫn tới các mục đã triển khai; mục 13 hiển thị bảng tổng hợp. Các lệnh lưu hình của experiment hiện trỏ tới `images/`; output dài dùng chế độ cuộn. Giữ chỉ số cell để các trích dẫn trong báo cáo còn đúng.
- [prediction.md](prediction.md): dự đoán ban đầu, không sửa theo kết quả.
- [error_analysis.md](error_analysis.md): bằng chứng thực tế và heatmap của 6 mẫu test.
- [reflection.md](reflection.md): đối chiếu dự đoán và nội dung phản tư/khai báo AI hiện có.
- [calculations.md](calculations.md): phần tính toán cần sinh viên tự thực hiện hoặc xác minh theo policy.

Không cần train lại chỉ để xem CSV, hình hoặc output đã lưu. Việc code chạy thành công và có đủ hình không xác nhận sinh viên đã tự hiểu hoặc hoàn thành các phần policy yêu cầu làm độc lập. Phạm vi sử dụng AI và xác nhận của sinh viên được ghi trong `reflection.md`; không mặc định toàn bộ quá trình đã tuân thủ policy.

