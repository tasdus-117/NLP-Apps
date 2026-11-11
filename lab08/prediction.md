## 11. Prediction Before Experiment

### Prediction 1 — Pretrained và train from scratch

**Prediction:** Pretrained Transformer được fine-tune trên IMDb sẽ đạt kết quả tốt hơn BERT cùng kiến trúc train from scratch trên cùng dữ liệu huấn luyện.

**Reason:** Pretraining giúp mô hình học sẵn các biểu diễn về từ vựng, ngữ cảnh và quan hệ ngôn ngữ, tạo nền tảng để fine-tuning nhận diện cảm xúc trong bài đánh giá phim. Transformer train from scratch phải học cả biểu diễn ngôn ngữ lẫn cách phân loại từ dữ liệu IMDb có nhãn.

**Expected evidence:** Trên cùng tập đánh giá cố định và cùng metric accuracy, accuracy của pretrained Transformer sau fine-tuning cao hơn accuracy của Transformer train from scratch khi hai mô hình dùng cùng tập dữ liệu huấn luyện. Nếu chạy nhiều seed, accuracy trung bình cao hơn và chênh lệch nhất quán qua các seed sẽ củng cố dự đoán.

### Prediction 2 — Freeze encoder và chỉ train classification head

**Prediction:** C. Kém hơn — freeze encoder và chỉ train classification head sẽ có hiệu năng thấp hơn fine-tuning toàn bộ mô hình.

**Reason:** Khi encoder bị đóng băng, các biểu diễn đã học không thể điều chỉnh để nhận diện tốt hơn những đặc điểm cảm xúc của IMDb, chẳng hạn phủ định và cách diễn đạt đánh giá phim. Fine-tuning toàn bộ mô hình cho phép cả encoder và classification head thích nghi với bài toán.

**Expected evidence:** Với cùng checkpoint ban đầu, cùng dữ liệu huấn luyện, cùng tập đánh giá và cùng metric accuracy, cấu hình freeze encoder có accuracy thấp hơn cấu hình fine-tuning toàn bộ mô hình. Nếu chạy nhiều seed, chênh lệch accuracy trung bình theo hướng này và nhất quán qua các seed sẽ ủng hộ lựa chọn C.

### Prediction 3 — Giảm lượng dữ liệu huấn luyện

**Prediction:** Khi xét các mức 100%, 50%, 25% và 10% dữ liệu huấn luyện, Transformer train from scratch sẽ giảm hiệu năng mạnh hơn pretrained Transformer khi chuyển từ 100% xuống 10%.

**Reason:** Khi ít dữ liệu gán nhãn, pretrained Transformer vẫn có các biểu diễn ngôn ngữ đã học từ pretraining để hỗ trợ phân loại cảm xúc. Transformer train from scratch phụ thuộc nhiều hơn vào dữ liệu IMDb để học các biểu diễn này, nên dự kiến bị ảnh hưởng mạnh hơn khi lượng dữ liệu huấn luyện giảm.

**Expected evidence:** Huấn luyện mỗi mô hình ở cả bốn mức dữ liệu với các tập con phân tầng, lồng nhau và dùng chung giữa hai mô hình; đánh giá trên cùng tập đánh giá cố định bằng accuracy. Với từng mô hình, tính mức giảm theo điểm phần trăm so với mốc 100% của chính nó: Δ = accuracy(100%) − accuracy(10%). Kết quả Δ của Transformer train from scratch lớn hơn Δ của pretrained Transformer sẽ ủng hộ dự đoán; các mức 50% và 25% cho biết xu hướng suy giảm ở giữa. Nếu chạy nhiều seed, so sánh mức giảm trung bình của mỗi mô hình.


### Đối chiếu sau thí nghiệm

Các dự đoán trên được giữ là giả thuyết trước thí nghiệm. Phần này đối chiếu với output mới đã lưu trong `tokenization.ipynb`: 3 epoch, seed 42, max length 128 và chọn checkpoint bằng validation macro-F1.

| Prediction | Bằng chứng thực nghiệm | Đối chiếu |
|---|---|---|
| 1 | Pretrained Accuracy 87,968%, macro-F1 0.879678; scratch 79,828%, macro-F1 0.796637. Chênh lệch Accuracy 8,140 điểm %. | Được ủng hộ trong cấu hình đã thử. |
| 2 | Fine-tuning Accuracy 87,968%; Frozen 76,172%. Chênh lệch 11,796 điểm %. | Được ủng hộ trong cấu hình đã thử. |
| 3 | Khi từ 100% xuống 10%, Accuracy pretrained giảm 2,392 điểm %, scratch giảm 29,828 điểm %. | Được ủng hộ trong cấu hình đã thử. |

Cả hai model có cùng 109.483.778 tham số; so sánh này không dùng một Transformer nhỏ hơn cho scratch. Chi tiết cả bốn mức dữ liệu, chi phí huấn luyện và giới hạn diễn giải ở `error_analysis.md`. Đây là một seed; log còn có cảnh báo attention backward không hoàn toàn deterministic nên chưa đánh giá độ ổn định qua nhiều lần chạy.
