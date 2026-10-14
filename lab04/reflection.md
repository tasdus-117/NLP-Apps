## 8. Đánh giá giả định (Hypothesis / Predictions) so với Thực nghiệm

| Giả định (Prediction) | Lý luận ban đầu | Kết quả kiểm chứng thực tế trên IMDb |
| :--- | :--- | :--- |
| **1. TF-IDF + Logistic Regression** | Hoạt động tốt, train nhanh, ít overfit. | **Đúng hoàn toàn:** Mô hình đạt **Accuracy = 0.8800**, thời gian huấn luyện cực nhanh ($\approx 8.62\text{s}$), là baseline mạnh mẽ và ổn định. |
| **2. Average Word2Vec vs. TF-IDF** | Không chắc Word2Vec tốt hơn vì lấy trung bình làm mất thứ tự từ và trọng số từ khóa. | **Đúng hoàn toàn:** Phép lấy trung bình vector làm pha loãng ý nghĩa và mất cấu trúc phủ định, khiến TF-IDF với từ khóa đặc trưng giữ vững ưu thế. |
| **3. Tăng Vocabulary Size** | Không chắc tăng accuracy, dễ bị overfitting và tăng nhiễu từ hiếm. | **Đúng hoàn toàn:** Thử nghiệm từ Unigram (20,000 features, $F_1 = 0.8801$) lên Uni+Bigram (50,000 features, $F_1 = 0.8827$) cho thấy hiệu năng tăng không đáng kể nhưng thời gian train tăng gấp đôi. |
| **4. Bỏ Stopwords** | Không chắc chắn, phụ thuộc bài toán vì có thể làm mất từ quan trọng (như từ phủ định). | **Đúng hoàn toàn:** Việc loại bỏ vô điều kiện có thể cắt đứt các cụm từ ngữ cảnh cục bộ (ví dụ: cấu trúc phủ định), đòi hỏi phải dùng n-gram hoặc giữ lại các từ liên kết quan trọng. |
| **5. Accuracy vs. F1-score** | Có thể khác nhau rõ rệt trên dữ liệu mất cân bằng (Accuracy bị bóp méo, F1 phản ánh chuẩn). | **Đúng hoàn toàn:** Trên tập IMDb cân bằng, các chỉ số Accuracy, Precision, Recall và F1 đồng loạt bằng nhau ($\approx 0.8800$), minh chứng gián tiếp cho việc Accuracy sẽ bị bóp méo nếu dữ liệu mất cân bằng. |

---

## 9. AI Assistance Statement

* **Tool:** Google Gemini AI (Collaborative Assistant)
* **Purpose:** Hỗ trợ giải thích lý thuyết, xây dựng và sửa lỗi mã nguồn Python (`scikit-learn`, `datasets`) và đóng gói tài liệu báo cáo.
* **Generated:** Toàn bộ mã nguồn Python thực nghiệm baseline, các đoạn code trực quan hóa kết cấu vector, bảng so sánh hiệu năng và các khung phân tích định tính.
* **Modified:** Mã nguồn và các lập luận lý thuyết đã được tùy chỉnh, chạy kiểm thử thực tế và đồng bộ hóa kết quả trực tiếp với dữ liệu trên máy local.
* **Verified:** Sinh viên đã trực tiếp chạy các đoạn mã trên môi trường cục bộ, kiểm tra độ chính xác của các bảng kết quả thực nghiệm (`Accuracy`, `F1-score`, `Confusion Matrix`), xác thực trọng số đặc trưng và rà soát toàn bộ nội dung báo cáo trước khi nộp.