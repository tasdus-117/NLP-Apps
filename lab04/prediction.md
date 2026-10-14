### **Prediction 1: TF-IDF + Logistic Regression có khả năng hoạt động tốt không?**

*   **Prediction:** Có, mô hình này hoạt động khá tốt.
*   **Reason:** TF-IDF cho thông tin trọng số từ khóa quan trọng trong văn bản khi kết hợp với Logistic Regression nó huấn luyện rất nhanh, dễ hội tụ và ít bị overfit trên dữ liệu văn bản chuẩn.
*   **Confidence:** Cao

---

### **Prediction 2: Average Word2Vec có chắc chắn tốt hơn TF-IDF không?**

*   **Prediction:** Không chắc chắn.
*   **Reason:** Mặc dù Average Word2Vec nắm bắt được ngữ nghĩa của từ, nhưng việc lấy trung bình cộng vector của tất cả các từ trong câu lại **làm mất đi thứ tự từ** và **bỏ qua trọng số từ khóa quan trọng**. Trong nhiều trường hợp, TF-IDF với từ khóa đặc trưng lại vượt trội hoặc tương đương Average Word2Vec, trừ khi dùng các kiến trúc phức tạp hơn như LSTM, CNN hoặc Transformers.
*   **Confidence:** Cao

---

### **Prediction 3: Nếu tăng vocabulary size thì accuracy chắc chắn tăng không?**

*   **Prediction:** Không.
*   **Reason:** Tăng kích thước từ vựng sẽ đưa thêm nhiều từ ít xuất hiện hoặc nhiễu vào mô hình. Điều này làm tăng độ phức tạp của không gian đặc trưng, dễ dẫn đến hiện tượng **overfitting** và làm giảm khả năng tổng quát hóa trên tập test, khiến accuracy có thể giảm hoặc không đổi.
*   **Confidence:** Rất cao

---

### **Prediction 4: Nếu bỏ stopwords thì performance chắc chắn tăng không?**

*   **Prediction:** Không chắc chắn điều này phụ thuộc vào bài toán.
*   **Reason:** Stopwords (như *the, is, at, a*...) thường mang ít ý nghĩa ngữ nghĩa trong các bài toán phân loại chủ đề, việc loại bỏ chúng giúp giảm chiều dữ liệu. Tuy nhiên, trong các bài toán đòi hỏi cấu trúc câu, ngữ pháp, hoặc phân tích cảm xúc khó ví dụ: *"not good"* khác hoàn toàn với *"good"* nếu bỏ chữ *"not"*, việc loại bỏ stopwords có thể làm mất đi thông tin quan trọng và làm giảm performance.
*   **Confidence:** Cao

---

### **Prediction 5: Accuracy và F1 có thể cho kết luận khác nhau không?**

*   **Prediction:** Có, hoàn toàn có thể đặc biệt rõ ràng trên dữ liệu mất cân bằng.
*   **Reason:** Accuracy tính tổng tỷ lệ dự đoán đúng trên toàn bộ các lớp, dễ bị bóp méo bởi lớp đa số. Trong khi đó, F1-score là trung bình điều hòa của Precision và Recall, phản ánh khả năng dự đoán thực sự của mô hình trên từng lớp. Một mô hình có Accuracy rất cao nhưng F1-score lại cực kỳ thấp.
*   **Confidence:** Rất cao