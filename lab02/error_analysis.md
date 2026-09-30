### 2 trường hợp dự đoán đúng

* Trường hợp 1:
  * Context: natural language
  * Model prediction: processing
  * Expected: processing
  * Probability: 0.8524
  * Nguyên nhân: Dữ liệu huấn luyện đủ lớn. Cụm từ natural language processing xuất hiện rất dày đặc trong tập dữ liệu, giúp mô hình học được xác suất chuyển tiếp cực kỳ chính xác.

* Trường hợp 2:
  * Context: machine
  * Model prediction: learning
  * Expected: learning
  * Probability: 0.6130
  * Nguyên nhân: Bigram machine learning rất phổ biến và tập huấn luyện cung cấp đủ tần suất thống kê. Ngữ cảnh một từ là hoàn toàn đủ để mô hình nhận diện từ tiếp theo trong ngữ cảnh chuyên ngành.

---

### 2 trường hợp dự đoán sai

* Trường hợp 3:
  * Context: a highly sophisticated
  * Model prediction: UNK
  * Expected: cyberattack
  * Probability: 0.0012
  * Nguyên nhân: Giới hạn từ vựng. Từ này xuất hiện với tần suất quá thấp trong tập huấn luyện nên bị loại bỏ trong bước xây dựng từ vựng, khiến mô hình phải gán nhãn khuyết và không thể dự đoán chính xác.

* Trường hợp 4:
  * Context: the cat eats
  * Model prediction: the
  * Expected: fish
  * Probability: 0.0005
  * Nguyên nhân: Tác dụng phụ của thuật toán làm mịn kết hợp với dữ liệu thưa. Cụm bốn từ này chưa từng xuất hiện trong tập huấn luyện. Khi thuật toán làm mịn phân bổ lại xác suất cho toàn bộ không gian từ vựng, các từ có tần suất độc lập rất cao như the vô tình nhận được điểm xác suất nhỉnh hơn từ đúng, dẫn đến việc mô hình dự đoán lệch hướng.