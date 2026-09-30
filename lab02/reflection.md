### Phần 24: Reflection

* **Câu 1:** 
  Nếu tăng $n$, mô hình nhận thêm thông tin về **ngữ cảnh dài hơn**, giúp nắm bắt tốt hơn cấu trúc cục bộ và sự phụ thuộc xa giữa các từ trong câu.

* **Câu 2:** 
  Tăng $n$ làm sparsity tăng vì không gian tổ hợp của n-gram phình trướng theo cấp số mũ ($|V|^n$), trong khi kích thước của corpus thực tế là hữu hạn, khiến phần lớn các n-gram có thể xảy ra không xuất hiện trong tập huấn luyện.

* **Câu 3:** 
  Smoothing (làm mịn) cần thiết để **tránh xác suất bằng 0** khi mô hình gặp các n-gram mới trên tập kiểm thử, từ đó cứu mô hình khỏi lỗi Perplexity bằng vô cực và giúp hệ thống tính toán được xác suất cho toàn bộ câu.

* **Câu 4:** 
  Perplexity đo lường **độ bối rối** của mô hình khi dự đoán một tập dữ liệu; giá trị này phản ánh mức độ tự tin và khả năng dự đoán chính xác của mô hình đối với các từ tiếp theo.

* **Câu 5:** 
  Không. Một mô hình có perplexity thấp hơn **không hẳn luôn tạo ra văn bản tốt hơn** đối với con người, vì Perplexity chỉ đo lường khớp thống kê bề mặt chứ không hiểu về ngữ nghĩa sâu, tính mạch lạc logic dài hạn hay dụng ý thực tế của đoạn văn.

* **Câu 6:** 
  N-gram language model thất bại ở chỗ nó **thiếu khả năng hiểu ngữ nghĩa và phụ thuộc tầm xa**, chỉ dựa vào việc đếm tần suất bề mặt theo cửa sổ trượt ngắn hạn, trong khi con người hiểu ngôn ngữ dựa trên ngữ cảnh toàn cục, cú pháp phức tạp và tri thức thế giới.

* **Câu 7:** 
  Không. Nếu context dài 100 từ, **trigram hoàn toàn không thể sử dụng thông tin của 97 từ đầu**, vì giới hạn của trigram chỉ nhìn lại 2 từ trước đó ($n=3$, tức $n-1 = 2$ từ context), hoàn toàn mù lòa trước các thông tin ở xa hơn. Đây chính là điểm giới hạn lớn nhất thúc đẩy sự ra đời của neural language models (như RNN, LSTM và Transformer).

* **Câu 8:** AI đã được sử dụng ở những phần nào và đóng góp cụ thể là gì?
AI đã hỗ trợ trong việc viết code và sinh ra các câu trả lời dựa trên định hướng và ý tưởng ban đầu của em. Sau đó, em kiểm tra, tinh chỉnh và viết lại các nội dung này để đảm bảo câu từ mạch lạc, dễ hiểu và phản ánh đúng quá trình tư duy cá nhân.
