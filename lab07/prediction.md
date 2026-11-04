# 8. Prediction trước experiment

## 1. Một Transformer có thể xử lý các token trong sequence song song hơn RNN không?

- **Prediction:** Em dự đoán có: Transformer có thể xử lý các vị trí token song song hơn RNN khi toàn bộ sequence đầu vào đã có sẵn.
- **Reason:** Trong mỗi lớp Transformer, các phép tính trên nhiều token có thể thực hiện đồng thời, còn RNN cần trạng thái ở bước trước để tính bước tiếp theo. Tuy nhiên, khi sinh văn bản tự hồi quy, Transformer vẫn phải sinh lần lượt từng token.
- **Confidence:** 99%.

## 2. Nếu bỏ positional encoding, Transformer có còn biết thứ tự token không?

- **Prediction:** Em dự đoán không, nếu không có bất kỳ nguồn thông tin vị trí nào, kể cả attention mask mang thông tin thứ tự.
- **Reason:** Self-attention khi đó dựa vào nội dung token và có tính tương biến với phép hoán vị: hoán vị đầu vào sẽ làm đầu ra theo token bị hoán vị tương ứng. Việc đầu ra vẫn nằm ở các chỉ số tương ứng trong tensor không có nghĩa mô hình nhận biết được thứ tự token.
- **Confidence:** 99%.

## 3. Nếu hai câu có cùng tập token nhưng khác thứ tự, Transformer không có positional information có thể phân biệt chúng không?

- **Prediction:** Em dự đoán mô hình không thể phân biệt chúng dựa trên thứ tự, với giả định hai câu có cùng token và cùng số lần xuất hiện của mỗi token, đồng thời không có bất kỳ nguồn thông tin vị trí nào.
- **Reason:** Hai đầu ra theo token có thể khác nhau ở cách sắp xếp, nhưng chỉ là các biểu diễn bị hoán vị tương ứng, không phải bằng chứng mô hình hiểu thứ tự. Nếu dùng pooling bất biến với hoán vị như mean hoặc sum, hai câu sẽ có cùng biểu diễn tổng hợp; chọn hoặc ghép đầu ra theo chỉ số để phân biệt sẽ đưa thông tin vị trí vào bước đọc đầu ra, trái với giả định.
- **Confidence:** 99%.

## 4. Hai attention heads có nhất thiết học cùng một attention pattern không?

- **Prediction:** Em dự đoán không: hai attention heads có thể học các attention pattern khác nhau, nhưng cũng có thể học các pattern tương tự.
- **Reason:** Mỗi head có các tham số chiếu riêng nên có thể chú ý đến những quan hệ khác nhau giữa các token. Tuy nhiên, cơ chế nhiều head không bắt buộc các pattern phải khác nhau.
- **Confidence:** 98%.

## 5. Tăng số attention heads có chắc chắn làm accuracy tăng không?

- **Prediction:** Em dự đoán không: tăng số attention heads có thể làm accuracy tăng, gần như không đổi hoặc giảm.
- **Reason:** Hiệu quả phụ thuộc vào dữ liệu, nhiệm vụ, kích thước mô hình và cách huấn luyện. Khi giữ nguyên chiều biểu diễn tổng, tăng số head thường làm giảm chiều của mỗi head, nên không bảo đảm mô hình học tốt hơn.
- **Confidence:** 98%.

## 6. Transformer có luôn nhanh hơn RNN trên mọi sequence length không?

- **Prediction:** Em dự đoán không: khả năng tính song song cao hơn không bảo đảm Transformer chạy nhanh hơn RNN ở mọi độ dài sequence và trên mọi phần cứng.
- **Reason:** Với sequence ngắn, chi phí khởi tạo và thực thi các phép toán có thể làm lợi thế song song không đáng kể; với sequence dài, full self-attention có chi phí tính toán và bộ nhớ tăng theo bình phương độ dài, trong khi RNN thường tăng tuyến tính khi giữ nguyên kích thước mô hình. Tốc độ thực tế còn phụ thuộc GPU/CPU, batch size, cách triển khai và chế độ huấn luyện hay sinh tự hồi quy.
- **Confidence:** 98%.
