# Reflection - Lab 01

**1. Prediction nào của em sai?**
Dựa trên tài liệu, em đã dự đoán tập dữ liệu 36K documents sẽ có khoảng 300.000 đến 400.000 terms và ma trận TF-IDF sẽ rất thưa (tỉ lệ zero entries > 90%). Tuy nhiên, trong quá trình thực nghiệm, một dự đoán ngầm về việc tiền xử lý văn bản càng nhiều càng tốt đã bị chứng minh là không hoàn toàn chính xác. Việc loại bỏ stop words có thể dẫn đến mất ngữ nghĩa. Ngoài ra, việc xử lý dấu câu triệt để sẽ làm mất các thông tin quan trọng như ngắt quãng câu, biểu hiện cảm xúc, tiền tệ, email, hashtag. 

**2. Kết quả nào bất ngờ nhất?**
Kết quả bất ngờ nhất là sự khác biệt trong cách cài đặt cốt lõi giữa thư viện `sklearn` và việc tự code. Cụ thể, `sklearn` áp dụng Smooth IDF để triệt tiêu 0 và tự động sử dụng chuẩn hóa L2 Norm. Về mặt tokenization, `sklearn` sử dụng regex trong khi tự cài đặt chỉ dùng hàm `split()`. Một điểm bất ngờ khác là Pipeline tạo ra ma trận thưa nhất không đồng nghĩa với kết quả tìm kiếm tốt nhất; một hệ thống tốt cần tập từ vựng tối ưu, dù gọn và đủ nghĩa.

**3. Experiment nào cung cấp evidence mạnh nhất?**
Phần thực nghiệm phân tích lỗi cung cấp bằng chứng mạnh mẽ nhất về cách hoạt động của mô hình. Hệ thống hoạt động xuất sắc với các truy vấn như "Medical image classification" hay "deep learning healthcare" vì các tài liệu trả về có sự trùng khớp từ vựng lên đến 100% với truy vấn. Điều này chứng minh rõ ràng rằng mô hình phụ thuộc hoàn toàn vào sự xuất hiện của các từ vựng cụ thể.

**4. Failure case quan trọng nhất là gì?**
Trường hợp thất bại quan trọng nhất là truy vấn "natural language processing" đối với Doc 2. Doc 2 bị đẩy xuống thứ hạng dưới với độ tương đồng thấp. Lỗi này chỉ ra điểm yếu cốt lõi: TF-IDF biểu diễn số thay vì chữ và không hiểu được từ viết tắt. Từ "NLP" và cụm "natural language processing" dù tương đương về mặt ngữ nghĩa nhưng không trùng lặp ký tự, nên hệ thống coi chúng là khác nhau. Thêm vào đó, truy vấn "transformer language model" cũng cho thấy hệ thống dễ thất bại do hình thái từ.

**5. Nếu được xây lại search engine, em sẽ thay đổi điều gì?**
Dựa trên những thất bại đã phân tích, để giải quyết vấn đề về similarity nghĩa, em sẽ bổ sung các kỹ thuật có khả năng sử dụng context hay ngữ nghĩa thay vì chỉ so khớp từ vựng đơn thuần. 

**6. AI đã được sử dụng ở những phần nào và đóng góp cụ thể là gì?**
AI đã hỗ trợ trong việc viết code và sinh ra các câu trả lời dựa trên định hướng và ý tưởng ban đầu của em. Sau đó, em kiểm tra, tinh chỉnh và viết lại các nội dung này để đảm bảo câu từ mạch lạc, dễ hiểu và phản ánh đúng quá trình tư duy cá nhân.