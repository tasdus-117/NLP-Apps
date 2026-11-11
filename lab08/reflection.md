## 24. AI Usage

**AI-assisted code:** Em dùng AI để soạn prompt triển khai `pretrained_classification.py` và các thí nghiệm 14–19 trong `tokenization.ipynb`.

**What was generated:** Toàn bộ phần code và  triển khai các ý chính của em.

**What was modified:** Bổ sung các ý kiến còn thiếu và check lại số liệu, ở các phần training model

**What was verified:** Em fine-tune lại model từ model của AI

## 27. Individual Learning Check

1. **Pretraining khác fine-tuning ở đâu?**  
   Pretraining giúp mô hình học các đặc trưng ngôn ngữ từ một lượng lớn văn bản. Fine-tuning tiếp tục huấn luyện mô hình trên dữ liệu của một nhiệm vụ cụ thể, chẳng hạn phân loại cảm xúc.

2. **MLM là gì?**  
   MLM (Masked Language Modeling) là bài toán che một số token trong câu và yêu cầu mô hình dự đoán các token gốc dựa vào ngữ cảnh xung quanh.

3. **Tại sao cần tokenizer?**  
   Tokenizer chia văn bản thành các token và chuyển chúng thành ID để mô hình có thể xử lý.

4. **Subword token giải quyết vấn đề gì?**  
   Subword giúp biểu diễn từ hiếm hoặc từ chưa gặp bằng cách tách chúng thành các phần nhỏ hơn có trong từ vựng.

5. **`[CLS]` thường được sử dụng như thế nào?**  
   Với BERT, biểu diễn của `[CLS]` ở lớp cuối thường được đưa vào classification head để dự đoán nhãn cho cả văn bản.

6. **Attention mask dùng để làm gì?**  
   Attention mask đánh dấu token hợp lệ bằng `1` và vị trí padding bằng `0`, giúp mô hình bỏ qua padding khi tính attention.

7. **Frozen encoder nghĩa là gì?**  
   Encoder vẫn tạo biểu diễn cho đầu vào, nhưng trọng số của nó được giữ cố định trong lúc huấn luyện classification head.

8. **Fine-tuning thay đổi những tham số nào?**  
   Fine-tuning toàn bộ cập nhật tham số của cả encoder và classification head. Nếu chỉ fine-tune một phần, quá trình huấn luyện cập nhật các tham số được phép huấn luyện.

9. **Vì sao pretrained model có thể hữu ích khi downstream dataset nhỏ?**  
   Mô hình có thể tận dụng các biểu diễn ngôn ngữ đã học trong pretraining, nên có thể học nhiệm vụ mới với ít dữ liệu gán nhãn hơn.

10. **Pretrained model có đảm bảo không bị domain shift không?**  
    Không. Mô hình fine-tune trên review phim có thể giảm hiệu năng khi gặp review sản phẩm vì từ vựng và cách thể hiện cảm xúc khác nhau.

11. **Vì sao tăng sequence length làm chi phí attention tăng?**  
    Trong self-attention thông thường, mỗi token tính điểm với mọi token trong chuỗi. Với $T$ token, số điểm là $T^2$, nên khi độ dài tăng gấp đôi, số điểm tăng gấp bốn.

12. **Confidence cao có đảm bảo prediction đúng không?**  
    Không. Confidence cao cho thấy mô hình gán xác suất cao cho nhãn nó chọn. Mô hình vẫn có thể rất tự tin khi dự đoán sai.

## 29. Reflection

Sau LAB 07, em đã biết xây dựng Transformer, nhưng LAB 08 giúp em hiểu rằng có kiến trúc chưa đồng nghĩa với có mô hình ngôn ngữ tốt. Nếu huấn luyện từ đầu, em cần lượng dữ liệu lớn để mô hình học quy luật ngôn ngữ. Dữ liệu gán nhãn cho bài toán downstream thường ít hơn và tốn công thu thập, nên khó đồng thời đáp ứng việc học ngôn ngữ và học nhiệm vụ.

Vì vậy, em thấy lợi ích của pretrained representation. Qua pretraining trên văn bản lớn, encoder học biểu diễn từ theo ngữ cảnh, quan hệ cú pháp và ngữ nghĩa; cùng một từ có thể được biểu diễn khác nhau tùy câu. Em có thể tái sử dụng các trọng số này làm nền tảng cho nhiệm vụ mới, thay vì yêu cầu mô hình học lại mọi thứ từ dữ liệu gán nhãn.

Với downstream task phân loại cảm xúc IMDb, em chuyển kiến thức ngôn ngữ sang dự đoán review tích cực hoặc tiêu cực. Classification head nhận vector của token [CLS] từ encoder, tạo điểm số cho hai nhãn và học bằng cách giảm sai số so với nhãn thật. Khi fine-tuning cả encoder, gradient còn có thể điều chỉnh biểu diễn theo các tín hiệu cảm xúc của tác vụ. Em hiểu rằng cập nhật encoder giúp đặc trưng phù hợp hơn với nhiệm vụ, trong khi chỉ học head sẽ giữ nguyên biểu diễn ban đầu. Trong thí nghiệm, Frozen đạt Accuracy 76,172%, thấp hơn fine-tuning 11,796 điểm phần trăm; thời gian tương ứng khoảng 182,53 và 504,67 giây.

Trong kết quả đã chạy ở tokenization.ipynb, em thấy sau ba epoch, với cùng kiến trúc và tập huấn luyện, mô hình pretrained đạt test accuracy 87,968% và macro-F1 0.879678, còn mô hình khởi tạo ngẫu nhiên đạt 79,828% và macro-F1 0.796637. Checkpoint được chọn bằng macro-F1 trên validation. Pretrained cao hơn 8,140 điểm phần trăm Accuracy. Quan sát này ủng hộ lợi ích của điểm khởi đầu đã học, nhưng em không xem một lần chạy là bảo đảm cho mọi cấu hình.

Về computational cost, em hiểu pretraining từ đầu đòi hỏi nhiều bước cập nhật, thời gian GPU và năng lượng. Chi phí ban đầu ấy đã được bỏ ra để tạo checkpoint. Khi dùng lại checkpoint, em chỉ trả thêm chi phí fine-tuning, thường nhỏ hơn nhiều so với toàn bộ quá trình pretraining. Điều đó không có nghĩa mỗi epoch fine-tuning luôn rẻ hơn khi hai mô hình có cùng kiến trúc.

Ở thí nghiệm giảm dữ liệu, pretrained chỉ giảm 2,392 điểm phần trăm Accuracy khi chuyển từ 100% xuống 10%, trong khi scratch giảm 29,828 điểm. Điều này cho thấy lợi ích của pretraining khi ít nhãn trong ngân sách đã thử; chưa phải kết luận về hiệu năng tốt nhất của scratch nếu được train lâu hơn. Tăng max length từ 128 lên 256 giúp Accuracy tăng từ 87,968% lên 91,476%, nhưng thời gian tăng từ khoảng 512,59 lên 986,19 giây và bộ nhớ GPU đỉnh tăng từ 2578,47 lên 3895,99 MiB. Em cần cân đối lợi ích giữ thêm nội dung với tài nguyên thực tế.

Domain shift làm thay đổi từ vựng, chủ đề và dấu hiệu đánh giá, nhưng không mặc định làm điểm giảm. Với cùng model IMDb, kết quả Amazon Polarity test gồm 400.000 mẫu đạt Accuracy khoảng 90,0883%, cao hơn IMDb khoảng 2,120 điểm phần trăm. Hai tập có thể khác nhau về độ khó và quy trình gán nhãn nên em chưa thể quy chênh lệch cho khả năng chuyển giao đơn thuần. Model vẫn có lỗi tự tin cao: trên IMDb, mẫu #4 có nhãn negative nhưng được dự đoán positive với confidence 0.987808. Với em, checkpoint là nền tảng hữu ích nhưng chất lượng dự đoán và xác suất vẫn cần được kiểm tra trên từng domain.

 