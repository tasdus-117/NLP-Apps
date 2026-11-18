<a id="lab09-section-31"></a>
## 31. Individual Learning Check


1. **Sequence khác token classification?** Một nhãn cho chuỗi so với một nhãn tại mỗi vị trí; logits lần lượt B×C và B×T×C.
2. **BIO để làm gì?** Biểu diễn bắt đầu/tiếp diễn/ngoài entity cùng loại entity, cho phép khôi phục spans.
3. **B-PER khác I-PER?** B mở người mới, I tiếp diễn người đang có; hai B-PER kề nhau là hai entities.
4. **Vì sao cần alignment?** Gold theo words nhưng tokenizer có thể tách một word thành nhiều subwords và thêm special/pad.
5. **IGNORE trong loss?** Target -100 không đóng góp trực tiếp cross entropy; không đồng nghĩa attention mask=0.
6. **Vì sao không chỉ accuracy?** O thường nhiều; sai một đầu span vẫn có thể đạt nhiều token đúng nhưng entity sai.
7. **Entity F1 khác token F1?** Exact entity yêu cầu cả start/end/type đúng; token F1 xét từng nhãn vị trí.
8. **Vì sao boundary khó?** Tên dài, function words và entities liền nhau cần quyết định ranh giới theo context; một vị trí sai phá exact match.
9. **Contextual representation hữu ích?** Cùng surface có thể là LOC/ORG trong các câu khác; representation phụ thuộc ngữ cảnh hỗ trợ phân biệt.
10. **NER dùng cho IE thế nào?** Decode BIO thành spans, ánh xạ schema và xuất JSON; relation extraction hay suy luận sự kiện cần bước tiếp theo.

<a id="lab09-section-32"></a>
## 32. Reflection

### Reflection (300–500 từ)

Pretrained Transformer có thể hỗ trợ nhiều tác vụ vì backbone tạo biểu diễn theo ngữ cảnh thay vì trực tiếp cố định một loại đầu ra. Tokenizer chuyển văn bản thành input IDs và attention mask. Qua các lớp attention, representation của mỗi vị trí chứa thông tin từ những vị trí khác trong chuỗi. Vì vậy, cùng một tên riêng có thể mang representation khác nhau khi ngữ cảnh thay đổi. Đầu ra backbone có dạng B×T×d, với B là batch size, T là chiều dài token và d là hidden dimension.

Trong sequence classification, prediction head thường nhận representation của [CLS] hoặc một vector pooling từ cả chuỗi. Linear classifier ánh xạ vector d chiều sang các class của câu. Logits có shape B×C và labels thường có shape B. Hàm loss so sánh mỗi dự đoán với một nhãn sequence. [CLS] không tự bảo đảm hiểu toàn câu; cách pooling và quá trình fine-tuning vẫn ảnh hưởng đến chất lượng.

Trong token classification, cùng phép chiếu tuyến tính được áp dụng tại từng vị trí. Logits có shape B×T×C, còn labels có shape B×T. Với NER của bài này, C=9 biểu diễn O cùng B/I của bốn loại entity. Khi một word thành nhiều subwords, chiến lược first-subtoken chỉ giữ target cho mảnh đầu. Special tokens, padding và các mảnh tiếp theo nhận -100 trong labels. Những mảnh tiếp theo vẫn tham gia attention, nên ignore index trong loss khác attention mask.

Thay head chỉ thay cách đọc contextual representations; nó chưa tự giải quyết downstream task. Head mới cần học từ annotation đúng, và backbone thường được fine-tune cùng head. Nhãn BIO, alignment và entity decoding phải phù hợp với cách training. Một prediction đúng vài token vẫn có thể sai entity nếu thiếu ranh giới hoặc nhầm type. Vì thế, đánh giá cần exact span cùng type, không chỉ token accuracy.

Validation dùng để chọn checkpoint trước khi xem test. Baseline giúp lượng hóa phần cải thiện của Transformer, còn phân tích theo type, độ dài và tần suất giúp xác định giới hạn. Cuối cùng, NER có thể chuyển raw text thành JSON gồm người, tổ chức và địa điểm, nhưng chưa tự suy ra quan hệ giữa chúng. Trong thực nghiệm CoNLL-2003, Transformer đạt entity F1 cao hơn baseline. Tuy nhiên, kết quả IE trên 30 câu cùng miền chưa chứng minh khả năng khái quát ngoài miền. Sự khác biệt giữa các loại entity cũng cho thấy prediction head phải được đánh giá cùng dữ liệu và tiêu chí tác vụ.

