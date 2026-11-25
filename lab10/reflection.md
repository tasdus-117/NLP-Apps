# LAB 10 — Learning Check và Reflection

## 28. Phân biệt QA và RAG

Extractive QA nhận question và context đã cung cấp rồi trích một span nguyên văn. RAG bổ sung retrieval để tìm tài liệu liên quan từ một kho trước khi generator hoặc QA model dùng các tài liệu đó. [document_qa.py](document_qa.py) nhận context/document trực tiếp; không có retriever, không tìm trên web và không sinh câu trả lời mới ngoài context. Một span mà model chọn được không tự chứng minh document thật sự chứa câu trả lời đúng.

## 29. AI Usage

- **AI-assisted code:** pipeline, alignment, metric, visualization và ứng dụng được khai báo trong README/notebook.
- **What was generated:** toàn bộ phần code
- **What was modified:** sửa lại form code, code theo yêu cầu.
- **What was verified:** lý thuyết/các phép tính có thể đối chiếu với code/tests; dữ kiện thực nghiệm của reflection sẽ dẫn đúng output full run. 

## 33. Individual Learning Check

1. Extractive QA khác classification như thế nào?

   Classification dự đoán một nhãn cho toàn input, thường từ vector `[CLS]` hoặc pooled representation. Extractive QA nhận cả question và context, dự đoán start/end của một đoạn nằm trong context, rồi cắt đoạn nguyên văn từ offsets. Output QA có vị trí và độ dài biến thiên, không phải một nhãn lớp cố định.

2. Tại sao QA cần start và end position?

   Start xác định đáp án bắt đầu ở đâu nhưng không xác định độ dài. Hai span có thể cùng start nhưng khác end, chẳng hạn `New York` và `New York City`. Hai phân phối cho phép học hai biên; decoder phải xét cặp hợp lệ `start≤end` và giới hạn độ dài thay vì ghép hai argmax tùy ý.

3. Character offset khác token index như thế nào?

   Character offset đếm ký tự trong chuỗi gốc; token index đếm token sau tokenizer, bao gồm question và special tokens nếu đó là pair input. Một từ có thể tách thành nhiều subword. Với `Paris` trong ví dụ đề, char span là `[25,30)`, còn token index phụ thuộc cả tokenizer và question; không dùng số 25 làm nhãn token.

4. Tại sao cần offset mapping?

   Dataset cung cấp answer_start và answer text theo ký tự, còn mô hình phân loại vị trí token. Offset mapping nối mỗi token với khoảng ký tự `[start,end)` để tạo nhãn huấn luyện và cắt prediction về text gốc. Phải lọc `sequence_ids==1`, vì offset question cũng bắt đầu từ 0 và không quy chiếu vào context.

5. `stride` dùng để làm gì?

   Khi dùng overflow, stride là số token context chồng lấp giữa các window liên tiếp. Nó giữ các span gần biên có cơ hội nằm trọn trong một window. Stride phải nhỏ hơn context capacity thực sau question/special tokens; cùng stride64 dùng cho ba length trong bài để tránh thêm biến thay đổi.

6. Vì sao context dài có thể phải chia thành nhiều window?

   Transformer có giới hạn số vị trí; pair còn dành chỗ cho question và special tokens. Cắt một window có thể bỏ đáp án ở cuối. Sliding windows bao phủ context, giữ mapping từ từng feature về sample gốc và cho decoder so các span hợp lệ trên tất cả window. Nhiều window làm tăng số feature và chi phí ngay cả khi số câu hỏi không đổi.

7. Exact Match đo điều gì?

   EM đo prediction có bằng hoàn toàn một gold answer sau normalization hay không. Bài dùng lowercase, bỏ ASCII punctuation và article tiếng Anh, rồi chuẩn hóa whitespace theo SQuAD. Với nhiều reference, lấy điểm tốt nhất; EM trung bình tính trên câu hỏi gốc, không trên window.

8. F1 của QA khác gì với accuracy?

   QA F1 đo overlap token giữa prediction và gold bằng precision/recall, tính theo số lần xuất hiện của token bằng Counter. Nó cho điểm từng phần khi prediction thừa hoặc thiếu từ. Accuracy thường đếm tỷ lệ quyết định/nhãn đúng; F1 này không đếm tỷ lệ token context được gán nhãn đúng và không dùng subword model làm đơn vị metric.

9. Vì sao prediction có thể có $F1 > 0$ nhưng $EM = 0$?

   Vì prediction có chung một số token nhưng chưa bằng toàn bộ gold. Gold `Paris France`, prediction `France` có precision1, recall0.5, F1=2/3 nhưng EM0. Boundary sai nhẹ cũng có thể cho hiện tượng này; phải đọc prediction thật để giải thích từng ví dụ, không suy nguyên nhân chỉ từ hai điểm số.

10. Wrong-start và wrong-end khác nhau như thế nào?

    Wrong-start là sai biên trái, wrong-end là sai biên phải. `York City` so với `New York City` sai start nhưng đúng end; `New York` đúng start nhưng sai end. Báo cáo dùng một gold pair chung được chọn theo quy tắc đã công bố, không ghép hai endpoint từ hai reference khác nhau. Wrong-span cả hai biên sai được tách với mọi span_mismatch.

11. Extractive QA khác RAG ở đâu?

    Extractive QA giả định đã biết context và tìm answer span trong đó. RAG có thêm retrieval từ tập tài liệu và thường có generation tổng hợp câu trả lời. Retrieval sai có thể làm mất bằng chứng dù QA head tốt; LAB10 không giải quyết việc chọn tài liệu từ hàng nghìn document.

12. Khi nào extractive QA không thể trả lời câu hỏi?

    Khi context không chứa đáp án, thông tin không đủ, câu hỏi mơ hồ hoặc cần đáp án không thể biểu diễn bằng một đoạn liên tục. Giới hạn window/decoder hoặc chất lượng model cũng có thể khiến không trích được đáp án. Core SQuAD1.1 luôn chọn span nên có thể trả một đoạn sai; no-answer cần mô hình/ngưỡng được huấn luyện và đánh giá riêng, không diễn giải raw logits thành xác suất có đáp án.

## 34. Reflection

```text
                  Pretrained Transformer
                /           |            \
       [CLS] representation  Token reps   Token reps
                ↓                ↓            ↓
         Classification         NER      Start / End
                ↓                ↓            ↓
             LAB 08           LAB 09       LAB 10
```

<!-- REFLECTION_START -->
LAB 08, LAB 09 và LAB 10 dùng pretrained Transformer để tạo biểu diễn phụ thuộc ngữ cảnh, nhưng backbone không tự quyết định nhiệm vụ. LAB 08 dùng vector toàn chuỗi, thường là [CLS], qua classification head để dự đoán một nhãn văn bản. Target là lớp của sample và loss thường là cross-entropy trên các lớp.

LAB 09 dùng biểu diễn từng token, áp dụng classifier dùng chung để dự đoán nhãn NER. Target có một nhãn tại mỗi vị trí hợp lệ; padding và các vị trí không được gán nhãn phải được loại khỏi loss. LAB 10 cũng dùng token representations nhưng có hai bộ logits start/end. Target là hai vị trí giới hạn đoạn trả lời, và loss là trung bình hai cross-entropy theo chiều vị trí. Hai argmax riêng không bảo đảm span hợp lệ, nên decoder chỉ xét context tokens, start không vượt end và tối đa 30 subword.

Alignment cho thấy vì sao đổi prediction head kéo theo đổi cách xử lý dữ liệu. Trong ví dụ Paris đã chạy, character span là [25,30), còn pair-token index là 14: cùng một đáp án có hai hệ tọa độ. End token inclusive khác end character exclusive. Sliding windows giữ đáp án ở context dài; window thiếu toàn bộ gold dùng CLS khi huấn luyện, nhưng cơ chế này không phải bộ phát hiện câu hỏi không có đáp án.

Lượt đầy đủ dùng 79.214 mẫu train, 8.385 validation và 10.570 heldout từ public dev SQuAD; dev này không phải official hidden test. Cấu hình 384, epoch 2 được chọn bằng validation F1 79,96% trước đánh giá heldout. Heldout đạt EM 76,63% và F1 84,72%, trong khi baseline câu có overlap đạt 0,22% và 18,88%. Khoảng cách EM/F1 phân biệt đúng một phần với đúng hoàn toàn. Có 2.245 wrong-start, 1.924 wrong-end và 1.302 trường hợp sai cả hai biên; các số này không tự giải thích nguyên nhân.

F1 của ba độ dài 128/256/384 là 83,15/84,10/84,72%. Xu hướng một seed không chứng minh context dài luôn tốt hơn; số window và optimizer steps cũng thay đổi dù epochs giữ nguyên.

Trên cùng 10.438 mẫu đủ điều kiện, thêm nhiễu làm F1 giảm 0,93 điểm phần trăm; đảo câu giảm 1,96 điểm. Kết quả ủng hộ prediction về tính không bất biến, nhưng không chứng minh mọi thay đổi là distractor. Có 132 mẫu bị loại vì gold vượt biên cắt câu, nên kết luận paired có phạm vi rõ ràng.

Extractive QA vẫn cần context chứa bằng chứng và đáp án liên tục. LAB 11 bổ sung retrieval để tìm context; RAG kết hợp truy xuất với sinh. Đây là đối chiếu lý thuyết và dữ kiện LAB 10, không nhận đã chạy lại LAB 08/09 hoặc kể trải nghiệm sinh viên.
<!-- REFLECTION_END -->

Nguồn đối chiếu: [results.csv](results.csv), `.runtime/full_run/tokenizer_alignment.json`, `.runtime/full_run/error_counts.json`, `.runtime/full_run/perturbation_results.json`; run ID `8faacbbd16e7d738dcf71bbdd8f51d12ffe7b805c1b43f8ea62df17775b0772c`. Nhóm wrong-span là cả hai biên sai theo gold pair được chọn chung; không đồng nhất với mọi span mismatch. Chi tiết case và giả thuyết nguyên nhân ở [error_analysis.md](error_analysis.md).

