### 24. Error analysis

#### 1. Error Analysis Table

| Text | True | Predicted | Analysis |
| :--- | :---: | :---: | :--- |
| **1.** *"This movie was absolutely wonderful with great acting and a brilliant storyline."* | **positive** | **positive** | **Phân loại ĐÚNG:** Mẫu câu ngắn, chứa nhiều từ khóa cảm xúc tích cực mạnh (`wonderful`, `great`, `brilliant`) xuất hiện liên tiếp, không chứa từ phủ định hay nhiễu. |
| **2.** *"Worst film I have ever seen. Waste of money and time. Unbelievably boring."* | **negative** | **negative** | **Phân loại ĐÚNG:** Tập hợp các unigram/bi-gram tiêu cực rõ ràng (`worst`, `waste of money`, `boring`), tín hiệu cảm xúc đồng nhất toàn bài. |
| **3.** *"I expected the movie to be bad, but it was actually wonderful and charming."* | **positive** | **positive** | **Phân loại ĐÚNG:** Mô hình bắt đúng cấu trúc tương phản qua liên từ `but` và ưu tiên tín hiệu tích cực phía sau thay vì bị đánh lừa bởi từ `bad` ở đầu câu. |
| **4.** *"I expected the movie to be terrible, ... [chuỗi dài 150 words] ... but it turned out to be an absolute masterpiece."* | **positive** | **negative** | **Phân loại SAI - Lỗi Truncation & Sequence Length:** Đoạn đánh giá chốt hạ (`masterpiece`) bị cắt mất do `max_length = 100`. Mô hình chỉ đọc được đoạn đầu chứa từ `terrible`. |
| **5.** *"Oh, brilliant! Another cliché horror film with terrible actors and predictable jumpscares."* | **negative** | **positive** | **Phân loại SAI - Lỗi Sarcasm / Mỉa mai:** Ngôn ngữ mỉa mai sử dụng từ khen ngợi (`brilliant!`) ở đầu câu. Mô hình bị thiên vị bởi trọng số tích cực của từ này mà bỏ qua ngữ cảnh tiêu cực phía sau. |
| **6.** *"The direction was not bad at all, though the screenplay lacked depth in several parts."* | **positive** | **negative** | **Phân loại SAI - Lỗi Negation & Ambiguous Sentiment:** Không xử lý tốt cụm phủ định kép (`not bad at all` = tích cực) kết hợp với nhận xét pha trộn (`lacked depth`), làm hidden state bị kéo về nhãn tiêu cực. |

---

#### 2. Chẩn đoán chi tiết nguyên nhân lỗi

##### A. Lỗi do Cắt ngắn chuỗi (Truncation & Sequence Length) — *Mẫu số 4*
* **Hiện tượng:** Cấu trúc câu dạng *"I expected [Negative], but [Positive]"* nhưng phần khen ngợi nằm ở cuối văn bản dài hơn 100 tokens.
* **Xem xét các yếu tố:**
  * **Sequence Length & Truncation:** Do cấu hình `max_length = 100`, phần văn bản quan trọng chứa từ khóa quyết định (`masterpiece`) bị cắt bỏ hoàn toàn ở bước tiền xử lý.
  * **Xác định bản chất:** **KHÔNG PHẢI lỗi do Sequence Modeling.** Mô hình không nhận được dữ liệu đầu vào ở giai đoạn suy luận do thiết lập siêu tham số tiền xử lý bị giới hạn.
* **Giải pháp:** Tăng `max_length` lên 200–256 tokens hoặc áp dụng chiến lược truncation từ 2 phía lấy 50 tokens đầu + 50 tokens cuối.

##### B. Lỗi do Ngôn ngữ Mỉa mai (Sarcasm / Irony) — *Mẫu số 5*
* **Hiện tượng:** Văn bản sử dụng từ biểu cảm tích cực (`brilliant!`) với ý nghĩa mỉa mai để chê bai bộ phim.
* **Xem xét các yếu tố:**
  * **Vocabulary & Weight Bias:** Từ `brilliant` có trọng số tích cực rất cao trong không gian Embedding được học từ tập Train.
  * **Contrast & Sequence Context:** Tín hiệu tiêu cực xuất hiện sau đó (`cliché`, `terrible`, `predictable`) bị lấn át bởi từ khóa tích cực đứng ngay vị trí đầu chuỗi.
  * **Xác định bản chất:** **Lỗi do Giới hạn Biểu diễn Từ vựng & Biểu diễn Ngữ nghĩa sâu.** Kiến trúc RNN/GRU thuần thiếu cơ chế chú ý để liên kết sự mâu thuẫn giữa từ `brilliant` và các thuộc tính tiêu cực theo sau trong ngữ cảnh mỉa mai.

##### C. Lỗi do Phủ định & Cảm xúc Mơ hồ (Negation & Ambiguous Sentiment) — *Mẫu số 6*
* **Hiện tượng:** Câu kết hợp giữa cụm phủ định để khen (`not bad at all`) và một lời chê nhẹ (`lacked depth`).
* **Xem xét các yếu tố:**
  * **Negation:** Mô hình gặp khó khăn khi xử lý cụm `not bad`. Nếu chỉ nhận diện từ `bad` mà không truyền qua cổng điều khiển kết hợp với `not`, mô hình sẽ gán trọng số tiêu cực.
  * **Ambiguous / Mixed Sentiment:** Văn bản chứa cả ý khen lẫn ý chê khiến tổng trạng thái ẩn $h_T$ bị kéo về gần ranh giới quyết định ($0.5$).
  * **Xác định bản chất:** **LỖI DO SEQUENCE MODELING & CONVERGENCE.** Mô hình Recurrent chưa học đủ sâu các quan hệ phụ thuộc ngắn giữa từ phủ định và từ đi kèm trong quá trình lan truyền đệ quy, đồng thời kích thước hidden state nhỏ ($HIDDEN\_DIM = 64$) chưa đủ dung lượng để tách biệt các vế câu mang cảm xúc pha trộn.