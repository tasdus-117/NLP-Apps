# 27. Error analysis (Phân tích lỗi trên Attention-based Model)

## 1. Bảng phân tích mẫu dữ liệu

| Text | True | Predicted | Classification & Root Cause Analysis |
| :--- | :---: | :---: | :--- |
| **1.** *"This movie was absolutely wonderful with great acting and a brilliant storyline."* | **positive** | **positive** | **Phân loại ĐÚNG:** Mẫu câu ngắn, chứa nhiều từ biểu cảm tích cực mạnh. Trọng số Attention phân bổ tập trung vào các từ `wonderful`, `great`, `brilliant`. |
| **2.** *"Worst film I have ever seen. Waste of money and time. Unbelievably boring."* | **negative** | **negative** | **Phân loại ĐÚNG:** Tập hợp các từ tiêu cực nhất quán. Attention tập trung cao vào `worst`, `waste`, `boring`. |
| **3.** *"I expected the movie to be bad, but it was actually wonderful and charming."* | **positive** | **positive** | **Phân loại ĐÚNG:** Mô hình bắt đúng cấu trúc tương phản qua liên từ `but`. Attention dồn trọng số vào các từ chốt hạ phía sau (`wonderful`, `charming`). |
| **4.** *"I expected the movie to be terrible, ... [chuỗi dài 150 words] ... but it turned out to be an absolute masterpiece."* | **positive** | **negative** | **Phân loại SAI:** Do cấu hình `max_length = 100`, vế khen ngợi ở cuối bị cắt mất. Attention chỉ hoạt động trên 100 từ đầu tiên chứa từ `terrible`. |
| **5.** *"Oh, brilliant! Another cliché horror film with terrible actors and predictable jumpscares."* | **negative** | **positive** | **Phân loại SAI:** Ngôn ngữ mỉa mai. Attention tập trung trọng số lớn vào từ `brilliant` ở đầu câu thay vì các từ tiêu cực phía sau. |
| **6.** *"The direction was not bad at all, though the screenplay lacked depth in several parts."* | **positive** | **negative** | **Phân loại SAI:** Attention không tập trung đúng vào cụm liên kết `not` $\rightarrow$ `bad` mà bị phân tán sang vế chê `lacked depth`. |

---

## 2. Phân tích chuyên sâu Attention Behavior & Kiểm chứng Giả thuyết

Đối với các mô hình dựa trên cơ chế Attention, việc phân tích hành vi của ma trận trọng số chú ý cần được thực hiện thông qua việc **đối chiếu giữa dự đoán cảm tính của con người và hành vi thực tế của mô hình**:

### Case 1: Lỗi do Ngôn ngữ Mỉa mai — *Mẫu số 5*
- **Dự đoán của sinh viên:** Con người kỳ vọng mô hình sẽ nhận diện từ `brilliant` mang ý nghĩa mỉa mai khi đi kèm với các thuộc tính chê bai như `cliché`, `terrible`, `predictable`, từ đó Attention sẽ hạ trọng số của `brilliant` và nâng trọng số của các từ mô tả bản chất tiêu cực.
- **Quan sát Attention thực tế:** Ma trận Attention phân bổ trọng số cực kỳ cao nổi bật trên đồ thị nhiệt vào ngay từ `brilliant` đứng ở vị trí đầu câu.
- **Đặt Giả thuyết:** 
  * *Giả thuyết:* Do từ `brilliant` có biểu diễn vector tích cực rất mạnh trong không gian từ vựng và đứng ở vị trí mở đầu, cơ chế Self-Attention đã hội tụ lệch, biến `brilliant` thành hạt nhân ngữ cảnh chính và làm lu mờ tín hiệu tiêu cực của các từ theo sau.
- **Kiểm chứng Giả thuyết:** Đối chiếu với đầu ra lớp phân loại, xác suất tích cực đạt mức cao ($> 0.80$). Giả thuyết này giải thích hợp lý cho sai số: Attention đã tập trung vào một từ mang trọng số cảm xúc ngược lại với nhãn tổng thể của toàn câu, dẫn đến việc xây dựng Vector Ngữ cảnh bị sai lệch hướng trước khi đưa vào bộ phân loại.

### Case 2: Lỗi do Phủ định & Cảm xúc Mơ hồ — *Mẫu số 6*
- **Dự đoán của sinh viên:** Con người kỳ vọng ma trận Attention sẽ tạo ra một liên kết tương quan mạnh giữa từ phủ định `not` và từ tiêu cực `bad` (cụm `not bad`), giúp mô hình hiểu đây là một lời khen nhẹ.
- **Quan sát Attention thực tế:** Trọng số chú ý giữa `not` và `bad` rất thấp. Thay vào đó, Attention phân bổ đồng đều và nhỉnh hơn ở vế sau (`screenplay lacked depth`).
- **Đặt Giả thuyết:** 
  * *Giả thuyết:* Mô hình Self-Attention chưa qua huấn luyện sâu hoặc chưa có cơ chế vị trí đủ mạnh sẽ không tự động liên kết các từ phủ định ngắn. Việc Attention bỏ qua mối liên kết `not` $\rightarrow$ `bad` khiến biểu diễn vế đầu bị kéo về trung tính, trong khi vế chê `lacked depth` chiếm ưu thế.
- **Kiểm chứng Giả thuyết:** Dự đoán cuối cùng bị lệch sang `Negative`. Điều này củng cố giả thuyết rằng việc Attention **không tập trung vào mối quan hệ phủ định mà con người dự đoán là quan trọng** đã trực tiếp làm sai lệch thông tin ngữ cảnh tổng hợp.

---

## 3. Bài học phương pháp luận

> **Không được kết luận ngắn gọn hoặc cảm tính rằng: *"Attention nhìn sai từ nên model sai"*.**
>
> Trong báo cáo khoa học, trọng số Attention chỉ là một chỉ dấu trực quan phản ánh cách mô hình biến đổi dữ liệu. Sinh viên phải luôn trình bày dưới dạng **Giả thuyết** về sự lệch pha giữa Attention thực tế và kỳ vọng lý thuyết, sau đó kiểm chứng giả thuyết đó bằng cách đối chiếu với công thức tính toán và kết quả đầu ra của mô hình.