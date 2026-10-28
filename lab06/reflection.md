# 28. So sánh LAB 05 và LAB 06

## 1. Bảng so sánh đặc tính giữa hai phương pháp

| Đặc tính | RNN / LSTM / GRU | Attention Mechanism |
| :--- | :---: | :---: |
| **Thứ tự chuỗi** | $\checkmark$ | $\checkmark$ |
| **Trạng thái ẩn đệ quy** | $\checkmark$ | Không bắt buộc |
| **Tính toán tuần tự** | $\checkmark$ | Không bắt buộc |
| **Tương tác trực tiếp giữa các token** | Limited | $\checkmark$ |
| **Tương tác khoảng cách xa** | Difficult | Easier |
| **Tính toán song song** | Limited | $\checkmark$ |
| **Chi phí theo độ dài chuỗi** | Gần tuyến tính theo bước thời gian $\mathcal{O}(T)$ | $\mathcal{O}(T^2)$ |
| **Biểu diễn ngữ cảnh** | Recurrent state ($h_T$) | Weighted combination |

---

## 2. Phân tích nguyên lý và Trả lời câu hỏi chuyên sâu

### A. Attention giải quyết được những vấn đề gì của kiến trúc đệ quy tuần tự?

1. **Khắc phục nút thắt tính toán tuần tự và mở ra khả năng song song hóa:**
   * **Mô hình đệ quy:** Việc tính toán tại bước thời gian $t$ bắt buộc phải chờ kết quả bước $t-1$. Bản chất đệ quy tuần tự làm cho mô hình không thể tính toán song song hóa trên phần xử lý đồ họa, khiến thời gian huấn luyện bị kéo dài.
   * **Cơ chế chú ý:** Tính toán ma trận chú ý giữa toàn bộ các cặp từ cùng một lúc thông qua các phép nhân ma trận song song. Điều này loại bỏ hoàn toàn nút thắt phụ thuộc thời gian, giúp tăng tốc độ huấn luyện lên nhiều lần.

2. **Giải quyết vấn đề phụ thuộc khoảng cách xa và tiêu biến gradient:**
   * **Mô hình đệ quy:** Dù có các cổng điều khiển và trạng thái ô nhớ, tín hiệu ngữ cảnh vẫn phải truyền qua các bước trung gian. Khi độ dài chuỗi lớn, thông tin ở các bước đầu tiên vẫn bị nhạt dần hoặc biến dạng.
   * **Cơ chế chú ý:** Tạo ra kết nối trực tiếp giữa hai từ bất kỳ trong chuỗi với khoảng cách đường truyền cố định. Nhờ đó, thông tin ngữ cảnh giữa các từ xa nhau được trao đổi trực tiếp mà không bị suy hao qua các bước trung gian.

3. **Loại bỏ hiện tượng nút thắt thông tin:**
   * **Mô hình đệ quy:** Bắt buộc phải nén toàn bộ thông tin của một văn bản dài vào một vector trạng thái ẩn cố định, dẫn đến việc thất thoát thông tin quan trọng.
   * **Cơ chế chú ý:** Tạo ra biểu diễn ngữ cảnh linh hoạt bằng cách tính tổng có trọng số của toàn bộ các vector giá trị, giữ lại đầy đủ thông tin chi tiết của từng vị trí trong câu.

---

### B. Attention tạo ra những vấn đề mới nào?

1. **Bùng nổ chi phí tính toán và bộ nhớ theo cấp số nhân bậc hai $\mathcal{O}(T^2)$:**
   * Việc tính toán ma trận tương quan giữa mọi cặp từ đòi hỏi bộ nhớ và số phép tính tăng theo cấp số nhân bậc hai. Khi độ dài chuỗi tăng gấp 10 lần, số phần tử ma trận chú ý cần lưu trữ tăng lên 100 lần, dẫn đến sự cố tràn bộ nhớ rất nhanh khi xử lý các văn bản dài.

2. **Thất thoát thông tin thứ tự tự nhiên nếu không có mã hóa vị trí:**
   * Do phép nhân ma trận không phụ thuộc vào thứ tự xuất hiện của các từ, cơ chế chú ý thuần túy có tính chất bất biến với phép hoán vị. Nếu không chèn thêm mã hóa vị trí, mô hình sẽ xem các câu chứa cùng tập hợp từ nhưng khác thứ tự là hoàn toàn giống nhau.

3. **Hiện tượng hội tụ lệch trọng số chú ý và khó giải thích hoàn toàn:**
   * Trọng số chú ý lớn không đồng nghĩa hoàn toàn với mức độ đóng góp thực sự của từ đó vào quyết định phân loại. Việc phân tích trọng số chú ý chỉ mang tính chất đưa ra giả thuyết chỉ dấu trực quan chứ không phải là lời giải thích nguyên nhân hoàn chỉnh cho mô hình.

# 30. AI policy — Quy định và Báo cáo minh bạch sử dụng AI (LAB 06)

## 1. Nguyên tắc tuân thủ chính sách AI

Báo cáo này được thực hiện tuân thủ nghiêm ngặt các quy định về việc sử dụng công cụ Trí tuệ nhân tạo:

* **Các mục tiêu ĐƯỢC PHÉP sử dụng AI:**
  * Hỗ trợ tìm và sửa lỗi cú pháp trong mã nguồn.
  * Tự động hóa tạo tài liệu và nhận xét mã nguồn.
  * Xuất các đồ thị trực quan hóa ma trận chú ý và biểu đồ dữ liệu.
  * Tra cứu và giải thích chi tiết các hàm API của thư viện NumPy và PyTorch.
  * Vẽ đồ thị biểu diễn sự tăng trưởng độ phức tạp tính toán.
  * Trích xuất và giải thích nguyên nhân các lỗi thời gian chạy.
  * Xây dựng khung mã nguồn mẫu cho quy trình huấn luyện và suy luận.

* **Các mục tiêu TUYỆT ĐỐI KHÔNG sử dụng AI thay thế suy luận cá nhân:**
  * Thực hiện các bước tính toán số học giá trị nhân ma trận và chuẩn hóa Softmax.
  * Đưa ra các dự đoán ban đầu trước khi chạy thực nghiệm.
  * Giải thích ý nghĩa hình học và vai trò của các ma trận Query, Key, Value.
  * Tính toán trực tiếp giá trị ma trận chú ý bằng thủ công hay tự suy luận số liệu.
  * Biện giải và phân tích ý nghĩa hình ảnh trực quan hóa ma trận chú ý.
  * Xác định nguyên nhân gốc rễ và phân loại các lỗi phân loại cảm xúc.
  * Tổng kết bài học kinh nghiệm và tự đánh giá năng lực cá nhân.

---

## 2. Tuyên bố mức độ hỗ trợ từ AI


---

**AI Assistance Statement 1 — Xuất Mã nguồn Tính toán Độ phức tạp và Vẽ Đồ thị**
* **Tool:** Gemini / ChatGPT
* **Purpose:** Tự động hóa viết mã nguồn Python để tính toán số phần tử ma trận chú ý theo độ dài chuỗi và vẽ đồ thị trực quan hóa sự tăng trưởng độ phức tạp cho Bài tập 25 và 29.
* **Generated:** Mã nguồn Python sử dụng `matplotlib` và `pandas` để tính số phần tử $T^2$, lưu kết quả ra file CSV và xuất file ảnh đồ thị.
* **Modified:** Điều chỉnh nhãn trục, định dạng hiển thị con số hàng triệu, điều chỉnh màu sắc và độ phân giải của đồ thị để hiển thị rõ ràng trên báo cáo.
* **Verified:** Đã kiểm tra tính chính xác của các con số $1000^2 = 1,000,000$ và $10000^2 = 100,000,000$ bằng tính toán số học độc lập.

---

**AI Assistance Statement 2 — Mã nguồn Mô hình Self-Attention và Vẽ Đồ thị Nhiệt**
* **Tool:** Gemini / ChatGPT
* **Purpose:** Xây dựng khung mã nguồn định nghĩa kiến trúc phân loại cảm xúc sử dụng Self-Attention và xuất đồ thị nhiệt biểu diễn ma trận chú ý cho Bài tập 26.
* **Generated:** Đoạn mã PyTorch định nghĩa các lớp chiếu Query, Key, Value, công thức toán học Scaled Dot-Product Attention và hàm vẽ ma trận chú ý sử dụng `seaborn.heatmap`.
* **Modified:** Tùy chỉnh danh sách các câu kiểm thử thực tế, căn chỉnh nhãn trục tọa độ hiển thị đúng các từ trong câu đầu vào.
* **Verified:** Chạy thử nghiệm trực tiếp trên môi trường mã nguồn Python, kiểm tra ma trận đầu ra có kích thước $T \times T$ và tổng các hàng qua hàm Softmax bằng đúng $1.0$.

---

**AI Assistance Statement 3 — Mã nguồn Bảng Phân tích Lỗi và Đánh giá Mô hình Chú ý**
* **Tool:** Gemini / ChatGPT
* **Purpose:** Tạo mã nguồn thiết lập bảng phân tích lỗi đánh giá trên các mẫu câu chứa ngôn ngữ mỉa mai, phủ định và câu dài cho Bài tập 27.
* **Generated:** Khung hàm Python lọc các trường hợp dự đoán đúng/sai và hỗ trợ xuất bảng phân tích nguyên nhân ra file ảnh.
* **Modified:** Nhập trực tiếp các phân loại lỗi và tự viết toàn bộ phần phân tích giả thuyết lệch trọng số chú ý mà không sử dụng văn bản sinh tự động.
* **Verified:** Đã đối chiếu toàn bộ các mẫu câu với kết quả dự đoán thực tế từ mô hình để đảm bảo thông tin trong bảng hoàn toàn chính xác.