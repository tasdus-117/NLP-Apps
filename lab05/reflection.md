### 25. So sánh với LAB 04

#### 1. Bảng so sánh đặc tính giữa các phương pháp

| Đặc tính | TF-IDF + LR | Avg Word2Vec + LR | RNN | LSTM | GRU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Biểu diễn dữ liệu** | Sparse | Dense | Dense | Dense | Dense |
| **Thứ tự từ** | Limited | $\times$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| **Trạng thái phụ thuộc ngữ cảnh** | $\times$ | $\times$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| **Phụ thuộc xa** | $\times$ | $\times$ | Limited | Improved | Improved |
| **Chi phí tính toán** | Low | Low | Higher | Higher | Higher |
| **Biểu diễn từ** | 1 Vector / Word | Vector Embedding | Vector Embedding | Vector Embedding | Vector Embedding |

---

#### 2. Phân tích nguyên lý và hạn chế của mô hình

##### A. RNN / LSTM / GRU giải quyết vấn đề gì mà Average Word2Vec không giải quyết tốt?

1. **Khôi phục và bảo toàn Thứ tự từ:**
   * **Average Word2Vec:** Thực hiện lấy trung bình cộng tất cả các vector từ trong văn bản. Phép cộng có tính chất giao hoán nên làm mất hoàn toàn thứ tự từ. Hai câu có ý nghĩa trái ngược nhau như *"not bad, actually good"* và *"bad, not actually good"* sẽ cho ra biểu diễn vector trùng khớp hoàn toàn.
   * **RNN / LSTM / GRU:** Xử lý dữ liệu tuần tự theo từng bước thời gian. Việc cập nhật trạng thái ẩn đệ quy qua công thức $h_t = f(h_{t-1}, x_t)$ giúp mô hình mã hóa trực tiếp thứ tự xuất hiện của từ vào không gian ngữ cảnh.

2. **Xây dựng Trạng thái phụ thuộc ngữ cảnh:**
   * **Average Word2Vec:** Mỗi từ chỉ có một biểu diễn vector tĩnh cố định, không thay đổi theo các từ xung quanh.
   * **RNN / LSTM / GRU:** Trạng thái ẩn $h_t$ biến đổi liên tục qua từng bước thời gian. Cùng một từ khi đứng ở các ngữ cảnh khác nhau sẽ tạo ra các trạng thái ẩn khác nhau, giúp nhận diện chính xác cấu trúc phủ định hoặc đảo ngữ.

3. **Mã hóa mối quan hệ phụ thuộc theo chuỗi:**
   * Các kiến trúc đệ quy duy trì đường truyền bộ nhớ qua thời gian, cho phép trích xuất các mẫu ngữ pháp và ngữ cảnh kéo dài thay vì nén phẳng toàn bộ văn bản thành một vector tĩnh.

##### B. RNN / LSTM / GRU vẫn còn tồn tại những vấn đề gì?

1. **Chi phí tính toán cao và nút thắt tính toán tuần tự:**
   * Việc tính toán ở bước $t$ bắt buộc phải chờ kết quả từ bước $t-1$. Bản chất phụ thuộc thời gian này làm mô hình không thể tính toán song song hóa hiệu quả trên GPU, dẫn đến thời gian huấn luyện tăng mạnh so with các phương pháp truyền thống.

2. **Hạn chế về khoảng cách phụ thuộc xa và nút thắt thông tin:**
   * **Simple RNN:** Bị ảnh hưởng nghiêm trọng bởi hiện tượng tiêu biến gradient khi lan truyền ngược qua chuỗi dài, làm mất hoàn toàn thông tin ở đầu câu.
   * **LSTM / GRU:** Mặc dù đã khắc phục hiện tượng tiêu biến gradient nhờ cơ chế cổng và trạng thái ô nhớ, mô hình vẫn gặp hiện tượng nút thắt thông tin khi bắt buộc phải nén toàn bộ nội dung của một chuỗi dài vào một vector trạng thái ẩn có kích thước cố định.

3. **Rủi ro mất mát thông tin do cắt ngắn chuỗi và nhiễu từ token độn:**
   * Việc cố định độ dài đầu vào làm mất các thông tin quan trọng nằm ngoài ngưỡng cắt, trong khi việc chèn quá nhiều token độn đối với câu ngắn sẽ làm loãng tín hiệu gradient.

4. **Hạn chế đối với ngữ cảnh mỉa mai và cấu trúc phi tuần tự:**
   * Do cơ chế đọc duyệt theo một chiều, mô hình dễ bị thiên vị bởi các từ xuất hiện gần cuối hoặc các từ có trọng số từ vựng mạnh ở đầu câu, dẫn đến dự đoán sai đối with các câu mang ý nghĩa mỉa mai hoặc câu có cấu trúc phức tạp.

### 27. AI policy — Quy định và Báo cáo minh bạch sử dụng AI
---

#### 2. Tuyên bố mức độ hỗ trợ từ AI

---

**AI Assistance Statement 1 — Xuất Bảng và Đồ thị Thực nghiệm 4 & 5**
* **Tool:** Gemini / ChatGPT
* **Purpose:** Tự động hóa viết mã nguồn Python để thu thập thông số, tính số lượng tham số, đo thời gian huấn luyện và vẽ các biểu đồ trực quan hóa cho Bài tập 19 và 20.
* **Generated:** Mã nguồn Python sử dụng `matplotlib` và `pandas` để tính toán chỉ số, lưu kết quả ra file CSV và xuất các file ảnh.
* **Modified:** Chỉnh sửa định dạng bảng, điều chỉnh độ phân giải, căn chỉnh màu sắc các đường biểu đồ để hiển thị rõ ràng trên báo cáo.
* **Verified:** Đã kiểm tra tính chính xác của các con số thống kê bằng cách đối chiếu trực tiếp với kết quả chạy trên môi trường thực thi PyTorch.

---

**AI Assistance Statement 2 — Mã nguồn Tạo Bảng Phân tích Lỗi và Đánh giá Toàn diện**
* **Tool:** Gemini / ChatGPT
* **Purpose:** Tạo mã nguồn thiết lập khung đánh giá đa tiêu chí và trích xuất bảng phân tích lỗi cho Bài tập 22 và 24.
* **Generated:** Khung hàm Python lọc các trường hợp dự đoán đúng/sai, tính toán các độ đo và vẽ bảng tổng hợp xuất ra file ảnh.
* **Modified:** Tùy chỉnh danh sách các câu kiểm thử thực tế, bổ sung các nhãn phân tích nguyên nhân lỗi do cắt ngắn chuỗi, mỉa mai và phủ định.
* **Verified:** Đã kiểm tra lại từng dòng mã nguồn, đảm bảo công thức tính các độ đo phân loại hoàn toàn chính xác theo đúng định nghĩa toán học.

---

**AI Assistance Statement 3 — Khung Ứng dụng Phân loại Cảm xúc**
* **Tool:** Gemini / ChatGPT
* **Purpose:** Xây dựng khung mã nguồn quy trình hoàn chỉnh từ tiền xử lý đầu vào văn bản thô đến đưa ra dự đoán xác suất cho Bài tập 26.
* **Generated:** Đoạn mã Python định nghĩa hàm dự đoán tích hợp bộ mã hóa câu và suy luận mô hình.
* **Modified:** Thêm các trường hợp kiểm thử đầu vào đa dạng (review ngắn, review dài, review chứa từ phủ định và review chứa cấu trúc tương phản).
* **Verified:** Chạy thử nghiệm trực tiếp trên môi trường mã nguồn Python và tự thực hiện phân tích chuyên sâu các đoạn ngữ cảnh ảnh hưởng tới kết quả dự đoán mà không phụ thuộc vào văn bản sinh tự động.