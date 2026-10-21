### Prediction 1

* **Prediction:** RNN + Embedding
* **Reason:** 
  * **TF-IDF + Logistic Regression:** TF-IDF hoạt động theo cơ chế Bag-of-Words (BoW). Nó đếm tần suất xuất hiện của từ/n-gram và nhân với nghịch đảo tần suất văn bản, hoàn toàn bỏ qua thứ tự xuất hiện của các từ trong câu trừ khi dùng n-gram kích thước cố định, nhưng vẫn chỉ bắt được phụ thuộc rất ngắn cục bộ.
  * **RNN + Embedding:** RNN xử lý dữ liệu tuần tự theo từng timestep ($t = 1, 2, \dots, T$) thông qua công thức đệ quy $h_t = f(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$. Trạng thái $h_t$ liên tục tích lũy và phụ thuộc trực tiếp vào các trạng thái $h_{t-1}$ trước đó, giúp mô hình ghi nhận trực tiếp thứ tự xuất hiện của từng từ trong chuỗi.
* **Confidence:** High 

---

### Prediction 2

* **Prediction:** LSTM
* **Reason:**
  * **Hạn chế của RNN:** RNN gặp hiện tượng **Vanishing Gradient** khi BPTT qua chuỗi dài, do đạo hàm liên tục nhân qua các ma trận trọng số $W_{hh}$ và hàm kích hoạt $\tanh$ (có đạo hàm $\le 1$). Tín hiệu lỗi bị triệt tiêu về 0, làm mô hình không thể học được phụ thuộc xa.
  * **Cơ chế vượt trội của LSTM:** LSTM giới thiệu **Cell State ($c_t$)** đóng vai trò như một đường cao tốc thông tin. Cập nhật $c_t$ chủ yếu dựa trên **phép cộng**:
    $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
    Phép cộng giúp gradient truyền ngược qua thời gian mà không bị nhân liên tiếp với ma trận trọng số, giảm thiểu tối đa hiện tượng tiêu biến gradient. Ngoài ra, LSTM có các cổng điều khiển dòng thông tin:
    * **Forget Gate ($f_t$):** Quyết định bỏ bao nhiêu thông tin cũ.
    * **Input Gate ($i_t$):** Quyết định thêm bao nhiêu thông tin mới.
    * **Output Gate ($o_t$):** Quyết định thông tin nào được đưa ra Hidden State ($h_t$).
* **Confidence:** High 

---

### Prediction 3

* **Prediction:** Thời gian huấn luyện và sử dụng bộ nhớ sẽ tăng đáng kể gần như tuyến tính $\mathcal{O}(T)$ theo độ dài chuỗi $T$, đồng thời rủi ro bị Vanishing hoặc Exploding Gradient tăng cao.
* **Reason:**
  * **Tính toán và Bộ nhớ:** Việc lan truyền tiến (Forward Pass) và lan truyền ngược qua thời gian (BPTT) phải duyệt qua $200$ timesteps thay vì $50$. Quá trình BPTT phải lưu giữ toàn bộ các trạng thái trung gian $h_t$ của 200 timesteps trong RAM/VRAM để tính đạo hàm, làm tăng dung lượng bộ nhớ sử dụng.
  * **Ổn định gradient:** Tích của các ma trận Jacobian tăng từ 50 lên 200 thành phần, làm hiện tượng vanishing gradient nghiêm trọng hơn nhiều, khiến RNN càng khó học và lâu hội tụ hơn.
* **Confidence:** High 

---

### Prediction 4

* **Prediction:** Có, thông tin classification chắc chắn có thể bị ảnh hưởng đặc biệt là giảm độ chính xác nếu văn bản mang thông tin quan trọng ở phần sau.
* **Reason:**
  * Nếu thông tin quyết định nhãn (ví dụ: các từ thể hiện cảm xúc/đánh giá then chốt như *"However, the ending was terrible and ruined the whole movie"*) nằm sau token thứ 30, việc truncate sẽ làm mất hoàn toàn ngữ cảnh này.
  * Tuy nhiên, nếu thông tin phân loại tập trung chủ yếu ở 30 từ đầu tiên, mô hình vẫn có thể dự đoán đúng. Nhưng việc mất mát dữ liệu là không thể tránh khỏi đối với các văn bản dài.
* **Confidence:** High 

---

### Prediction 5

* **Prediction:** Không chắc chắn.
* **Reason:**
  * **Độ phức tạp và Overfitting:** LSTM có số lượng tham số gấp khoảng 4 lần so với RNN thuần do chứa 4 ma trận gate khác nhau: $f, i, o, c$. Trên các dataset quá nhỏ, LSTM rất dễ bị hiện tượng overfitting nếu không được regularization đúng cách.
  * **Đặc điểm dữ liệu:** Nếu dataset chỉ bao gồm các chuỗi rất ngắn hoặc các mối quan hệ ngữ nghĩa mang tính ngắn hạn, kiến trúc đơn giản của RNN hoặc thậm chí các mô hình truyền thống (TF-IDF + LR) có thể đạt accuracy tương đương hoặc tốt hơn LSTM do ít bị nhiễu và dễ hội tụ hơn.
  * Không có mô hình nào vượt trội tuyệt đối trên mọi tập dữ liệu.
* **Confidence:** High