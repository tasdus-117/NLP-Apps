# PREDICTION TRƯỚC EXPERIMENT (ATTENTION MECHANISM)

---

## Prediction 1

* **Prediction:** Attention weight của key đó sẽ **lớn nhất**.
* **Reason:** Dot product $s_i = \mathbf{q}^T \mathbf{k}_i$ đo lường mức độ tương đồng giữa Query $\mathbf{q}$ và Key $\mathbf{k}_i$. Trọng số Softmax được định nghĩa là:
  $$\alpha_i = \frac{e^{s_i}}{\sum_{j=1}^N e^{s_j}}$$
  Do hàm mũ $f(x) = e^x$ là hàm đồng biến tăng nghiêm ngặt trên $\mathbb{R}$, giá trị $s_i$ lớn nhất chắc chắn sẽ cho ra tử số $e^{s_i}$ lớn nhất. Vì tất cả các phần tử có chung một mẫu số chuẩn hóa, trọng số $\alpha_i$ tương ứng chắc chắn sẽ đạt giá trị lớn nhất.
* **Confidence:** 100%

---

## Prediction 2

* **Prediction:** Tất cả các attention weights sẽ **bằng nhau** và cùng có giá trị bằng $0.25$.
* **Reason:** Với $\mathbf{s} = [1, 1, 1, 1]^T$, độ dài chuỗi là $N = 4$. Áp dụng công thức Softmax:
  $$\alpha_i = \frac{e^1}{e^1 + e^1 + e^1 + e^1} = \frac{e}{4e} = \frac{1}{4} = 0.25 \quad \forall i \in \{1, 2, 3, 4\}$$
  Khi tất cả các score đều bằng nhau, mô hình phân bổ trọng số attention đồng đều trên toàn bộ danh sách các token.
* **Confidence:** 100%

---

## Prediction 3

* **Prediction:** Attention sẽ **tập trung gần như toàn bộ trọng số ($\approx 99.98\%$) vào phần tử thứ 3** (phần tử có score = 10), trong khi hai phần tử đầu tiên nhận trọng số xấp xỉ $0$.
* **Reason:** Hàm mũ $e^x$ làm tăng kích thước giá trị theo tỷ lệ cấp số nhân. Xét $\mathbf{s} = [1, 1, 10]^T$:
  - $e^1 \approx 2.7183$
  - $e^{10} \approx 22026.4658$
  - Tổng các tử số: $\sum_{j=1}^3 e^{s_j} \approx 2.7183 + 2.7183 + 22026.4658 = 22031.9024$
  - $\alpha_3 = \frac{22026.4658}{22031.9024} \approx 0.999753 \quad (\approx 99.98\%)$
  - $\alpha_1 = \alpha_2 = \frac{2.7183}{22031.9024} \approx 0.000123 \quad (\approx 0.012\%)$

  Hiện tượng này được gọi là sự khuếch đại của hàm Softmax, khiến phân bố trọng số chú ý tiệm cận dạng hard attention hay vector một vị trí nổi bật - one-hot vector).
* **Confidence:** 100% 

---

## Prediction 4

* **Prediction:** Chi phí tính toán và bộ nhớ của cơ chế attention sẽ **tăng 10.000 lần** tăng theo độ phức tạp bậc hai $\mathcal{O}(N^2)$.
* **Reason:** Chi phí tính toán của cơ chế Self-Attention chuẩn giữa ma trận Query $Q \in \mathbb{R}^{N \times d}$ và Key $K \in \mathbb{R}^{N \times d}$ có độ phức tạp thời gian và không gian bộ nhớ là $\mathcal{O}(N^2 \cdot d)$, trong đó $N$ là độ dài chuỗi và $d$ là số chiều vector.
  - Tỷ lệ tăng độ dài chuỗi từ $N_1 = 10$ lên $N_2 = 1000$ là:
    $$\frac{N_2}{N_1} = \frac{1000}{10} = 100 \text{ lần}$$
  - Do chi phí phụ thuộc bậc hai vào $N$, khối lượng phép tính nhân ma trận và kích thước ma trận Softmax ($N \times N$) sẽ tăng thêm:
    $$\left(\frac{N_2}{N_1}\right)^2 = 100^2 = 10.000 \text{ lần}$$
* **Confidence:** 100% 

---

## Prediction 5

* **Prediction:** **Không**, attention không phải lúc nào cũng chọn đúng token mà con người cho là quan trọng nhất.
* **Reason:**
  1. **Tối ưu hóa mục tiêu toán học:** Mô hình học trọng số Attention nhằm tối ưu hóa hàm mất mát trên toàn bộ tập dữ liệu huấn luyện, không phải để bắt chước quy tắc cú pháp hay nhận thức ngữ nghĩa của con người.
  2. **Multi-Head Attention:** Trong các mô hình Transformer thực tế, các head khác nhau đảm nhận các vai trò đa dạng (ví dụ: chú ý đến dấu câu, từ dừng "the/is", vị trí tương đối, hoặc cấu trúc cú pháp). Nhiều head tập trung vào các token trung tính về mặt ý nghĩa nhưng cần thiết cho cơ chế lan truyền thông tin của mạng neural.
  3. **Vấn đề giải thích được:** Các nghiên cứu chuyên sâu (tiêu biểu như bài báo *"Attention is not Explanation"* - Jain & Wallace, 2019) đã chứng minh rằng trọng số attention lớn không tương quan $1:1$ với mức độ đóng góp thực sự của token đó vào kết quả dự báo cuối cùng của mô hình.
* **Confidence:** 95% 