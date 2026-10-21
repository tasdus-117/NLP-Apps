### Bài 1 — Một bước RNN

**Bước 1: Tính $a_t = W_{xh}x_t + W_{hh}h_{t-1} + b_h$**

* **Tính tích $W_{xh}x_t$:**
  $$W_{xh}x_t = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.4 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.5 \cdot 1 + 0.2 \cdot 0 \\ 0.1 \cdot 1 + 0.4 \cdot 0 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix}$$

* **Tính tích $W_{hh}h_{t-1}$:**
  $$W_{hh}h_{t-1} = \begin{bmatrix} 0.3 & 0.1 \\ 0.2 & 0.5 \end{bmatrix} \begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix} = \begin{bmatrix} 0.3 \cdot 0.5 + 0.1 \cdot 0.2 \\ 0.2 \cdot 0.5 + 0.5 \cdot 0.2 \end{bmatrix} = \begin{bmatrix} 0.15 + 0.02 \\ 0.10 + 0.10 \end{bmatrix} = \begin{bmatrix} 0.17 \\ 0.20 \end{bmatrix}$$

* **Cộng với $b_h = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$:**
  $$a_t = \begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix} + \begin{bmatrix} 0.17 \\ 0.20 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.67 \\ 0.30 \end{bmatrix}$$

**Bước 2: Tính $h_t = \tanh(a_t)$**

Áp dụng hàm kích hoạt $\tanh$ cho từng phần tử của $a_t$:
$$h_t = \begin{bmatrix} \tanh(0.67) \\ \tanh(0.30) \end{bmatrix} \approx \begin{bmatrix} 0.585 \\ 0.291 \end{bmatrix}$$

---

### Bài 2 — Information flow

**Mô tả hidden state qua từng bước trong RNN:**
1. $h_1 = \text{RNN}(x_{\text{"I"}}, h_0)$: Chứa thông tin về chủ từ "I".
2. $h_2 = \text{RNN}(x_{\text{"did"}}, h_1)$: Kết hợp "did" với thông tin "I" ở trước.
3. $h_3 = \text{RNN}(x_{\text{"not"}}, h_2)$: Cập nhật thông tin phủ định "not" vào ngữ cảnh ("I did").
4. $h_4 = \text{RNN}(x_{\text{"like"}}, h_3)$: Đọc từ "like" và nhận toàn bộ thông ngữ cảnh từ $h_3$.
5. $h_5 = \text{RNN}(x_{\text{"this"}}, h_4)$ và $h_6 = \text{RNN}(x_{\text{"movie"}}, h_5)$: Tiếp tục tích lũy ngữ cảnh đằng sau.

**Ảnh hưởng của "not" tới hidden state tại thời điểm đọc "like" ($h_4$):**
* **Trong RNN:** $h_4 = \tanh(W_{xh}x_{\text{"like"}} + W_{hh}h_3 + b_h)$. Vì $h_3$ đã nén thông tin từ "not", thông tin phủ định này đi qua ma trận trọng số $W_{hh}$ truyền trực tiếp vào $h_4$. Nhờ đó, $h_4$ không chỉ hiểu từ "like" đơn lẻ mà hiểu được hành động mang ý nghĩa phủ định ("not like").
* **So sánh với Average Word2Vec:**
  * **Average Word2Vec:** Vector biểu diễn được tính bằng trung bình cộng không trọng số của các vector từ đơn lẻ: $v = \frac{1}{N} \sum v_{i}$. Phương pháp này hoàn toàn loại bỏ mối quan hệ phụ thuộc tuyến tính và khoảng cách giữa các từ. "not" và "like" chỉ đóng góp hai vector độc lập vào tổng thể mà không tương tác trực tiếp để đảo ngược ý nghĩa của nhau.
  * **Khác biệt:** RNN duy trì một bộ nhớ trạng thái biến đổi qua thời gian, giúp từ "not" biến đổi trạng thái ngữ cảnh trước khi đến "like", giữ lại được tính ngữ nghĩa phức hợp.

---

### Bài 3 — Sequence order

**1. Khả năng phân biệt của Average Word2Vec dựa trên thứ tự:**
* **Trả lời:** **Không.**
* **Cơ chế:** Phép cộng/trung bình các vector có tính chất giao hoán:
  $$v(S_1) = \frac{1}{3}(v_{\text{dog}} + v_{\text{bites}} + v_{\text{man}}) = \frac{1}{3}(v_{\text{man}} + v_{\text{bites}} + v_{\text{dog}}) = v(S_2)$$
  Do $S_1$ và $S_2$ có cùng vocabulary, biểu diễn vector của hai câu là **hoàn toàn giống hệt nhau**, làm mất sạch thông tin vị trí.

**2. Khả năng phân biệt của RNN dựa trên thứ tự:**
* **Trả lời:** **Có.**
* **Cơ chế:** RNN xử lý chuỗi theo trình tự thời gian tuần tự thông qua công thức đệ quy $h_t = f(W_{xh}x_t + W_{hh}h_{t-1})$. Trạng thái ẩn tại mỗi bước phụ thuộc vào **thứ tự xuất hiện** của các từ:
  * **Đối với $S_1$ ("dog bites man"):**
    * $h_1 = f(W_{xh}x_{\text{dog}} + W_{hh}h_0)$ (Chủ thể thực hiện là "dog")
    * $h_2 = f(W_{xh}x_{\text{bites}} + W_{hh}h_1)$
    * $h_3 = f(W_{xh}x_{\text{man}} + W_{hh}h_2)$ (Đối tượng chịu tác động là "man")
  * **Đối với $S_2$ ("man bites dog"):**
    * $h'_1 = f(W_{xh}x_{\text{man}} + W_{hh}h_0)$ (Chủ thể thực hiện là "man")
    * $h'_2 = f(W_{xh}x_{\text{bites}} + W_{hh}h'_1)$
    * $h'_3 = f(W_{xh}x_{\text{dog}} + W_{hh}h'_2)$ (Đối tượng chịu tác động là "dog")

Vì phép nhân ma trận không có tính chất giao hoán và $h_{t-1}$ liên tục đóng vai trò ngữ cảnh cho $x_t$, kết quả $h_3 \neq h'_3$. RNN phân biệt được ai là chủ thể hành động dựa trên thứ tự xuất hiện trong chuỗi.

---

### Bài 4 — Vanishing gradient

**Giải thích lý do chuỗi dài khiến RNN khó học phụ thuộc xa (long-term dependencies):**

Khi huấn luyện RNN bằng thuật toán **Backpropagation Through Time (BPTT)**, để cập nhật trọng số theo lỗi tại thời điểm $T$ dựa trên thông tin ở thời điểm $t$ ($t \ll T$), ta cần tính đạo hàm $\frac{\partial L_T}{\partial h_t}$. 

Theo quy tắc chuỗi, đạo hàm này phụ thuộc vào tích của chuỗi ma trận Jacobian liên tiếp:
$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}$$

Đạo hàm thành phần tại từng timestep là:
$$\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}(1 - \tanh^2(a_k)) \cdot W_{hh}^T$$

Nguyên nhân làm vanishing gradient qua khoảng cách xa:
1. **Giá trị của hàm kích hoạt:** Đạo hàm của hàm $\tanh$ thỏa mãn $0 < 1 - \tanh^2(x) \le 1$. Tích của nhiều giá trị nhỏ hơn hoặc bằng 1 liên tiếp sẽ tiến về 0 rất nhanh.
2. **Lũy thừa ma trận trọng số:** Chuỗi càng dài, số lượng ma trận $W_{hh}^T$ nhân liên tiếp càng lớn. Nếu các giá trị riêng của $W_{hh}$ nhỏ hơn 1, độ lớn của tích ma trận sẽ suy giảm theo cấp số nhân và tiến về 0.

**Hệ quả:** Khi $T - t$ lớn, $\frac{\partial L_T}{\partial h_t} \approx 0$. Tín hiệu lỗi từ tương lai không thể truyền ngược hiệu quả về các timestep ở xa trong quá khứ. Trọng số của mô hình không nhận được tín hiệu cập nhật phù hợp, khiến RNN quên thông tin xa và chỉ học được các mối liên hệ ngắn hạn.

### Bài 17 — Bài tập tính toán: LSTM gates

#### 1. Tính toán Cell State $c_t$

Công thức cập nhật Cell State tại bước thời gian $t$:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Thay số:**
* $f_t = 0.9$ (Forget gate)
* $i_t = 0.2$ (Input gate)
* $\tilde{c}_t = 0.5$ (Candidate cell state)
* $c_{t-1} = 0.8$ (Cell state ở bước thời gian trước)

**Chi tiết từng bước tính:**
1. **Lượng thông tin cũ được giữ lại từ $c_{t-1}$:**
   $$f_t \cdot c_{t-1} = 0.9 \cdot 0.8 = 0.72$$

2. **Lượng thông tin mới được cập nhật từ $\tilde{c}_t$:**
   $$i_t \cdot \tilde{c}_t = 0.2 \cdot 0.5 = 0.10$$

3. **Cell state mới $c_t$:**
   $$c_t = 0.72 + 0.10 = 0.82$$

---

#### 2. Cơ chế điều khiển dòng thông tin

* **Nếu Forget Gate ($f_t$) gần $1$:**
  * **Xu hướng:** Mô hình có xu hướng **retain / preserve** hầu như toàn bộ thông tin từ Cell State cũ ($c_{t-1}$).
  * **Giải thích:** Phép nhân $1 \cdot c_{t-1} \approx c_{t-1}$ cho phép bộ nhớ dài hạn truyền qua thời gian mà không bị suy giảm hay biến đổi, giúp mô hình lưu giữ được ngữ cảnh từ rất xa khắc phục hiện tượng Vanishing Gradient.

* **Nếu Forget Gate ($f_t$) gần $0$:**
  * **Xu hướng:** Mô hình có xu hướng **forget / erase** thông tin cũ trong $c_{t-1}$.
  * **Giải thích:** Phép nhân $0 \cdot c_{t-1} \approx 0$ triệt tiêu bộ nhớ cũ, giúp mô hình reset lại trạng thái bộ nhớ để sẵn sàng ghi nhận một chủ đề hoặc ngữ cảnh hoàn toàn mới trong chuỗi.