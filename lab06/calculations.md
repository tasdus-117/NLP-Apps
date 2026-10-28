# Bài tập tính toán

---

## Bài 1 — Attention scores

### 1. Đề bài
Cho vector truy vấn $\mathbf{q}$ và ma trận chìa khóa $K$:

$$\mathbf{q} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$

Tính từng điểm số chú ý (attention score) $s_i = \mathbf{q}^T \mathbf{k}_i$ và viết toàn bộ vector score $\mathbf{s}$.

---

### 2. Lời giải

Ma trận $K$ có $3$ hàng tương ứng với $3$ vector key $\mathbf{k}_1, \mathbf{k}_2, \mathbf{k}_3$:

$$\mathbf{k}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{k}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad \mathbf{k}_3 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

Tính từng score $s_i = \mathbf{q}^T \mathbf{k}_i$:

- **Tính $s_1$:**
  $$s_1 = \mathbf{q}^T \mathbf{k}_1 = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = 1 \cdot 1 + 2 \cdot 0 = 1$$

- **Tính $s_2$:**
  $$s_2 = \mathbf{q}^T \mathbf{k}_2 = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = 1 \cdot 0 + 2 \cdot 1 = 2$$

- **Tính $s_3$:**
  $$s_3 = \mathbf{q}^T \mathbf{k}_3 = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 1 \cdot 1 + 2 \cdot 1 = 3$$

**Kết luận:** Vector score toàn bộ $\mathbf{s}$ là:

$$\mathbf{s} = \begin{bmatrix} s_1 \\ s_2 \\ s_3 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \quad \text{hoặc} \quad \mathbf{s} = [1, 2, 3]^T$$

---

## Bài 2 — Softmax

### 1. Đề bài
Cho vector score $\mathbf{s} = [1, 2, 3]^T$.
- Tính vector trọng số chú ý $\boldsymbol{\alpha} = \text{softmax}(\mathbf{s})$.
- Chứng minh $\sum_{i} \alpha_i = 1$.
- Trả lời câu hỏi: *Vì sao phần tử có score lớn nhất nhận weight lớn nhất?*

---

### 2. Lời giải

#### a) Tính $\boldsymbol{\alpha} = \text{softmax}(\mathbf{s})$

Công thức hàm Softmax cho từng phần tử $\alpha_i$:

$$\alpha_i = \frac{e^{s_i}}{\sum_{j=1}^{3} e^{s_j}}$$

Tính các giá trị mũ $e^{s_i}$:
- $e^{s_1} = e^1 \approx 2.7183$
- $e^{s_2} = e^2 \approx 7.3891$
- $e^{s_3} = e^3 \approx 20.0855$

Tổng mẫu số:

$$\sum_{j=1}^{3} e^{s_j} = e^1 + e^2 + e^3 \approx 2.7183 + 7.3891 + 20.0855 = 30.1929$$

Tính từng trọng số $\alpha_i$:

- $\alpha_1 = \frac{e^1}{e^1 + e^2 + e^3} = \frac{2.7183}{30.1929} \approx 0.0900$
- $\alpha_2 = \frac{e^2}{e^1 + e^2 + e^3} = \frac{7.3891}{30.1929} \approx 0.2447$
- $\alpha_3 = \frac{e^3}{e^1 + e^2 + e^3} = \frac{20.0855}{30.1929} \approx 0.6653$

Vậy:

$$\boldsymbol{\alpha} \approx [0.0900, 0.2447, 0.6653]^T$$

---

#### b) Chỉ ra $\sum_{i} \alpha_i = 1$

Về mặt đại số:

$$\sum_{i=1}^{3} \alpha_i = \alpha_1 + \alpha_2 + \alpha_3 = \frac{e^{s_1}}{\sum_{j} e^{s_j}} + \frac{e^{s_2}}{\sum_{j} e^{s_j}} + \frac{e^{s_3}}{\sum_{j} e^{s_j}} = \frac{e^{s_1} + e^{s_2} + e^{s_3}}{e^{s_1} + e^{s_2} + e^{s_3}} = 1$$

Về mặt giá trị số:

$$\sum_{i=1}^{3} \alpha_i \approx 0.0900 + 0.2447 + 0.6653 = 1.0000$$

---

#### c) Giải thích: Vì sao phần tử có score lớn nhất nhận weight lớn nhất?

1. **Tính đơn điệu tăng của hàm mũ ($e^x$):** Hàm $f(x) = e^x$ là hàm số đồng biến nghiêm ngặt trên tập số thực $\mathbb{R}$. Do đó, nếu $s_i > s_j$ thì luôn kéo theo $e^{s_i} > e^{s_j}$.
2. **Mẫu số chuẩn hóa chung:** Mẫu số $Z = \sum_{j} e^{s_j}$ là một hằng số dương cố định cho tất cả các phần tử. Khi tử số $e^{s_i}$ lớn nhất thì tỷ lệ $\alpha_i = \frac{e^{s_i}}{Z}$ chắc chắn sẽ lớn nhất.
3. **Hiệu ứng khuếch đại:** Hàm mũ không chỉ bảo toàn thứ tự mà còn khuếch đại sự chênh lệch giữa các score. Nhờ đó, mô hình có thể tập trung mạnh vào vị trí phù hợp nhất và làm mờ đi các vị trí ít liên quan.

---

## Bài 3 — Context vector

### 1. Đề bài
Cho trọng số chú ý $\boldsymbol{\alpha} = [0.1, 0.2, 0.7]^T$ và các vector giá trị:

$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad \mathbf{v}_3 = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$

- Tính vector ngữ cảnh $\mathbf{c} = 0.1 \mathbf{v}_1 + 0.2 \mathbf{v}_2 + 0.7 \mathbf{v}_3$.
- Giải thích: Value nào đóng góp nhiều nhất vào context vector?

---

### 2. Lời giải

#### a) Tính vector ngữ cảnh $\mathbf{c}$

$$\mathbf{c} = 0.1 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 0.2 \begin{bmatrix} 0 \\ 1 \end{bmatrix} + 0.7 \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$

$$\mathbf{c} = \begin{bmatrix} 0.1 \cdot 1 \\ 0.1 \cdot 0 \end{bmatrix} + \begin{bmatrix} 0.2 \cdot 0 \\ 0.2 \cdot 1 \end{bmatrix} + \begin{bmatrix} 0.7 \cdot 2 \\ 0.7 \cdot 2 \end{bmatrix}$$

$$\mathbf{c} = \begin{bmatrix} 0.1 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ 0.2 \end{bmatrix} + \begin{bmatrix} 1.4 \\ 1.4 \end{bmatrix} = \begin{bmatrix} 0.1 + 0 + 1.4 \\ 0 + 0.2 + 1.4 \end{bmatrix} = \begin{bmatrix} 1.5 \\ 1.6 \end{bmatrix}$$

---

#### b) Giải thích Value đóng góp nhiều nhất

- **Kết luận:** Vector giá trị **$\mathbf{v}_3$** đóng góp nhiều nhất vào context vector $\mathbf{c}$.
- **Lý do:** Trọng số chú ý tương ứng với $\mathbf{v}_3$ là $\alpha_3 = 0.7$ (chiếm $70\%$ tổng trọng số của cả hệ thống), lớn hơn nhiều so với $\alpha_1 = 0.1$ và $\alpha_2 = 0.2$. Vì $\mathbf{c}$ là tổng có trọng số, phần tử có trọng số lớn hơn sẽ đóng góp nhiều hơn vào độ lớn và hướng của vector kết quả.

---

## Bài 4 — Toàn bộ attention computation

### 1. Đề bài
Cho:
- Query: $\mathbf{q} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
- Keys: $\mathbf{k}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{k}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad \mathbf{k}_3 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$
- Values: $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 0 \\ 2 \end{bmatrix}, \quad \mathbf{v}_3 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$

Thực hiện quy trình chú ý đầy đủ:
$$\mathbf{q} \xrightarrow{\quad} \text{scores} \xrightarrow{\quad} \text{softmax} \xrightarrow{\quad} \text{attention weights} \xrightarrow{\quad} \text{weighted sum} \xrightarrow{\quad} \text{context vector}$$

---

### 2. Chi tiết các bước tính toán

#### Bước 1: Tính Attention Scores ($s_i = \mathbf{q}^T \mathbf{k}_i$)

- $s_1 = \mathbf{q}^T \mathbf{k}_1 = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = 1 \cdot 1 + 0 \cdot 0 = 1$
- $s_2 = \mathbf{q}^T \mathbf{k}_2 = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = 1 \cdot 0 + 0 \cdot 1 = 0$
- $s_3 = \mathbf{q}^T \mathbf{k}_3 = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 1 \cdot 1 + 0 \cdot 1 = 1$

$\rightarrow$ Vector scores: $\mathbf{s} = [1, 0, 1]^T$

---

#### Bước 2 & 3: Tính Softmax và Attention Weights ($\boldsymbol{\alpha}$)

Tính các giá trị $e^{s_i}$:
- $e^{s_1} = e^1 \approx 2.7183$
- $e^{s_2} = e^0 = 1.0000$
- $e^{s_3} = e^1 \approx 2.7183$

Tổng các giá trị mũ:

$$\sum_{j=1}^{3} e^{s_j} = e + 1 + e = 2e + 1 \approx 2.7183 + 1.0000 + 2.7183 = 6.4366$$

Tính từng trọng số $\alpha_i$:

- $\alpha_1 = \frac{e^1}{2e + 1} \approx \frac{2.7183}{6.4366} \approx 0.4223$
- $\alpha_2 = \frac{1}{2e + 1} \approx \frac{1.0000}{6.4366} \approx 0.1554$
- $\alpha_3 = \frac{e^1}{2e + 1} \approx \frac{2.7183}{6.4366} \approx 0.4223$

$\rightarrow$ Vector trọng số chú ý: $\boldsymbol{\alpha} \approx [0.4223, 0.1554, 0.4223]^T$

---

#### Bước 4 & 5: Weighted Sum và thu được Context Vector ($\mathbf{c}$)

Công thức tổng có trọng số:

$$\mathbf{c} = \sum_{i=1}^{3} \alpha_i \mathbf{v}_i = \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \alpha_3 \mathbf{v}_3$$

Thay số vào biểu thức:

$$\mathbf{c} = 0.4223 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 0.1554 \begin{bmatrix} 0 \\ 2 \end{bmatrix} + 0.4223 \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

$$\mathbf{c} = \begin{bmatrix} 0.4223 \cdot 1 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ 0.1554 \cdot 2 \end{bmatrix} + \begin{bmatrix} 0.4223 \cdot 2 \\ 0.4223 \cdot 1 \end{bmatrix}$$

$$\mathbf{c} = \begin{bmatrix} 0.4223 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ 0.3108 \end{bmatrix} + \begin{bmatrix} 0.8446 \\ 0.4223 \end{bmatrix}$$

$$\mathbf{c} = \begin{bmatrix} 0.4223 + 0 + 0.8446 \\ 0 + 0.3108 + 0.4223 \end{bmatrix} = \begin{bmatrix} 1.2669 \\ 0.7331 \end{bmatrix}$$

*(Dưới dạng phân thức chính xác chứa $e$: $\mathbf{c} = \frac{1}{2e+1} \begin{bmatrix} 3e \\ e + 2 \end{bmatrix}$)*

---

### 3. Tóm tắt toàn bộ luồng tính toán

$$\mathbf{q} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \implies \mathbf{s} = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} \implies \boldsymbol{\alpha} \approx \begin{bmatrix} 0.4223 \\ 0.1554 \\ 0.4223 \end{bmatrix} \implies \mathbf{c} \approx \begin{bmatrix} 1.2669 \\ 0.7331 \end{bmatrix}$$


## Bài tập — Scaling
Cho:
- Chiều của key: $d_k = 4$
- Ma trận tích vô hướng chưa scale: $QK^T = [4, 2, 0]$

Yêu cầu:
1. Tính scaled scores: $\frac{QK^T}{\sqrt{d_k}}$.
2. Tính Softmax của scaled scores.
3. Tính Softmax của unscaled scores để so sánh.
4. Trả lời: *Scaling làm thay đổi độ tập trung của Softmax như thế nào?*

---

## 2. Chi tiết các bước tính toán

### Bước 1: Tính Scaled Scores

Ta có $\sqrt{d_k} = \sqrt{4} = 2$.

Vector score sau khi scale là:

$$\mathbf{s}_{\text{scaled}} = \frac{QK^T}{\sqrt{d_k}} = \frac{[4, 2, 0]}{2} = [2, 1, 0]$$

---

### Bước 2: Tính Softmax cho Scaled Scores ($\boldsymbol{\alpha}_{\text{scaled}}$)

Với $\mathbf{s}_{\text{scaled}} = [2, 1, 0]$:

- $e^{s_1} = e^2 \approx 7.3891$
- $e^{s_2} = e^1 \approx 2.7183$
- $e^{s_3} = e^0 = 1.0000$

**Tổng mẫu số:**

$$\sum_{j=1}^{3} e^{s_j} = e^2 + e^1 + e^0 \approx 7.3891 + 2.7183 + 1.0000 = 11.1074$$

**Các trọng số Softmax:**

- $\alpha_{\text{scaled}, 1} = \frac{e^2}{11.1074} \approx \frac{7.3891}{11.1074} \approx 0.6652 \quad (66.52\%)$
- $\alpha_{\text{scaled}, 2} = \frac{e^1}{11.1074} \approx \frac{2.7183}{11.1074} \approx 0.2447 \quad (24.47\%)$
- $\alpha_{\text{scaled}, 3} = \frac{e^0}{11.1074} \approx \frac{1.0000}{11.1074} \approx 0.0901 \quad (9.01\%)$

$$\rightarrow \boldsymbol{\alpha}_{\text{scaled}} \approx [0.6652, 0.2447, 0.0901]^T$$

---

### Bước 3: Tính Softmax cho Unscaled Scores ($\boldsymbol{\alpha}_{\text{unscaled}}$)

Với $\mathbf{s}_{\text{unscaled}} = QK^T = [4, 2, 0]$:

- $e^{s_1} = e^4 \approx 54.5982$
- $e^{s_2} = e^2 \approx 7.3891$
- $e^{s_3} = e^0 = 1.0000$

**Tổng mẫu số:**

$$\sum_{j=1}^{3} e^{s_j} = e^4 + e^2 + e^0 \approx 54.5982 + 7.3891 + 1.0000 = 62.9873$$

**Các trọng số Softmax:**

- $\alpha_{\text{unscaled}, 1} = \frac{e^4}{62.9873} \approx \frac{54.5982}{62.9873} \approx 0.8668 \quad (86.68\%)$
- $\alpha_{\text{unscaled}, 2} = \frac{e^2}{62.9873} \approx \frac{7.3891}{62.9873} \approx 0.1173 \quad (11.73\%)$
- $\alpha_{\text{unscaled}, 3} = \frac{e^0}{62.9873} \approx \frac{1.0000}{62.9873} \approx 0.0159 \quad (1.59\%)$

$$\rightarrow \boldsymbol{\alpha}_{\text{unscaled}} \approx [0.8668, 0.1173, 0.0159]^T$$

---

## 3. Bảng so sánh kết quả

| Chỉ số | Không Scaling ($QK^T$) | Có Scaling ($\frac{QK^T}{\sqrt{d_k}}$) |
| :--- | :---: | :---: |
| **Vector Scores** | $[4, 2, 0]$ | $[2, 1, 0]$ |
| **Trọng số $\alpha_1$** | **$86.68\%$** | **$66.52\%$** |
| **Trọng số $\alpha_2$** | $11.73\%$ | $24.47\%$ |
| **Trọng số $\alpha_3$** | $1.59\%$ | $9.01\%$ |
| **Đặc điểm phân bố** | Độ tập trung rất cao | Độ tập trung vừa phải |

---

## 4. Trả lời câu hỏi: Scaling làm thay đổi độ tập trung của Softmax như thế nào?

1. **Làm giảm độ tập trung cực đoan:** 
   - Khi **không scale**, trọng số chú ý dồn rất mạnh vào vị trí lớn nhất ($\alpha_1$ đạt tới $86.68\%$), các vị trí còn lại bị đẩy về sát $0$.
   - Khi **có scale**, độ chênh lệch giữa các score giảm bớt ($[4, 2, 0] \rightarrow [2, 1, 0]$), giúp phân bố trọng số Softmax "mềm" hơn ($\alpha_1$ hạ xuống $66.52\%$, $\alpha_2$ và $\alpha_3$ nhận thêm trọng số).

2. **Ý nghĩa kỹ thuật trong mô hình Transformer:**
   - Khi số chiều $d_k$ lớn, tích vô hướng $\mathbf{q}^T \mathbf{k}$ thường có giá trị tuyệt đối rất lớn.
   - Các giá trị score quá lớn sẽ đẩy hàm Softmax vào **vùng bão hòa** với độ dốc cực kỳ nhỏ.
   - Việc thu nhỏ các score bằng hằng số $\sqrt{d_k}$ giữ cho gradient của Softmax không bị triệt tiêu, giúp việc huấn luyện bằng backpropagation diễn ra ổn định.

## 1. Temperature

Cho công thức Softmax có Temperature $T$:

$$\text{softmax}\left(\frac{\mathbf{s}}{T}\right)_i = \frac{e^{\frac{s_i}{T}}}{\sum_{j} e^{\frac{s_j}{T}}}$$

Thực hiện tính toán với vector score $\mathbf{s} = [1, 2, 3]^T$ tại 3 mức nhiệt độ:
- $T = 0.5$
- $T = 1.0$
- $T = 2.0$

Lập bảng ghi chép các chỉ số: **Temperature**, **Max weight**, **Entropy**, và **Nhận xét**.

---

## 2. Công thức tính toán

1. **Trọng số chú ý:** 
   $$\alpha_i = \text{softmax}\left(\frac{\mathbf{s}}{T}\right)_i$$

2. **Max weight:** 
   $$\text{Max weight} = \max_i (\alpha_i)$$

3. **Entropy (Độ hỗn loạn / Độ phẳng của phân bố):**
   $$H(\boldsymbol{\alpha}) = -\sum_{i} \alpha_i \ln(\alpha_i) \quad (\text{đơn vị: nats})$$

---

## 3. Chi tiết tính toán theo từng Temperature

### Trường hợp 1: $T = 0.5$

Scaled score vector: $\frac{\mathbf{s}}{0.5} = [2, 4, 6]^T$

- $e^2 \approx 7.3891$
- $e^4 \approx 54.5982$
- $e^6 \approx 403.4288$
- Tổng mẫu số $\sum e^{s_i / T} \approx 465.4161$

**Vector trọng số $\boldsymbol{\alpha}$:**
- $\alpha_1 = \frac{7.3891}{465.4161} \approx 0.0159 \quad (1.59\%)$
- $\alpha_2 = \frac{54.5982}{465.4161} \approx 0.1173 \quad (11.73\%)$
- $\alpha_3 = \frac{403.4288}{465.4161} \approx 0.8668 \quad (86.68\%)$

$$\boldsymbol{\alpha}_{T=0.5} \approx [0.0159, 0.1173, 0.8668]^T$$

- **Max weight:** $0.8668$ ($86.68\%$)
- **Entropy:** 
  $$H = -(0.0159 \ln 0.0159 + 0.1173 \ln 0.1173 + 0.8668 \ln 0.8668) \approx 0.4411 \text{ nats}$$

---

### Trường hợp 2: $T = 1.0$

Scaled score vector: $\frac{\mathbf{s}}{1.0} = [1, 2, 3]^T$

- $e^1 \approx 2.7183$
- $e^2 \approx 7.3891$
- $e^3 \approx 20.0855$
- Tổng mẫu số $\sum e^{s_i / T} \approx 30.1929$

**Vector trọng số $\boldsymbol{\alpha}$:**
- $\alpha_1 = \frac{2.7183}{30.1929} \approx 0.0900 \quad (9.00\%)$
- $\alpha_2 = \frac{7.3891}{30.1929} \approx 0.2447 \quad (24.47\%)$
- $\alpha_3 = \frac{20.0855}{30.1929} \approx 0.6653 \quad (66.53\%)$

$$\boldsymbol{\alpha}_{T=1.0} \approx [0.0900, 0.2447, 0.6653]^T$$

- **Max weight:** $0.6653$ ($66.53\%$)
- **Entropy:** 
  $$H = -(0.0900 \ln 0.0900 + 0.2447 \ln 0.2447 + 0.6653 \ln 0.6653) \approx 0.8324 \text{ nats}$$

---

### Trường hợp 3: $T = 2.0$ 

Scaled score vector: $\frac{\mathbf{s}}{2.0} = [0.5, 1.0, 1.5]^T$

- $e^{0.5} \approx 1.6487$
- $e^{1.0} \approx 2.7183$
- $e^{1.5} \approx 4.4817$
- Tổng mẫu số $\sum e^{s_i / T} \approx 8.8487$

**Vector trọng số $\boldsymbol{\alpha}$:**
- $\alpha_1 = \frac{1.6487}{8.8487} \approx 0.1863 \quad (18.63\%)$
- $\alpha_2 = \frac{2.7183}{8.8487} \approx 0.3072 \quad (30.72\%)$
- $\alpha_3 = \frac{4.4817}{8.8487} \approx 0.5065 \quad (50.65\%)$

$$\boldsymbol{\alpha}_{T=2.0} \approx [0.1863, 0.3072, 0.5065]^T$$

- **Max weight:** $0.5065$ ($50.65\%$)
- **Entropy:** 
  $$H = -(0.1863 \ln 0.1863 + 0.3072 \ln 0.3072 + 0.5065 \ln 0.5065) \approx 1.0201 \text{ nats}$$

---

## 4. Bảng tổng hợp kết quả

| Temperature ($T$) | Vector Trọng số ($\boldsymbol{\alpha}$) | Max weight | Entropy (nats) | Nhận xét |
| :---: | :---: | :---: | :---: | :--- |
| **0.5** | $[0.0159, 0.1173, 0.8668]$ | **0.8668** (86.68%) | **0.4411** | **Tập trung rất cao:** Trọng số dồn mạnh vào phần tử lớn nhất; phân bố tiệm cận hàm $\text{argmax}$. Entropy nhỏ thể hiện độ vô định thấp. |
| **1.0** | $[0.0900, 0.2447, 0.6653]$ | **0.6653** (66.53%) | **0.8324** | **Cân bằng tiêu chuẩn:** Giữ nguyên phân bố gốc của bài toán Softmax thông thường. |
| **2.0** | $[0.1863, 0.3072, 0.5065]$ | **0.5065** (50.65%) | **1.0201** | **Phẳng hơn:** Sự chênh lệch giữa các trọng số bị giảm đáng kể; phân bố tiệm cận phân bố đều (Uniform). Entropy cao thể hiện độ đa dạng lớn. |

---

## 5. Kết luận về ảnh hưởng của Temperature

1. **Khi $T \to 0$ (Nhiệt độ thấp):**
   - Làm khuếch đại sự chênh lệch giữa các score.
   - Phân bố trở nên **sắc nhọn** và tập trung hầu như hoàn toàn vào vị trí có score lớn nhất (dẫn tới **Max weight tăng**, **Entropy giảm**).
   - *Ứng dụng trong LLM:* Dùng khi cần mô hình trả lời chính xác, mang tính quyết đoán cao (ví dụ: giải toán, trích xuất thông tin, viết mã nguồn).

2. **Khi $T \to \infty$ (Nhiệt độ cao):**
   - Làm san bằng khoảng cách giữa các score.
   - Phân bố trở nên **phẳng hơn** và dàn đều xác suất cho tất cả phần tử (dẫn tới **Max weight giảm**, **Entropy tăng**).
   - *Ứng dụng trong LLM:* Dùng khi cần mô hình sinh nội dung sáng tạo, đa dạng từ ngữ (ví dụ: viết văn, làm thơ, brainstorming).



## Lời giải Bài 21 — Cơ chế Self-Attention

### 1. Giả định Dữ liệu Đầu vào (Input Setup)

Cho chuỗi đầu vào gồm 3 token:
$$T = [\text{"I"}, \text{"like"}, \text{"cats"}]$$

#### Ma trận Embedding ($X \in \mathbb{R}^{3 \times 2}$):
Mỗi token được biểu diễn bằng một vector $d_e = 2$ chiều:
$$X = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{bmatrix} \quad \begin{matrix} \text{("I")} \\ \text{("like")} \\ \text{("cats")} \end{matrix}$$

#### Các ma trận trọng số chiếu ($W_Q, W_K, W_V \in \mathbb{R}^{2 \times 2}$):
Giả sử $d_k = d_v = 2$:
$$W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad W_K = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad W_V = \begin{bmatrix} 1 & 0 \\ 1 & 2 \end{bmatrix}$$

---

### 2. Bước 1: Tính các ma trận Query ($Q$), Key ($K$), Value ($V$)

#### Ma trận Query ($Q = X W_Q$):
$$Q = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{bmatrix}$$

#### Ma trận Key ($K = X W_K$):
$$K = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 0 & 1 \end{bmatrix}$$

#### Ma trận Value ($V = X W_V$):
$$V = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 2 & 2 \\ 1 & 2 \end{bmatrix}$$

---

### 3. Bước 2: Tính Ma trận Tương đồng Scores ($Q K^T$)

Ma trận chuyển vị của $K$:
$$K^T = \begin{bmatrix} 1 & 1 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

Thực hiện nhân ma trận $Q K^T \in \mathbb{R}^{3 \times 3}$:
$$Q K^T = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 & 0 \\ 1 & 2 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 1 & 0 \\ 2 & 3 & 1 \\ 1 & 2 & 1 \end{bmatrix}$$

**Chi tiết phép tính tích vô hướng (Dot Products):**
* **Hàng 1 ("I"):**
  * $q_{\text{I}} \cdot k_{\text{I}}^T = 1(1) + 0(1) = 1$
  * $q_{\text{I}} \cdot k_{\text{like}}^T = 1(1) + 0(2) = 1$
  * $q_{\text{I}} \cdot k_{\text{cats}}^T = 1(0) + 0(1) = 0$
* **Hàng 2 ("like"):**
  * $q_{\text{like}} \cdot k_{\text{I}}^T = 1(1) + 1(1) = 2$
  * $q_{\text{like}} \cdot k_{\text{like}}^T = 1(1) + 1(2) = 3$
  * $q_{\text{like}} \cdot k_{\text{cats}}^T = 1(0) + 1(1) = 1$
* **Hàng 3 ("cats"):**
  * $q_{\text{cats}} \cdot k_{\text{I}}^T = 0(1) + 1(1) = 1$
  * $q_{\text{cats}} \cdot k_{\text{like}}^T = 0(1) + 1(2) = 2$
  * $q_{\text{cats}} \cdot k_{\text{cats}}^T = 0(0) + 1(1) = 1$

---

### 4. Bước 3: Chia tỷ lệ Scaling ($S = \frac{Q K^T}{\sqrt{d_k}}$)

Với $d_k = 2 \implies \sqrt{d_k} = \sqrt{2} \approx 1.4142$:

$$S = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 & 0 \\ 2 & 3 & 1 \\ 1 & 2 & 1 \end{bmatrix} \approx \begin{bmatrix} 0.7071 & 0.7071 & 0.0000 \\ 1.4142 & 2.1213 & 0.7071 \\ 0.7071 & 1.4142 & 0.7071 \end{bmatrix}$$

---

### 5. Bước 4: Tính Ma trận Trọng số Attention ($A = \text{softmax}(S)$)

Áp dụng hàm Softmax theo từng hàng để chuẩn hóa tổng trọng số bằng $1$:

1. **Hàng 1 ("I"):** $s_1 = [0.7071, 0.7071, 0.0000]$
   * $e^{0.7071} \approx 2.0281, \quad e^0 = 1.0000$
   * Tổng $= 2.0281 + 2.0281 + 1.0000 = 5.0562$
   * $\alpha_1 = [0.4011, 0.4011, 0.1978]$

2. **Hàng 2 ("like"):** $s_2 = [1.4142, 2.1213, 0.7071]$
   * $e^{1.4142} \approx 4.1132, \quad e^{2.1213} \approx 8.3420, \quad e^{0.7071} \approx 2.0281$
   * Tổng $= 4.1132 + 8.3420 + 2.0281 = 14.4833$
   * $\alpha_2 = [0.2840, 0.5760, 0.1400]$

3. **Hàng 3 ("cats"):** $s_3 = [0.7071, 1.4142, 0.7071]$
   * $e^{0.7071} \approx 2.0281, \quad e^{1.4142} \approx 4.1132, \quad e^{0.7071} \approx 2.0281$
   * Tổng $= 2.0281 + 4.1132 + 2.0281 = 8.1694$
   * $\alpha_3 = [0.2483, 0.5035, 0.2483]$

**Ma trận Trọng số Attention ($A$):**
$$A = \begin{bmatrix} 0.4011 & 0.4011 & 0.1978 \\ 0.2840 & 0.5760 & 0.1400 \\ 0.2483 & 0.5035 & 0.2483 \end{bmatrix}$$

---

### 6. Bước 5: Tính Ma trận Biểu diễn Ngữ cảnh ($Z = A V$)

$$Z = \begin{bmatrix} 0.4011 & 0.4011 & 0.1978 \\ 0.2840 & 0.5760 & 0.1400 \\ 0.2483 & 0.5035 & 0.2483 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 2 & 2 \\ 1 & 2 \end{bmatrix}$$

**Kết quả ma trận đầu ra ($Z \in \mathbb{R}^{3 \times 2}$):**
$$Z = \begin{bmatrix} 1.4011 & 1.1978 \\ 1.5760 & 1.4320 \\ 1.5036 & 1.5036 \end{bmatrix}$$

---

### 7. Phân tích Bản chất Tương tác trong Self-Attention

| Token | Vector Embedding Ban đầu ($X$) | Vector Ngữ cảnh Mới ($Z$) | Nguồn đóng góp Trọng số Attention ($A$) |
| :--- | :---: | :---: | :--- |
| **"I"** | $[1.0, 0.0]$ | $[1.4011, 1.1978]$ | $40.11\%$ ("I") + $40.11\%$ ("like") + $19.78\%$ ("cats") |
| **"like"** | $[1.0, 1.0]$ | $[1.5760, 1.4320]$ | $28.40\%$ ("I") + $57.60\%$ ("like") + $14.00\%$ ("cats") |
| **"cats"** | $[0.0, 1.0]$ | $[1.5036, 1.5036]$ | $24.83\%$ ("I") + $50.35\%$ ("like") + $24.83\%$ ("cats") |

#### **Nhận xét quan trọng:**
1. **Sự tương tác đa chiều:** Ban đầu, từ **"cats"** chỉ mang thông tin cục bộ riêng lẻ $x_3 = [0, 1]$. Qua cơ chế Self-Attention, vector ngữ cảnh mới $z_3 = [1.5036, 1.5036]$ đã tích hợp:
   * **$24.83\%$** thông tin từ **"I"**
   * **$50.35\%$** thông tin từ động từ **"like"**
   * **$24.83\%$** thông tin bản thể của **"cats"**
2. **Cập nhật Ngữ cảnh Động:** Mọi token trong chuỗi đều trực tiếp attend đến các token còn lại, giúp mô hình hiểu được mối quan hệ cú pháp và ngữ nghĩa tổng thể (ví dụ: mối quan hệ Chủ ngữ - Động từ - Tân ngữ) thay vì xử lý từng từ một cách độc lập.