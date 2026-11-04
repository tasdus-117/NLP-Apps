## 6. Bài tập tính toán — Một attention head

Chọn bộ ma trận mẫu:

$$
X=
\begin{bmatrix}
1&0&0&1&0\\
0&1&0&0&1\\
0&0&1&1&1
\end{bmatrix}
\in\mathbb{R}^{3\times5}.
$$

$$
W_Q=W_K=
\begin{bmatrix}
1&0\\
0&1\\
1&1\\
0&0\\
0&0
\end{bmatrix},
\qquad
W_V=
\begin{bmatrix}
1&2\\
3&0\\
0&1\\
0&0\\
0&0
\end{bmatrix}
\in\mathbb{R}^{5\times2}.
$$

Với $d_k=2$, ta có:

$$
Q=XW_Q=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix},
\qquad
K=XW_K=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix},
\qquad
V=XW_V=
\begin{bmatrix}
1&2\\
3&0\\
0&1
\end{bmatrix}.
$$

$$
S=\frac{QK^T}{\sqrt{2}}
=\frac{1}{\sqrt{2}}
\begin{bmatrix}
1&0&1\\
0&1&1\\
1&1&2
\end{bmatrix}
\approx
\begin{bmatrix}
0.7071&0&0.7071\\
0&0.7071&0.7071\\
0.7071&0.7071&1.4142
\end{bmatrix}.
$$

Áp dụng softmax theo từng hàng:

$$
A=\operatorname{softmax}(S)
\approx
\begin{bmatrix}
0.4011&0.1978&0.4011\\
0.1978&0.4011&0.4011\\
0.2483&0.2483&0.5035
\end{bmatrix}.
$$

$$
Z=AV
\approx
\begin{bmatrix}
0.9944&1.2033\\
1.4011&0.7967\\
0.9930&1.0000
\end{bmatrix}
\in\mathbb{R}^{3\times2}.
$$

Một attention head là phép attention của LAB 06, trong đó
$Q,K,V$ được tạo từ $X$ bằng các ma trận chiếu $W_Q,W_K,W_V$.

## 7. Bài tập — Multi-head

Với $T=3$:

$$
\operatorname{head}_1,\operatorname{head}_2
\in\mathbb{R}^{3\times2}.
$$

$$
H=\operatorname{Concat}(\operatorname{head}_1,\operatorname{head}_2)
\in\mathbb{R}^{3\times4}.
$$

**Vì sao cần concatenation trước khi áp dụng $W_O$?**

Concatenation ghép các đặc trưng mà từng head học được thành một
vector chung tại mỗi token. Sau đó, $W_O$ học cách kết hợp thông tin
từ các head và chiếu về chiều đầu ra mong muốn.

Với chiều đầu vào bằng 5:

$$
W_O\in\mathbb{R}^{4\times5},
\qquad
HW_O\in\mathbb{R}^{3\times5}.
$$

## 10. Bài tập tính positional encoding

Với $d=4$, positional encoding được tính theo công thức:

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right),
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right).
$$

Với $i=0,1$, ta được:

$$
PE(pos)=
\left[
\sin(pos),\;
\cos(pos),\;
\sin\left(\frac{pos}{100}\right),\;
\cos\left(\frac{pos}{100}\right)
\right].
$$

Các góc được tính theo radian.

### Vị trí pos = 0

$$
PE(0)=[0,\;1,\;0,\;1].
$$

### Vị trí pos = 1

$$
PE(1)
=[\sin(1),\;\cos(1),\;\sin(0.01),\;\cos(0.01)]
$$

$$
PE(1)\approx[0.841471,\;0.540302,\;0.010000,\;0.999950].
$$

### Vị trí pos = 2

$$
PE(2)
=[\sin(2),\;\cos(2),\;\sin(0.02),\;\cos(0.02)]
$$

$$
PE(2)\approx[0.909297,\;-0.416147,\;0.019999,\;0.999800].
$$

### Kết quả

$$
PE\approx
\begin{bmatrix}
0&1&0&1\\
0.841471&0.540302&0.010000&0.999950\\
0.909297&-0.416147&0.019999&0.999800
\end{bmatrix}.
$$

Ba vị trí có ba vector positional representation khác nhau,
giúp bổ sung thông tin vị trí của token cho Transformer.

## 14. Bài tập tính FFN

Cho:

$$
x=\begin{bmatrix}1\\2\end{bmatrix},
\quad
W_1=\begin{bmatrix}1&0\\0&1\end{bmatrix},
\quad
b_1=\begin{bmatrix}0\\-1\end{bmatrix}.
$$


$$
W_2=\begin{bmatrix}2&-1\\3&1\end{bmatrix},
\qquad
b_2=\begin{bmatrix}1\\-2\end{bmatrix}.
$$

Ta có:

$$
h=\operatorname{ReLU}(W_1x+b_1)
=\operatorname{ReLU}
\left(
\begin{bmatrix}1\\2\end{bmatrix}
+
\begin{bmatrix}0\\-1\end{bmatrix}
\right)
=\begin{bmatrix}1\\1\end{bmatrix}.
$$

Sau đó:

$$
y=W_2h+b_2
=
\begin{bmatrix}2&-1\\3&1\end{bmatrix}
\begin{bmatrix}1\\1\end{bmatrix}
+
\begin{bmatrix}1\\-2\end{bmatrix}
=
\begin{bmatrix}2\\2\end{bmatrix}.
$$

**Kết quả:**

$$
\boxed{h=\begin{bmatrix}1\\1\end{bmatrix},
\qquad y=\begin{bmatrix}2\\2\end{bmatrix}.}
$$

**Nhận xét:** FFN biến đổi đặc trưng độc lập tại từng token bằng
các lớp tuyến tính và hàm kích hoạt, dùng chung trọng số giữa
các vị trí. FFN không tính trọng số tương tác giữa các token
như attention.

## 15. Residual connection

Nếu chỉ dùng $F(x)$, đầu ra hoàn toàn phụ thuộc vào phép biến đổi
của sublayer, nên thông tin đầu vào có thể bị mất hoặc thay đổi.

Khi dùng $y=x+F(x)$, residual connection tạo đường truyền trực tiếp
để giữ thông tin từ $x$. Sublayer chỉ cần học phần bổ sung hoặc
điều chỉnh $F(x)$.

Đường truyền này cũng giúp gradient lan truyền thuận lợi hơn,
hỗ trợ tối ưu hóa khi huấn luyện mạng sâu.