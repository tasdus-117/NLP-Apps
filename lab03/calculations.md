## Bài 1 — Co-occurrence

Với cửa sổ ngữ cảnh $k = 1$, chúng ta xét 1 từ lân cận bên trái và 1 từ lân cận bên phải của từ mục tiêu trong mỗi câu. 

Tập từ vựng xuất hiện trong toàn bộ dữ liệu bao gồm 8 từ: **the, cat, dog, eats, likes, fish, milk, meat**.

Thống kê các từ đồng xuất hiện cho từng từ mục tiêu:
*   **cat**: Xuất hiện trong "the **cat** eats" và "the **cat** likes". Các từ lân cận: *the* (2 lần), *eats* (1 lần), *likes* (1 lần).
*   **dog**: Xuất hiện trong "the **dog** eats" và "the **dog** likes". Các từ lân cận: *the* (2 lần), *eats* (1 lần), *likes* (1 lần).
*   **eats**: Xuất hiện trong "cat **eats** fish" và "dog **eats** fish". Các từ lân cận: *cat* (1 lần), *dog* (1 lần), *fish* (2 lần).
*   **likes**: Xuất hiện trong "cat **likes** milk" và "dog **likes** meat". Các từ lân cận: *cat* (1 lần), *dog* (1 lần), *milk* (1 lần), *meat* (1 lần).

Bảng biểu diễn word-context vector cho 4 từ mục tiêu:

| Target Word | the | cat | dog | eats | likes | fish | milk | meat |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **cat** | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| **dog** | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| **eats** | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 0 |
| **likes** | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |

---

## Bài 2 — Similarity

Cho hai vector $x = [1, 2, 1]$ và $y = [2, 4, 2]$.

**1. Tính tích vô hướng:**
$$x \cdot y = (1 \times 2) + (2 \times 4) + (1 \times 2) = 2 + 8 + 2 = 12$$

**2. Tính độ lớn của từng vector:**
$$\|x\| = \sqrt{1^2 + 2^2 + 1^2} = \sqrt{1 + 4 + 1} = \sqrt{6}$$
$$\|y\| = \sqrt{2^2 + 4^2 + 2^2} = \sqrt{4 + 16 + 4} = \sqrt{24} = 2\sqrt{6}$$

**3. Tính Cosine Similarity:**
$$\cos(x, y) = \frac{x \cdot y}{\|x\| \|y\|} = \frac{12}{\sqrt{6} \times 2\sqrt{6}} = \frac{12}{2 \times 6} = 1$$

**Giải thích:**
Hai vector có độ lớn khác nhau nhưng cosine similarity bằng $1$ có nghĩa là chúng cùng hướng hay góc giữa hai vector bằng $0^{\circ}$. Trong trường hợp này, vector $y$ là một phép nhân vô hướng của vector $x$ ($y = 2x$). 
Trong NLP, điều này mang ý nghĩa rằng phân bố ngữ cảnh của hai từ này giống hệt nhau về mặt tỉ lệ tương đối, chúng chỉ khác nhau về tần suất xuất hiện tuyệt đối.

---

## Bài 3 — So sánh semantic similarity

**Dự đoán trước khi tính:**
*Doctor* và *Physician* có chung trường ngữ nghĩa, do đó dự đoán $\cos(\text{doctor}, \text{physician})$ sẽ tiến gần tới $1$. Ngược lại, *doctor* và *banana* thuộc hai chủ đề khác biệt, dự đoán $\cos(\text{doctor}, \text{banana})$ sẽ xấp xỉ $0$ hoặc âm.

**Tính toán chi tiết:**

**1. Tính $\cos(\text{doctor}, \text{physician})$:**
*   Tích vô hướng: $(0.8 \times 0.7) + (0.1 \times 0.2) + (0.7 \times 0.8) = 0.56 + 0.02 + 0.56 = 1.14$
*   Độ lớn $\|v_{\text{doctor}}\| = \sqrt{0.8^2 + 0.1^2 + 0.7^2} = \sqrt{1.14}$
*   Độ lớn $\|v_{\text{physician}}\| = \sqrt{0.7^2 + 0.2^2 + 0.8^2} = \sqrt{1.17}$

$$\cos(\text{doctor}, \text{physician}) = \frac{1.14}{\sqrt{1.14} \times \sqrt{1.17}} \approx 0.987$$

**2. Tính $\cos(\text{doctor}, \text{banana})$:**
*   Tích vô hướng: $(0.8 \times -0.2) + (0.1 \times 0.9) + (0.7 \times -0.1) = -0.16 + 0.09 - 0.07 = -0.14$
*   Độ lớn $\|v_{\text{banana}}\| = \sqrt{(-0.2)^2 + 0.9^2 + (-0.1)^2} = \sqrt{0.86}$

$$\cos(\text{doctor}, \text{banana}) = \frac{-0.14}{\sqrt{1.14} \times \sqrt{0.86}} \approx -0.141$$

*Kết luận: Kết quả hoàn toàn khớp với dự đoán.*

---

## Bài 4 — Sparse vs dense

**1. Biểu diễn nào sparse?**
Biểu diễn word-context representation với $10,000$ chiều nhưng chỉ có $30$ non-zero entries là biểu diễn **sparse**.

**2. Biểu diễn nào dense?**
Biểu diễn embedding với $300$ chiều và hầu hết các thành phần đều khác $0$ là biểu diễn **dense** .

**3. Vì sao dense representation có thể thuận lợi hơn cho semantic similarity?**
*   **Bắt được quan hệ ngữ nghĩa tiềm ẩn:** Vector dense được học thông qua các mô hình giúp mã hóa các đặc trưng ngữ nghĩa vào một không gian liên tục. Các từ đồng nghĩa dù hiếm khi chia sẻ cùng một ngữ cảnh chính xác vẫn sẽ có vector dense gần nhau.
*   **Hiệu suất:** Giảm số chiều từ $10,000$ xuống $300$ giúp tiết kiệm bộ nhớ, giảm bớt chi phí tính toán và tránh hiện tượng curse of dimensionality trong các mô hình học máy.

**4. Dense representation có chắc chắn tốt hơn trong mọi bài toán không?**
**Không chắc chắn.** Dense không phải là một khẩu hiệu luôn tốt hơn. 
*   Biểu diễn Sparse vượt trội trong các bài toán **Exact Keyword Matching**, ví dụ như khi tìm kiếm một danh từ riêng, một mã số, hoặc tên một loại bệnh hiếm gặp trong hệ thống Information Retrieval. Trong các trường hợp này, dense embedding có thể trả về các kết quả tương tự về ngữ nghĩa nhưng lại sai lệch hoàn toàn về mặt thông tin cụ thể.
*   Ngoài ra, Sparse vector có **tính diễn giải** cao hơn rất nhiều, trong khi Dense vector là một hộp đen với các con số thực khó diễn giải trực tiếp.

## Bài tập tính analogy

**Tính toán:**
Thực hiện phép trừ giữa hai vector $\vec{king}$ và $\vec{man}$:
$$[8, 2, 7] - [5, 1, 5] = [8 - 5, 2 - 1, 7 - 5] = [3, 1, 2]$$

Tiếp tục cộng kết quả trên với vector $\vec{woman}$:
$$[3, 1, 2] + [5, 3, 5] = [3 + 5, 1 + 3, 2 + 5] = [8, 4, 7]$$

Kết quả cuối cùng của phép toán $\vec{king} - \vec{man} + \vec{woman}$ là vector **$[8, 4, 7]$**.

**Giải thích ý nghĩa:**
Vector mới $[8, 4, 7]$ đại diện cho khái niệm nữ hoàng. Phép toán này biểu hiện mối quan hệ tương tự về giới tính và vai trò xã hội. 

Phép trừ giữa vua và đàn ông đã loại bỏ các giá trị biểu diễn đặc tính nam giới, để lại một vector tịnh tiến mang đặc trưng cốt lõi của sự trị vì hoặc vị thế hoàng gia. Khi cộng đặc tính hoàng gia này vào vector phụ nữ, chúng ta dịch chuyển trong không gian vector đến một tọa độ hội tụ cả hai tính chất: nữ giới và sự trị vì. Tọa độ kết quả này sẽ nằm rất gần với vị trí của từ nữ hoàng trong tập dữ liệu.