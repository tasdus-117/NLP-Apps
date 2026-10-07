## 9. Prediction trước experiment

### Prediction 1: Những từ nào gần nhau nhất?
**Tập từ:** `doctor`, `physician`, `hospital`, `banana`, `car`

*   **Prediction:** `doctor` và `physician` sẽ có khoảng cách gần nhau nhất. Nhóm `doctor/physician` cũng sẽ tương đối gần với `hospital`. Hai từ `banana` và `car` sẽ nằm xa nhất so với nhóm trên và xa nhau.
*   **Reason:** Theo nguyên lý phân bố, các từ xuất hiện trong những ngữ cảnh giống nhau sẽ có vector gần nhau. `doctor` và `physician` là từ đồng nghĩa nên có thể thay thế trực tiếp cho nhau trong câu. `hospital` liên quan chặt chẽ về mặt chủ đề. `banana` và `car` không có liên hệ ngữ nghĩa nào đáng kể với nhóm y tế.
*   **Confidence:** Rất cao (95%)

---

### Prediction 2: Nếu context window tăng từ 2 $\rightarrow$ 5, similarity có thay đổi không?

*   **Prediction:** Có thay đổi rõ rệt.
*   **Reason:** Kích thước cửa sổ quyết định loại quan hệ ngữ nghĩa mà mô hình học được. 
    *   Cửa sổ nhỏ ($k=2$) thường bắt được các mối quan hệ ngữ pháp và sự thay thế trực tiếp. 
    *   Cửa sổ lớn ($k=5$ hoặc lớn hơn) sẽ bắt được các mối quan hệ theo chủ đề. Do đó, khi tăng window lên 5, độ tương đồng giữa các từ liên quan về mặt chủ đề (ví dụ: `doctor` và `hospital`) sẽ tăng lên đáng kể so với lúc $k=2$.
*   **Confidence:** Cao (90%)

---

### Prediction 3: Nếu embedding dimension 50 $\rightarrow$ 100 $\rightarrow$ 300 thì chất lượng có chắc chắn tăng không?

*   **Prediction:** Không chắc chắn. Việc tăng số chiều thường làm tăng chất lượng ở giai đoạn đầu, nhưng không tăng vô hạn và phụ thuộc rất lớn vào kích thước tập dữ liệu.
*   **Reason:** Vector nhiều chiều hơn cho phép mô hình biểu diễn nhiều đặc trưng ngữ nghĩa phức tạp hơn. Tuy nhiên, nếu corpus nhỏ, việc dùng số chiều quá lớn sẽ dẫn đến hiện tượng over-parameterized, gây ra nhiễu và làm giảm khả năng tổng quát. Chất lượng thường có điểm bão hòa.
*   **Confidence:** Cao (90%)

---

### Prediction 4: Từ `doctor` và `physician` có chắc chắn gần nhau không nếu corpus chỉ có 100 câu?

*   **Prediction:** Không chắc chắn.
*   **Reason:** Dữ liệu quá nhỏ sẽ dẫn đến vấn đề thưa thớt dữ liệu. Với chỉ 100 câu, khả năng cao là `doctor` và `physician` không xuất hiện đủ nhiều, hoặc ngữ cảnh xung quanh chúng trong tập 100 câu này bị lệch và không đủ đa dạng để mô hình học được phân bố thống kê chung của hai từ. Thuật toán embedding cần hàng triệu đến hàng tỷ token để hội tụ và nhận diện chính xác các từ đồng nghĩa.
*   **Confidence:** Rất cao (95%)

## CBOW và Skip-gram khác biệt cốt lõi ở việc đảo ngược vai trò của dữ liệu đầu vào và nhãn dự đoán cho nhau.

| Mô hình | Input| Target | Bản chất |
| :--- | :--- | :--- | :--- |
| **CBOW** | Các từ ngữ cảnh | Từ trung tâm | Dùng nhiều từ xung quanh để đoán 1 từ ở giữa. |
| **Skip-gram** | Từ trung tâm | Các từ ngữ cảnh | Dùng 1 từ ở giữa để đoán từng từ xung quanh nó. |

Ví dụ với câu `"the cat eats fish"`, xét cửa sổ ngữ cảnh $k=1$ và từ trung tâm là **"cat"**:

*   **CBOW:**
    *   Input: `["the", "eats"]`
    *   Target: `"cat"`
*   **Skip-gram:** 
    *   Input: `"cat"`
    *   Target: Sinh ra 2 mẫu huấn luyện với 2 target độc lập là `"the"` và `"eats"` (cặp `("cat", "the")` và `("cat", "eats")`).

## Bài tập prediction — CBOW vs Skip-gram

**Câu đầu vào:** "the cat eats fish"
**Tokens:** `["the", "cat", "eats", "fish"]`
**Context window:** $k = 1$

Dưới đây là các training examples được tạo ra cho hai kiến trúc CBOW và Skip-gram:

### 1. CBOW
**Mục tiêu:** Dự đoán từ trung tâm dựa trên các từ ngữ cảnh.
**Cấu trúc:** `([Context words], Target word)`

*   Từ trung tâm: **the** $\rightarrow$ Ngữ cảnh: `["cat"]`
    $\Rightarrow$ Training example 1: `(["cat"], "the")`
*   Từ trung tâm: **cat** $\rightarrow$ Ngữ cảnh: `["the", "eats"]`
    $\Rightarrow$ Training example 2: `(["the", "eats"], "cat")`
*   Từ trung tâm: **eats** $\rightarrow$ Ngữ cảnh: `["cat", "fish"]`
    $\Rightarrow$ Training example 3: `(["cat", "fish"], "eats")`
*   Từ trung tâm: **fish** $\rightarrow$ Ngữ cảnh: `["eats"]`
    $\Rightarrow$ Training example 4: `(["eats"], "fish")`

---

### 2. Skip-gram
**Mục tiêu:** Dự đoán các từ ngữ cảnh xung quanh khi biết từ trung tâm.
**Cấu trúc:** `(Target word, Context word)`

*   Từ trung tâm: **the** $\rightarrow$ Ngữ cảnh: `["cat"]`
    $\Rightarrow$ Training example 1: `("the", "cat")`
*   Từ trung tâm: **cat** $\rightarrow$ Ngữ cảnh: `["the", "eats"]`
    $\Rightarrow$ Training example 2: `("cat", "the")`
    $\Rightarrow$ Training example 3: `("cat", "eats")`
*   Từ trung tâm: **eats** $\rightarrow$ Ngữ cảnh: `["cat", "fish"]`
    $\Rightarrow$ Training example 4: `("eats", "cat")`
    $\Rightarrow$ Training example 5: `("eats", "fish")`
*   Từ trung tâm: **fish** $\rightarrow$ Ngữ cảnh: `["eats"]`
    $\Rightarrow$ Training example 6: `("fish", "eats")`


## Inspect embeddings

Giả sử chúng ta huấn luyện (hoặc sử dụng pre-trained model) trên một corpus văn bản tiếng Anh thông thường. Khi truy vấn `model.wv.most_similar("doctor")` trong tập các từ đã cho (`doctor`, `hospital`, `patient`, `disease`, `computer`, `football`, `banana`), kết quả Top các từ gần nhất sẽ có dạng:

**Kết quả Top-5:**
1. `patient` (Tương đồng cao nhất)
2. `hospital` 
3. `disease`
4. ... *(các từ y tế khác nếu có trong vocab như nurse, physician)*
*(Các từ `computer`, `football`, `banana` sẽ nằm ở cuối danh sách với độ tương đồng rất thấp hoặc âm).*

---

### Giải thích: Tại sao những từ này gần "doctor"?

Dựa trên **Giả thuyết phân bố**: *"Bạn có thể biết nghĩa của một từ thông qua những từ đi cùng với nó"*. Các từ `patient`, `hospital`, và `disease` có vector gần với `doctor` không phải vì mô hình tự hiểu nghĩa của chúng, mà vì **chúng chia sẻ chung một tập hợp các từ ngữ cảnh rất lớn trong corpus**.

**Evidence:**

1. **Đồng xuất hiện trong cùng một cửa sổ:**
   Trong các văn bản, `doctor` thường đứng rất gần với `hospital`, `patient`, và `disease`. 
   * Ví dụ trong corpus có các câu: 
     * *"The **doctor** examined the **patient**."*
     * *"The **doctor** works at the local **hospital**."*
     * *"The **doctor** treats the **disease**."*
   Vì chúng thường xuyên lọt vào cùng một context window, mô hình Skip-gram/CBOW sẽ liên tục ép vector của `doctor` phải dự đoán ra `patient`, `hospital` và ngược lại. Quá trình tối ưu hóa này trực tiếp kéo vector của chúng lại gần nhau.

2. **Chia sẻ chung các từ lân cận khác:**
   Ngay cả khi không đứng trực tiếp cạnh nhau, nhóm từ y tế này luôn được bao quanh bởi chung một tệp các động từ/tính từ như: *treat, cure, examine, medicine, health, emergency, bed*. 
   Vì vector của `doctor`, `hospital`, `patient` đều phải được tinh chỉnh để dự đoán ra chung cái đích là chữ *treat* hay *health*, theo toán học, các vector này buộc phải dịch chuyển về cùng một khu vực trong không gian vector.

3. **Ngược lại với `banana`, `computer`, `football`:**
   Các từ này có vector xa `doctor` vì tập ngữ cảnh xung quanh chúng trong corpus hoàn toàn khác biệt. 
   * `banana` đi với context: *eat, peel, yellow, sweet*.
   * `football` đi với context: *kick, goal, match, stadium*.
   * Trong corpus, bạn gần như không bao giờ gặp câu *"The banana treats the disease"* hay *"The football works at the hospital"*. Do không có giao điểm về context words, thuật toán như Word2Vec không có lý do gì để kéo vector của chúng lại gần vector của `doctor`.

## 21. Evaluation — Word similarity

**1. Định lượng và Xếp hạng**
Giả sử chúng ta đo lường cosine similarity trên một mô hình Word2Vec (hoặc GloVe) đã được huấn luyện chuẩ như GoogleNews 300d. Kết quả định lượng thu được sẽ xấp xỉ như bảng sau:

| Hạng | Word Pair | Mối quan hệ | Cosine Similarity |
| :--- | :--- | :--- | :--- |
| 1 | `doctor` – `physician` | Từ đồng nghĩa| $0.85$ |
| 2 | `car` – `automobile` | Từ đồng nghĩa| $0.82$ |
| 3 | `king` – `queen` | Cùng ngữ nghĩa cốt lõi, khác giới tính | $0.75$ |
| 4 | `cat` – `dog` | Cùng loại| $0.70$ |
| 5 | `computer` – `banana` | Không liên quan | $0.05$ |

**2. So sánh với trực giác con người**
Kết quả xếp hạng này hoàn toàn khớp với trực giác ngôn ngữ của con người:
*   Các cặp từ đồng nghĩa (`doctor - physician`, `car - automobile`) có điểm số cao nhất vì chúng có thể thay thế trực tiếp cho nhau trong hầu hết mọi câu ví dụ: *"He drove his [car/automobile] to the [doctor/physician]"*.
*   Các cặp không liên quan (`computer - banana`) có điểm xấp xỉ $0$, phản ánh đúng thực tế chúng không bao giờ xuất hiện trong cùng một bối cảnh.

**3. Những trường hợp bất ngờ**
Kết quả bất ngờ với kết quả của cặp `cat – dog`. Trực giác con người đôi khi coi chó và mèo là hai con vật đối lập hoặc khác biệt rõ ràng. Tuy nhiên, mô hình lại cho điểm similarity rất cao ($0.70$). 
*   **Lý do:** Mô hình embeddings không đo lường sự giống nhau về mặt sinh học. Nó đo lường sự tương đồng về ngữ cảnh phân bố. Cả `cat` và `dog` đều chia sẻ chung một tập khổng lồ các từ lân cận: *pet, feed, bark/meow, tail, fur, sleep*. 
*   Hiện tượng tương tự xảy ra với các **từ trái nghĩa (Antonyms)** như `hot – cold` hay `good – bad`. Chúng thường có similarity rất cao trong vector space vì luôn đi kèm với cùng một nhóm danh từ ví dụ: *weather, water, boy*.