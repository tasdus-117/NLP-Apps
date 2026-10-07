## Error analysis

Trong phân tích lỗi này, chúng ta sẽ khảo sát kết quả của một mô hình Word2Vec (hoặc GloVe) được huấn luyện trên một tập ngữ liệu tổng hợp. Chúng ta sẽ đánh giá 3 trường hợp mô hình trả về kết quả đúng với trực giác, và 3 trường hợp mô hình trả về kết quả sai hoặc gây bất ngờ, từ đó phân tích nguyên nhân dựa trên đặc tính của Word Embeddings.

### A. 3 Trường hợp Similarity Đúng

**1. Word pair: `doctor` $\rightarrow$ `physician`**
*   **Observed:** Cosine similarity rất cao ($> 0.85$).
*   **Expected:** Similarity cao.
*   **Possible explanation:** Mô hình bắt được quan hệ thay thế. Hai từ này đóng cùng một vai trò ngữ pháp và ngữ nghĩa, nên có tập hợp từ lân cận gần như giống hệt nhau.
*   **Evidence from corpus:** Cả hai đều xuất hiện trong bối cảnh: *"The ___ examined the patient"* hoặc *"He works as a ___ at the local hospital."*

**2. Word pair: `hospital` $\rightarrow$ `clinic`**
*   **Observed:** Cosine similarity cao ($> 0.75$).
*   **Expected:** Similarity cao.
*   **Possible explanation:** Hai từ này thường xuất hiện với các động từ chỉ sự di chuyển, khám chữa bệnh và chia sẻ chung một nhóm danh từ ngữ cảnh.
*   **Evidence from corpus:** Các câu như *"She was rushed to the ___"* hay *"The ___ provides medical services."*

**3. Word pair: `car` $\rightarrow$ `vehicle`**
*   **Observed:** Cosine similarity cao ($> 0.70$).
*   **Expected:** Similarity cao.
*   **Possible explanation:** Dù `vehicle` mang nghĩa rộng hơn, nhưng trong cách sử dụng ngôn ngữ thông thường, chúng được dùng trong các cấu trúc hành động tương tự nhau.
*   **Evidence from corpus:** *"Drive the ___"*, *"park the ___"*, *"motor ___"*.

---

### B. 3 Trường hợp Similarity Sai

**1. Word pair: `good` $\rightarrow$ `bad`**
*   **Observed:** Cosine similarity rất cao ($> 0.75$).
*   **Expected:** Similarity thấp.
*   **Possible explanation:** Vấn đề cốt lõi của **Context window** trong distributional semantics. Khác với tư duy con người, Word2Vec chỉ đếm bối cảnh xung quanh. Các từ trái nghĩa thường xuất hiện trong những bối cảnh cú pháp và chủ đề giống hệt nhau.
*   **Evidence from corpus:** Trong corpus, cấu trúc *"The movie was really ___"* hoặc *"It is a ___ idea"* đều được điền bằng cả `good` và `bad` với tần suất rất cao.

**2. Word pair: `apple` $\rightarrow$ `microsoft`**
*   **Observed:** `apple` có similarity rất cao với `microsoft`, `google`, nhưng lại có similarity rất thấp với `banana`, `orange`.
*   **Expected:** `apple` phải gần với các loại trái cây khác.
*   **Possible explanation:** **Polysemy** và **Domain bias** (Độ lệch miền dữ liệu). Từ `apple` vừa mang nghĩa quả táo, vừa là tên công ty. Nếu corpus được thu thập chủ yếu từ tin tức tài chính, công nghệ, nghĩa "công ty" sẽ xuất hiện với tần suất áp đảo, làm lu mờ hoàn toàn nghĩa "trái cây".
*   **Evidence from corpus:** Hàng triệu câu có dạng *"___ announced a new smartphone"*, *"Stock prices of ___ dropped today"*. Sự chênh lệch **frequency** giữa hai nghĩa khiến vector bị kéo hoàn toàn về cụm công nghệ.

**3. Word pair: `doctor` $\rightarrow$ `disease`**
*   **Observed:** Cosine similarity cao ($> 0.65$).
*   **Expected:** Similarity thấp, vì đây không phải là hai khái niệm tương đồng.
*   **Possible explanation:** Việc chọn **Context window** quá lớn (ví dụ $k = 10$) khiến mô hình học các mối quan hệ theo chủ đề thay vì quan hệ đồng nghĩa. Ngoài ra, nếu huấn luyện trên một **corpus nhỏ** với các văn bản y khoa lặp đi lặp lại, hai từ này sẽ trở thành ngữ cảnh thường xuyên của nhau.
*   **Evidence from corpus:** Các câu như *"The **doctor** successfully treated the rare **disease**."* Nếu window lớn, thuật toán sẽ liên tục ánh xạ chúng với nhau, gây ra hiện tượng gom cụm chủ đề thay vì phân loại ý nghĩa từ vựng.
