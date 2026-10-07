## Reflection — Từ Word2Vec đến Transformer

### 1. Bảng so sánh các phương pháp biểu diễn từ

| Representation | Context-dependent? | Sparse/Dense | Một từ có nhiều vector? |
| :--- | :--- | :--- | :--- |
| **TF-IDF** | Không | Sparse| Không |
| **Co-occurrence** | Không | Sparse | Không |
| **Word2Vec** | Không | Dense | Không |
| **Contextual embedding** | Có | Dense | Có |

---

### 2. Tại sao từ "bank" cần contextual representation?

Từ **"bank"** là một từ đa nghĩa điển hình trong ngôn ngữ tự nhiên. Ý nghĩa của nó thay đổi hoàn toàn tùy thuộc vào các từ ngữ xung quanh trong câu:
*   *Câu 1:* "He deposited money into his account at the **bank**." (Ngân hàng - Tài chính)
*   *Câu 2:* "They sat together on the grassy **bank** of the river." (Bờ sông - Tự nhiên)

**Hạn chế của mô hình tĩnh như Word2Vec:**
Trong các mô hình tĩnh, mỗi từ vựng chỉ được gán **duy nhất một vector cố định** trong toàn bộ không gian không phụ thuộc vào ngữ cảnh. 
* Nếu mô hình cố gắng biểu diễn cả hai nghĩa "ngân hàng" và "bờ sông" bằng một vector duy nhất, tọa độ của vector đó sẽ bị kéo về điểm trung gian giữa hai miền ngữ nghĩa. 
* Hậu quả là vector không phản ánh chính xác nghĩa nào cả, gây ra sự nhầm lẫn lớn trong các tác vụ xử lý ngôn ngữ phức tạp.

**Giải pháp từ Contextual Embeddings (Transformer / BERT):**
Các mô hình hiện đại dựa trên kiến trúc Transformer tạo ra biểu diễn động. Vector của từ **"bank"** sẽ **được tính toán trực tiếp dựa trên toàn bộ các từ xung quanh nó** thông qua cơ chế Self-Attention. 
* Khi đi kèm với các từ như *money, account*, vector của **"bank"** sẽ dịch chuyển sâu vào không gian tài chính.
* Khi đi kèm với các từ như *river, grassy*, vector của **"bank"** sẽ dịch chuyển sang không gian địa lý tự nhiên. 

Do đó, **contextual representation** là yếu tố cốt lõi giúp mô hình hiểu và phân biệt chính xác các tầng nghĩa khác nhau của từ vựng theo từng hoàn cảnh cụ thể.

### 3. AI đã được sử dụng ở những phần nào và đóng góp cụ thể là gì?
AI đã hỗ trợ trong việc viết code và sinh ra các câu trả lời dựa trên định hướng và ý tưởng ban đầu của em. Sau đó, em kiểm tra, tinh chỉnh và viết lại các nội dung này để đảm bảo câu từ mạch lạc, dễ hiểu và phản ánh đúng quá trình tư duy cá nhân.