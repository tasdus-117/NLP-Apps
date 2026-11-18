# Xử lý ngôn ngữ tự nhiên và ứng dụng
## LAB 09: Named Entity Recognition and Information Extraction

* **Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng
* **Học kỳ:** Học kỳ 1 - 2026
* **Giảng viên / TA:** Phạm Ngọc Hải
* **Hạn nộp:** 23:59 - 18/11/2026
* **Chủ đề:** From Sequence Classification to Token Classification
* **Đơn vị:** ĐH Khoa học Tự nhiên - ĐHQGHN

---

## 1. Vai trò của LAB 09

**LAB 08 kết thúc với:**
```text
Text
  ↓
Tokenizer
  ↓
Pretrained Transformer
  ↓
[CLS]
  ↓
Sequence Classification
  ↓
One label
```

**LAB 09 thay đổi đầu ra:**
```text
Text
  ↓
Tokenizer
  ↓
Pretrained Transformer
  ↓
Contextual representation
  ↓
One prediction / token
  ↓
Named Entity Recognition
```

**Ví dụ:**
> Barack Obama visited Paris.

Thay vì dự đoán một nhãn chung: `POLITICS`, model phải xác định nhãn cho từng token:

| Token | Barack | Obama | visited | Paris | . |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Label** | `B-PER` | `I-PER` | `O` | `B-LOC` | `O` |

Đây là bước chuyển: **Sequence Classification $\rightarrow$ Token Classification**

---

## 2. Learning Outcomes

Sau LAB 09, sinh viên có thể:
1. Phân biệt sequence classification và token classification.
2. Giải thích bài toán Named Entity Recognition (NER).
3. Hiểu BIO tagging.
4. Chuyển annotation entity thành token labels.
5. Hiểu vấn đề alignment giữa:
   * word,
   * subword token,
   * label.
6. Sử dụng pretrained Transformer cho token classification.
7. Fine-tune model trên CoNLL-2003.
8. Tính và giải thích precision, recall, F1 cho NER.
9. Phân biệt token-level metrics và entity-level metrics.
10. Phân tích lỗi NER.
11. Nhận diện lỗi do:
    * tokenization,
    * context,
    * entity boundary,
    * rare entity,
    * domain shift.
12. Sử dụng NER như một bước đầu của Information Extraction.

---

## 3. Dataset: CONLL-2003

* **Dataset chính:** CONLL-2003 Named Entity Recognition
* **Các entity chính:**
  * `PER`: Person
  * `ORG`: Organization
  * `LOC`: Location
  * `MISC`: Miscellaneous
* **Ngoài ra:**
  * `O`: Outside any entity

**Ví dụ:**
> Barack Obama visited Paris.

Có thể được biểu diễn:

| Token | Barack | Obama | visited | Paris | . |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tag** | `B-PER` | `I-PER` | `O` | `B-LOC` | `O` |

---

## 4. Sequence Classification vs Token Classification

### Sequence Classification (LAB 08)
$$X \rightarrow y$$

**Ví dụ:**
> "This movie is excellent." $\rightarrow$ `POSITIVE`

Một sequence có một label.

### Token Classification (LAB 09)
$$X = (x_1, \dots, x_T)$$
$$Y = (y_1, \dots, y_T)$$

Mỗi token có một label:
* $x_1 \rightarrow y_1$
* $x_2 \rightarrow y_2$
* $\dots$
* $x_T \rightarrow y_T$

**Ví dụ:**

| Token | John | works | at | Google |
| :--- | :---: | :---: | :---: | :---: |
| | $\downarrow$ | $\downarrow$ | $\downarrow$ | $\downarrow$ |
| **Label** | `PER` | `O` | `O` | `ORG` |

---

## 5. BIO Tagging

NER thường sử dụng **BIO scheme**:

| Tag | Meaning |
| :--- | :--- |
| `B-X` | Beginning of entity X |
| `I-X` | Inside entity X |
| `O` | Outside entity |

**Ví dụ:**
> Barack Obama visited New York.

| Token | Barack | Obama | visited | New | York | . |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tag** | `B-PER` | `I-PER` | `O` | `B-LOC` | `I-LOC` | `O` |

---

## 6. Calculation Exercise 1: BIO

**Cho câu:**
> Elon Musk founded SpaceX in California.

**Giả sử:**
* `Elon Musk`: `PER`
* `SpaceX`: `ORG`
* `California`: `LOC`

Sinh viên phải tự tạo danh sách token và BIO labels tương ứng:

| Token | Elon | Musk | founded | SpaceX | in | California | . |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BIO Label** | ? | ? | ? | ? | ? | ? | ? |

---

## 7. Calculation Exercise 2: Entity Boundary

**Cho cụm từ:**
> University of California Berkeley

Giả sử toàn bộ cụm là một organization (`ORG`). Sinh viên phải xác định:

| Token | University | of | California | Berkeley |
| :--- | :---: | :---: | :---: | :---: |
| **BIO Label** | `B-ORG` | `I-ORG` | `I-ORG` | `I-ORG` |

**Sau đó trả lời:**
> Vì sao không thể gán tất cả token là `ORG` nếu muốn biểu diễn ranh giới entity?

*Mục tiêu là hiểu rằng NER không chỉ nhận diện loại entity, mà còn phải nhận diện boundary.*

---

## 8. Subword Tokenization

LAB 08 đã giới thiệu vấn đề trung tâm:
```text
word $\rightarrow$ tokenizer $\rightarrow$ subword tokens
```

LAB 09 phải xử lý hậu quả của việc này.

**Ví dụ minh họa:**
* `Washington` $\rightarrow$ `Washington`
* Nhưng một từ khác có thể trở thành: `playing` $\rightarrow$ `play`, `##ing`

Trong annotation gốc: `playing` có một entity label (ví dụ `B-X`), nhưng tokenizer tạo ra hai token là `play` và `##ing`.
> Vậy label nào được gán cho `##ing`?

Đây là **label alignment problem**.

---

## 9. Label Alignment

**Pipeline:**
```text
Original words
  ↓
Word-level labels
  ↓
Tokenizer
  ↓
Subword tokens
  ↓
Alignment
  ↓
Token-level labels
```

Sinh viên phải hiểu rằng:
* Dataset NER thường annotation theo **word**, trong khi Transformer hoạt động trên **token IDs**.
* Đây là một trong những vấn đề kỹ thuật quan trọng nhất của LAB 09.

---

## 10. Alignment Strategies

Có thể sử dụng một trong các chiến lược:

* **Strategy A (Gán label cho token đầu tiên):**
  * `play` $\rightarrow$ `B-X`
  * `##ing` $\rightarrow$ `IGNORE`
* **Strategy B (Lan truyền label):**
  * `play` $\rightarrow$ `B-X`
  * `##ing` $\rightarrow$ `I-X`
* **Strategy C:**
  * Một số tokenizer/model pipeline có cơ chế alignment riêng.

**Trong lab này, sinh viên phải:**
1. Chọn một strategy;
2. Giải thích;
3. Áp dụng nhất quán;
4. Phân tích ảnh hưởng.

*(Không yêu cầu so sánh mọi strategy nếu thời gian hạn chế).*

---

## 11. Ignore Index

Nếu sử dụng **Strategy A**:
* `play` $\rightarrow$ `B-PER`
* `##ing` $\rightarrow$ `IGNORE`

Các token `IGNORE` không đóng góp vào hàm loss.

**Ví dụ công thức loss:**
$$L = -\sum_{i: y_i \neq \text{IGNORE}} \log P(y_i \mid x)$$

Sinh viên phải hiểu: `IGNORE` không có nghĩa token đó không tồn tại; nó chỉ có nghĩa label của token đó không được sử dụng trực tiếp trong loss.

---

## 12. Prediction Before Experiment

Trước khi chạy model, sinh viên phải dự đoán:

1. **Prediction 1:** Nếu sử dụng pretrained Transformer thay vì một classifier đơn giản dựa trên word features, NER sẽ thay đổi như thế nào?
2. **Prediction 2:** Các entity dài nhiều token sẽ dễ hay khó hơn entity chỉ có một token?
   * *Ví dụ:* `Microsoft` so với `United Nations Security Council`. Giải thích.
3. **Prediction 3:** Entity hiếm xuất hiện trong training data có khả năng gây lỗi nhiều hơn không? Phải đưa ra lý do trước khi chạy experiment.

---

## 13. Baseline

Không bắt đầu ngay bằng Transformer. Xây dựng baseline đơn giản:

```text
Token $\rightarrow$ TF-IDF / token features $\rightarrow$ Logistic Regression
```
hoặc:
```text
Token features $\rightarrow$ Linear classifier $\rightarrow$ NER
```

**Mục tiêu:** Có baseline để biết pretrained Transformer thực sự mang lại giá trị gì.

---

## 14. Transformer Token Classification

**Pipeline:**
```text
Input text
  ↓
Tokenizer
  ↓
Input IDs
  ↓
Pretrained Transformer
  ↓
Hidden representation
  ↓
Linear layer
  ↓
Logits for every token
  ↓
BIO labels
```

Với hidden representation:
$$H \in \mathbb{R}^{T \times d}$$

Classification head:
$$Z = HW + b$$

Trong đó $Z \in \mathbb{R}^{T \times C}$, với:
* $T$: số token;
* $d$: hidden dimension;
* $C$: số NER labels.

Mỗi token có một vector logits: $z_i \in \mathbb{R}^C$.

---

## 15. Calculation Exercise 3: Shape

**Giả sử:**
* $B = 8$
* $T = 128$
* $d = 768$
* $C = 9$

Sinh viên xác định shape của:
* `hidden states`
* `logits`
* `labels`

**Đáp án cần suy ra:**
* $H$: $8 \times 128 \times 768$
* $Z$: $8 \times 128 \times 9$
* $Y$: $8 \times 128$

---

## 16. Core Implementation

Sinh viên triển khai hoặc hoàn thiện pipeline:

```python
tokens = tokenizer(
    words,
    is_split_into_words=True,
    truncation=True,
    padding=True
)
```

Sau đó xây dựng:
```text
word_ids $\rightarrow$ alignment $\rightarrow$ token_labels
```

Phần core không nên chỉ gọi một hàm `pipeline("ner")`. Sinh viên phải nhìn thấy:
* `tokens`
* `input_ids`
* `attention_mask`
* `word_ids`
* `labels`
* `logits`
* `predictions`

---

## 17. Experiment 1: Tokenization Analysis

Chọn tối thiểu **20 câu**. Thống kê:
* `# words`
* `# tokens`
* `# subword splits`

Tính tỷ lệ:
$$R = \frac{\#\text{subword tokens}}{\#\text{words}}$$

So sánh giữa: câu ngắn, câu dài, entity, và non-entity.
* **Câu hỏi:** Những loại từ nào thường bị chia thành nhiều subword?

---

## 18. Experiment 2: Baseline vs Transformer

So sánh hiệu năng trên cùng một tập test set (không chỉ báo cáo accuracy):

| Model | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: |
| Baseline | | | |
| Pretrained Transformer | | | |

---

## 19. Experiment 3: Entity-Level Evaluation

Đây là phần quan trọng.

**Ví dụ:**
* **Gold:** `New` (`B-LOC`), `York` (`I-LOC`)
* **Prediction:** `New` (`B-LOC`), `York` (`O`)

Token `New` đúng, nhưng toàn bộ entity `New York` không được nhận diện hoàn chỉnh. Do đó, **token-level correctness** khác **entity-level correctness**. Sinh viên phải tính hoặc sử dụng metric phù hợp cho NER (như thư viện `seqeval`).

---

## 20. Precision / Recall / F1

Cho entity extraction:
$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$
$$F1 = \frac{2PR}{P + R}$$

Nhưng $TP, FP, FN$ cần được xác định ở **entity level** khi đánh giá entity-level NER.

---

## 21. Experiment 4: Entity Type Analysis

Báo cáo riêng theo từng loại entity:

| Entity | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: |
| `PER` | | | |
| `ORG` | | | |
| `LOC` | | | |
| `MISC` | | | |

**Câu hỏi:** Vì sao performance giữa các loại entity có thể khác nhau?
Sinh viên phải liên hệ với:
* số lượng training examples;
* lexical patterns;
* ambiguity;
* context;
* entity boundary;
* domain.

---

## 22. Experiment 5: Entity Length

Chia entity theo độ dài (1 token, 2 tokens, 3 tokens, 4+ tokens) và tính F1 hoặc entity-level performance:

| Entity length | F1 |
| :--- | :---: |
| 1 | |
| 2 | |
| 3 | |
| 4+ | |

**Mục tiêu:** Kiểm tra prediction về entity boundary bằng dữ liệu thực nghiệm.

---

## 23. Experiment 6: Error Analysis

Sinh viên chọn tối thiểu:
* 5 false positives;
* 5 false negatives;
* 5 boundary errors.

| Text | Gold | Prediction | Error type | Explanation |
| :--- | :---: | :---: | :---: | :--- |
| | | | | |

**Các loại lỗi gợi ý:**
* `wrong entity type`
* `wrong boundary`
* `missing entity`
* `spurious entity`
* `rare entity`
* `ambiguous entity`
* `subword alignment`
* `context-dependent entity`

---

## 24. Context Analysis

NER là một bài toán phụ thuộc mạnh vào context. Cùng một token có thể có vai trò khác nhau trong các context khác nhau.

Sinh viên tìm ít nhất **3 ví dụ** trong test set mà:
```text
same surface form $\rightarrow$ different entity type
```

Sau đó giải thích tại sao contextual representation có lợi thế so với representation tĩnh.

**Điểm nối trực tiếp trong khóa học:**
```text
LAB 03 (static embedding) $\rightarrow$ LAB 08 (contextual representation) $\rightarrow$ LAB 09 (token-level contextual prediction)
```

---

## 25. Application: Information Extraction

Sau khi model NER hoạt động, xây dựng một pipeline đơn giản:
```text
Raw text $\rightarrow$ NER $\rightarrow$ Entity extraction $\rightarrow$ Structured representation
```

**Ví dụ:**
> Barack Obama visited Paris.

Chuyển thành JSON:
```json
{
  "PERSON": ["Barack Obama"],
  "LOCATION": ["Paris"]
}
```

Không yêu cầu xây dựng hệ thống IE phức tạp. Mục tiêu là cho sinh viên thấy NER là bước trung gian để biến *unstructured text* thành *structured data*.

---

## 26. Mini Information Extraction Task

Sinh viên chọn một trong các tùy chọn:
* **Option A (News):** Trích xuất `PERSON`, `ORG`, `LOCATION`.
* **Option B (Academic):** Trích xuất `PERSON`, `ORGANIZATION`, `LOCATION`.
* **Option C (Domain - Mở rộng):** Nếu có corpus phù hợp, trích xuất `DISEASE`, `DRUG`, `PERSON`, `ORGANIZATION`. *(Không bắt buộc nếu dataset annotation chưa đủ)*.

---

## 27. Evaluation của Information Extraction

Không đánh giá IE bằng cảm tính (*"Output nhìn có vẻ đúng"*).
* Phải lấy một tập nhỏ, ví dụ **30 sentences**, và tạo gold annotations.
* Sau đó tính **Precision, Recall, F1** ở entity level.

---

## 28. AI Usage

* **Được phép dùng AI cho:**
  * tokenizer documentation;
  * API lookup;
  * debugging;
  * visualization;
  * boilerplate;
  * giải thích runtime error.
* **Không được giao AI làm thay:**
  * BIO labeling exercise;
  * label alignment reasoning;
  * prediction;
  * metric interpretation;
  * error analysis;
  * reflection.

Nếu AI hỗ trợ code, phải ghi rõ:
```text
AI-assisted code:
- What was generated: ...
- What was modified: ...
- What was verified: ...
```

---

## 29. Deliverables

Cấu trúc thư mục nộp bài:

```text
lab09/
├── README.md
├── calculations.md
├── prediction.md
├── baseline_ner.py
├── tokenizer_alignment.ipynb
├── ner_finetuning.ipynb
├── results.csv
├── entity_metrics.csv
├── entity_length_analysis.png
├── confusion_matrix.png
├── error_analysis.md
├── information_extraction.py
└── reflection.md
```

---

## 30. Rubric

| Component | Points |
| :--- | :---: |
| Theory & calculation | 10 |
| Prediction | 10 |
| BIO & label alignment | 15 |
| Baseline implementation | 10 |
| Transformer NER implementation | 15 |
| Tokenization experiment | 10 |
| Entity-level evaluation | 10 |
| Error analysis | 10 |
| Information Extraction application | 5 |
| Reflection | 5 |
| **Total** | **100** |

---

## 31. Individual Learning Check

Mỗi sinh viên trả lời ngắn 10 câu hỏi sau:
1. Sequence classification khác token classification như thế nào?
2. BIO tagging dùng để làm gì?
3. `B-PER` khác `I-PER` như thế nào?
4. Tại sao tokenizer gây ra label alignment problem?
5. `IGNORE` trong loss có ý nghĩa gì?
6. Tại sao NER không nên chỉ đánh giá bằng token accuracy?
7. Entity-level F1 khác token-level F1 như thế nào?
8. Vì sao entity boundary khó?
9. Vì sao contextual representation hữu ích cho NER?
10. NER có thể được sử dụng như thế nào trong Information Extraction?

---

## 32. Reflection

Viết **300–500 từ** trả lời câu hỏi:
> Tại sao cùng một pretrained Transformer có thể được sử dụng cho cả sequence classification và token classification chỉ bằng cách thay đổi phần prediction head?

Phải giải thích được luồng xử lý:
* `Pretrained Transformer` $\rightarrow$ `contextual representation`
* Nhánh Sequence: `[CLS]` $\rightarrow$ `classification` $\rightarrow$ `sequence label`
* Nhánh Token: `every token` $\rightarrow$ `token classifier` $\rightarrow$ `NER labels`

---

## 33. Bridge to LAB 10

LAB 09 xử lý `Text` $\rightarrow$ `Entities`. Nhưng nhiều hệ thống NLP cần trả lời câu hỏi: *Một đoạn văn bản chứa thông tin gì tại một vị trí cụ thể?*

**Ví dụ:**
* **Context:** Paris is the capital of France.
* **Question:** What is the capital of France?
* **Answer:** Paris

Đây không còn là sequence classification hay chỉ là token classification, mà là **Question Answering (Extractive QA)** sử dụng **Span Prediction** (dự đoán vị trí `start` và `end`):

```text
                Pretrained Transformer
               /          |           \
              ↓           ↓            ↓
       Sequence task  Token task   Span task
              ↓           ↓            ↓
       Classification    NER          QA
              |           |            |
           LAB 08      LAB 09       LAB 10
```

**Điểm nhấn mạnh trong khóa học:** Sinh viên bắt đầu nhìn pretrained Transformer như một **backbone** tạo contextual representation, còn downstream task được quyết định bởi **prediction head** và cách định nghĩa output/label.