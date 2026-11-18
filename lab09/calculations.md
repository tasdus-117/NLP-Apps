<a id="lab09-section-4"></a>
## 4. Sequence Classification vs Token Classification


Sequence classification ánh xạ cả chuỗi X sang một nhãn y, ví dụ sentiment. Backbone tạo H có B×T×d; head dùng [CLS]/pooled representation B×d để tạo logits B×C_seq và labels B.
Token classification tạo một nhãn cho mỗi vị trí: logits B×T×C_token, labels B×T. NER cần nhãn loại và ranh giới, không chỉ phân loại chủ đề câu. Words annotation và subword tokens có thể khác số lượng, nên cần alignment.

<a id="lab09-section-5"></a>
## 5. BIO Tagging

B đánh dấu bắt đầu, I là tiếp diễn; nhãn ORG đơn thuần không phân biệt một entity dài với nhiều ORG kề nhau. Hai B-ORG liên tiếp là hai entities, không được gộp chỉ vì cùng type. IOB2 yêu cầu I-X theo sau B-X/I-X của cùng entity; dataset thực tế được kiểm tra riêng trong notebook.

<a id="lab09-section-6"></a>
## 6. Calculation Exercise 1: BIO


Tách dấu chấm thành token riêng, theo giả định gold của đề:

| Word | Elon | Musk | founded | SpaceX | in | California | . |
|---|---|---|---|---|---|---|---|
| BIO | B-PER | I-PER | O | B-ORG | O | B-LOC | O |

Elon mở entity PER, Musk tiếp nối cùng entity; SpaceX và California mở các entity mới khác loại. Các words còn lại ngoài entity.

<a id="lab09-section-7"></a>
## 7. Calculation Exercise 2: Entity Boundary


| Word | University | of | California | Berkeley |
|---|---|---|---|---|
| BIO | B-ORG | I-ORG | I-ORG | I-ORG |

Đề giả định toàn bộ cụm là một ORG, nên California không tách thành LOC ở đây.

<a id="lab09-section-11"></a>
## 11. Ignore Index

Loss cross entropy trung bình trên tập V các vị trí có label khác -100:

$$
L=-\frac{1}{|V|}\sum_{(b,t)\in V}\log\operatorname{softmax}(Z_{b,t,:})_{Y_{b,t}}.
$$

-100 loại target khỏi loss, không xóa token khỏi contextual attention. Attention mask=0 là padding, khác với continuation subword thường có mask=1 nhưng label=-100.

<a id="lab09-section-15"></a>
## 15. Calculation Exercise 3: Shape


Cho B=8, T=128, d=768, C=9:

| Tensor | Shape | Ý nghĩa các trục |
|---|---|---|
| input_ids, attention_mask | (8,128) | câu trong batch; vị trí subword/padding |
| H hidden states | (8,128,768) | batch; vị trí; 768 contextual features |
| W, b của token head | (768,9), (9,) | chiếu hidden vector sang 9 BIO classes |
| Z=HW+b logits | (8,128,9) | batch; vị trí; điểm chưa softmax cho từng class |
| Y labels | (8,128) | một class ID hoặc -100 cho mỗi vị trí |

Đây là ví dụ shape của đề, không phải batch size của thực nghiệm (batch32, padding động). C=1+2×4=9 gồm O và B/I của PER/ORG/LOC/MISC. Labels không phải one-hot (8,128,9). Logits argmax theo trục cuối tạo class IDs (8,128).

<a id="lab09-section-19"></a>
## 19. Experiment 3: Entity-Level Evaluation

Gold New York/LOC có span [0,2), prediction New/LOC có span [0,1): không phải exact match. TP=0, FP=1, FN=1 cho entity đó, dù token New có nhãn đúng. Token accuracy có thể cao do nhiều O; không thay thế entity F1.
<a id="lab09-section-20"></a>
## 20. Precision / Recall / F1


Entity là tuple (sentence ID, start word inclusive, end word exclusive, type). TP chỉ khi tuple khớp hoàn toàn. FP là predicted tuple không trong gold, FN là gold tuple không trong predicted.

$$
P=\frac{TP}{TP+FP},\quad R=\frac{TP}{TP+FN},\quad F_1=\frac{2PR}{P+R}=\frac{2TP}{2TP+FP+FN}.
$$

Ví dụ **minh họa tính toán, không phải số liệu thực nghiệm**: gold có Elon Musk/PER, SpaceX/ORG, California/LOC; dự đoán đúng Elon Musk/PER và California/LOC, bỏ SpaceX, thêm founded/ORG. TP=2, FP=1, FN=1; P=R=F1=2/3≈0,6667. Sai type với đúng boundary cũng đóng góp một FP và một FN. Mẫu số 0 được code trả 0 và công bố support, không ngầm coi là thành công.

