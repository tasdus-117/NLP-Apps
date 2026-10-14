# **Bài 1 — Binary classification. Tính giá trị $z$**

Công thức tính:
$$z = w^T x + b$$

Thay các giá trị $x = [1, 2, 0]$, $w = [0.5, -0.2, 0.4]$ và $b = 0.1$:
*   $w^T x = (0.5 \times 1) + (-0.2 \times 2) + (0.4 \times 0)$
*   $w^T x = 0.5 - 0.4 + 0 = 0.1$
*   $z = 0.1 + 0.1 = \mathbf{0.2}$

Công thức hàm Sigmoid:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Thay $z = 0.2$ vào công thức:
*   $\sigma(0.2) = \frac{1}{1 + e^{-0.2}}$
*   $e^{-0.2} \approx 0.81873$
*   $\sigma(0.2) = \frac{1}{1 + 0.81873} = \frac{1}{1.81873} \approx \mathbf{0.5498}$

Vậy xác suất cần tìm là:
$$P(y = 1\vert{}x) = \sigma(z) \approx 0.5498 \text{ (hoặc } 54.98\% \text{)}$$

# **Bài 2 — Prediction threshold**

Quy tắc quyết định: Nếu $P(y = 1|x) \ge \tau$, model dự đoán **Class 1**; ngược lại dự đoán **Class 0**. Với xác suất cho trước $P(y = 1|x) = 0.72$:

*   **Với $\tau = 0.5$:** Vì $0.72 \ge 0.5$, model dự đoán **Class 1**.
*   **Với $\tau = 0.8$:** Vì $0.72 < 0.8$, model dự đoán **Class 0**.


Việc thay đổi ngưỡng phân loại ($\tau$) tạo ra sự thay đổi giữa **Precision** và **Recall**:

*   **Khi tăng Threshold:**
    *   Model trở nên khó tính hơn trong việc gán nhãn dương (Class 1). Số lượng dự đoán dương giảm đi.
    *   **Precision tăng:** Vì các mẫu được dự đoán là dương có độ tin cậy cao hơn, làm giảm số lượng Dương giả.
    *   **Recall giảm:** Vì model bỏ sót nhiều mẫu thực tế là dương nhưng xác suất thấp hơn ngưỡng mới, làm tăng số lượng Âm giả.

*   **Khi giảm Threshold:**
    *   Model dễ dãi hơn trong việc dự đoán nhãn dương. Số lượng dự đoán dương tăng lên.
    *   **Precision giảm:** Do xuất hiện nhiều Dương giả hơn.
    *   **Recall tăng:** Do bắt được nhiều mẫu dương hơn, làm giảm số lượng Âm giả.


# **Bài 3 — Confusion matrix**

Từ đề bài, ta xác định các thành phần của Confusion Matrix:
*   $\text{True Positive} = 80$
*   $\text{False Negative} = 20$
*   $\text{False Positive} = 10$
*   $\text{True Negative} = 90$

Các công thức và kết quả tính toán:

*   **Precision:**
    $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{80}{80 + 10} = \frac{80}{90} \approx \mathbf{0.8889} \text{ (hoặc } 88.89\% \text{)}$$

*   **Recall:**
    $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{80}{80 + 20} = \frac{80}{100} = \mathbf{0.80} \text{ (hoặc } 80\% \text{)}$$

*   **F1-score:**
    $$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times 0.8889 \times 0.80}{0.8889 + 0.80} \approx \frac{1.4222}{1.6889} \approx \mathbf{0.8421} \text{ (hoặc } 84.21\% \text{)}$$


Trong bài toán phát hiện spam:
*   **Positive:** Email là Thư rác.
*   **Negative :** Email là Thư bình thường.

Do đó:

*   **False Positive (FP = 10):** Model dự đoán là **Spam** nhưng thực tế đây là **email quan trọng của người dùng**.
    *   *Ý nghĩa thực tế:* Email công việc, thông báo quan trọng bị đưa nhầm vào mục Thư rác. Điều này gây phiền toái, khiến người dùng có thể bỏ lỡ thông tin quan trọng.
*   **False Negative (FN = 20):** Model dự đoán là **Email bình thường** nhưng thực tế đây là **Thư rác**.
    *   *Ý nghĩa thực tế:* Thư rác lọt qua bộ lọc và hiển thị ngay trong Hộp thư đến của người dùng. Điều này gây khó chịu, làm mất thời gian dọn dẹp thư rác hoặc tiềm ẩn nguy cơ lừa đảo, mã độc.


# **Bài 4 — Accuracy trap**

**Không, model này hoàn toàn không tốt** mặc dù đạt độ chính xác rất cao lên tới **95%**. 

Lý do là vì đây là một tập dữ liệu mất cân bằng nặng, với 95% là mẫu negative và chỉ có 5% là mẫu positive. Model trong trường hợp này chỉ là một **naive classifier**: nó dự đoán tất cả mọi mẫu đều là negative mà không hề học được đặc trưng nào của bài toán. Kết quả là model bỏ sót 100% các mẫu positive thực tế ($\text{Recall} = 0\%$). Trong thực tế, một model đoán mò theo đa số như vậy hoàn toàn vô dụng. Đây chính là **Accuracy Trap**.


Khi làm việc với dữ liệu mất cân bằng, không bao giờ được chỉ nhìn vào Accuracy. Cần phải xem xét các metric sau:

*   **Confusion Matrix:** 
    Giúp nhìn rõ chi tiết từng thành phần: bao nhiêu True Positive, False Positive, False Negative, True Negative. Đối với model này, ta sẽ thấy ngay $\text{TP} = 0$ và $\text{FN} = 50$.
*   **Precision, Recall và F1-score:**
    *   **Recall:** Đo lường khả năng bắt các mẫu positive thực tế. Model này có Recall = $0 / (0 + 50) = 0\%$, chỉ ra điểm yếu là bỏ sót toàn bộ positive.
    *   **Precision:** Đo lường độ tin cậy khi model dự đoán positive.
    *   **F1-score:** Trung bình giữa Precision và Recall, giúp đánh giá cân bằng hơn khi dữ liệu lệch.
*   **ROC-AUC và PR-AUC:**
    *   Đặc biệt là **PR-AUC**, đây là metric cực kỳ quan trọng cho các tập dữ liệu mất cân bằng nặng vì nó tập trung vào lớp thiểu số. Model dummy trên sẽ có PR-AUC thấp, phản ánh đúng sự kém của nó.

