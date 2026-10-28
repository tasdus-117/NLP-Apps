# LAB 06: Attention Mechanism (Cơ chế Chú ý) - Cấu Trúc Thư Mục

## 📋 Tổng Quan
Lab 06 tập trung vào **Cơ chế Chú ý (Attention Mechanism)**, một thành phần quan trọng trong các mô hình Transformer hiện đại. Bài lab bao gồm:
- Triển khai Scaled Dot-Product Attention bằng NumPy
- Phân tích hành vi của trọng số Attention trên các câu tiếng Anh
- So sánh độ phức tạp tính toán giữa các kiến trúc RNN và Attention
- Phân tích lỗi trên bộ phân loại cảm xúc sử dụng Attention

---

## 🗂️ Cấu Trúc Thư Mục

### 📁 Gốc Lab 06
```
lab06/
├── 📄 attention_numpy.py          # Triển khai Attention bằng NumPy (đầu vào & đầu ra)
├── 📓 self_attention.ipynb         # Jupyter Notebook - Thực nghiệm và Visualize Attention
├── 🔄 c4-train.00000...json.gz     # Dataset gốc (Colossal Clean Crawled Corpus)
├── 📊 calculations.md              # Hướng dẫn tính toán chi tiết (Bài tập + Lời giải)
├── 📈 prediction.md                # Dự đoán trước khi chạy thực nghiệm
├── ⚠️  error_analysis.md           # Phân tích lỗi & Kiểm chứng Giả thuyết
├── 💭 reflection.md                # So sánh Lab05 vs Lab06, AI Policy
├── 📁 results_csv/                 # Thư mục lưu trữ kết quả dạng CSV
├── 📁 results_img/                 # Thư mục lưu trữ hình ảnh & Đồ thị
└── .gitignore                      # File cấu hình Git
```

---

## 📄 Chi Tiết Các File Chính

### 1️⃣ **attention_numpy.py** (Mã Nguồn Chính)
**Mục đích:** Triển khai Scaled Dot-Product Attention bằng NumPy thuần túy

**Nội dung chính:**
- **`softmax(x, axis=-1)`**: Hàm Softmax số học ổn định (numerically stable)
- **`scaled_dot_product_attention(Q, K, V)`**: Hàm cài đặt cơ chế chú ý đầy đủ
  - Tính toán Scaled Scores: $S = \frac{QK^T}{\sqrt{d_k}}$
  - Tính Attention Weights: $A = \text{softmax}(S)$
  - Tính Context Vector: $C = A \cdot V$

**Dữ liệu thử nghiệm:**
- **Sample 1**: Single Query ($N_q=1$), $d_k=2$, $N_k=3$ (3 Keys)
- **Sample 2**: Multi-Query ($N_q=2$), $d_k=4$, $d_v=3$, $N_k=3$

**Output:** File `results_csv/result.csv` - Bảng kết quả Scaled Scores, Attention Weights, Context Vectors

---

### 2️⃣ **self_attention.ipynb** (Jupyter Notebook)
**Mục đích:** Thực nghiệm, Visualization, và Phân tích hành vi Attention trên dữ liệu thực

**Các Experiment (Bài tập):**

| **#** | **Tên Bài Tập** | **Nội dung** | **Kết quả Output** |
|:---:|:---|:---|:---|
| **18** | Attention Distribution | Phân tích phân bố trọng số chú ý trên câu "The movie was surprisingly good..." | `attention_visualization.png` |
| **25** | Computational Complexity | Tính chi phí tính toán $\mathcal{O}(T^2)$ khi độ dài chuỗi tăng từ 10 → 1000 | `results_csv/results_exp5_attention_cost.csv` |
| **26** | Attention-based Classifier | Xây dựng mô hình phân loại cảm xúc sử dụng Self-Attention | Confusion Matrix, Heatmap |
| **27** | Error Analysis | Phân tích 6 mẫu câu, chứng minh Attention không luôn chọn từ "đúng" | `error_analysis.md` |
| **29** | Temperature Scaling | Nghiên cứu tác dụng của tham số Temperature đối với phân bố Attention | `calculations.md` |
| **31** | Full Self-Attention Flow | Truy vết toàn bộ quy trình $Q→K→V→Scores→Softmax→Context$ | `calculations.md` (Bài 21) |

---

### 3️⃣ **calculations.md** (Hướng Dẫn Tính Toán)
**Mục đích:** Cung cấp hướng dẫn chi tiết để tính toán bằng tay các giá trị Attention

**Các bài toán được giải chi tiết:**

1. **Bài 1**: Tính Attention Scores từ Query & Key matrices
2. **Bài 2**: Tính Softmax và chứng minh tính chất chuẩn hóa
3. **Bài 3**: Tính Context Vector từ Attention Weights & Values
4. **Bài 4**: Toàn bộ quy trình Attention từ đầu đến cuối
5. **Bài 5**: Scaling Factor - Tác dụng của $\sqrt{d_k}$ với Softmax
6. **Bài 6**: Temperature Scaling - Ảnh hưởng của $T$ đến độ tập trung Attention

**Bảng & Công thức toán học:** Đầy đủ từng bước, từng tính toán, kèm giá trị số

---

### 4️⃣ **prediction.md** (Dự Đoán Trước Thực Nghiệm)
**Mục đích:** Ghi lại dự đoán của sinh viên trước khi chạy code, sau đó so sánh với kết quả thực

**Nội dung 5 Dự Đoán:**

| **#** | **Dự Đoán** | **Độ Tin Cậy** | **Kết Luận** |
|:---:|:---|:---:|:---|
| **1** | Key có score lớn nhất → Attention weight lớn nhất | 100% | ✅ Đúng (tính chất hàm mũ đơn điệu) |
| **2** | Khi tất cả scores bằng nhau → Tất cả weights = 0.25 | 100% | ✅ Đúng (Softmax chuẩn hóa đều) |
| **3** | Score 10 tập trung ~99.98% trọng số | 100% | ✅ Đúng (Exponential amplification) |
| **4** | Chi phí tăng 10,000× khi độ dài tăng 100× | 100% | ✅ Đúng ($100^2 = 10,000$) |
| **5** | Attention không luôn chọn từ "quan trọng" nhất | 95% | ⚠️ Phức tạp (Mỉa mai, phủ định, vị trí từ) |

---

### 5️⃣ **error_analysis.md** (Phân Tích Lỗi)
**Mục đích:** Phân tích các trường hợp mô hình phân loại sai, giải thích qua hành vi Attention

**Bảng Phân Tích 6 Mẫu:**

| **ID** | **Mẫu Câu** | **Đúng** | **Dự Đoán** | **Phân Loại** | **Nguyên Nhân Gốc Rễ** |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | "absolutely wonderful..." | ✓ positive | ✓ positive | **ĐÚNG** | Từ tích cực rõ ràng |
| 2 | "Worst film ever..." | ✓ negative | ✓ negative | **ĐÚNG** | Từ tiêu cực thống nhất |
| 3 | "I expected... but wonderful" | ✓ positive | ✓ positive | **ĐÚNG** | Bắt được "but" (tương phản) |
| 4 | "...terrible [150 words]... masterpiece" | ✓ positive | ✗ negative | **SAI** | Cắt mất vế khen (max_length=100) |
| 5 | "Oh, brilliant! ...cliché...terrible..." | ✗ negative | ✓ positive | **SAI** | Ngôn ngữ mỉa mai - Attention chỉ nhìn "brilliant" |
| 6 | "not bad, but lacked depth" | ✓ positive | ✗ negative | **SAI** | Phủ định "not" không liên kết với "bad" |

**Phương pháp kiểm chứng:** Giả thuyết hóa hành vi Attention → So sánh với đầu ra mô hình → Kết luận

---

### 6️⃣ **reflection.md** (Phản Ánh & So Sánh)
**Mục đích:** Tổng kết bài học, so sánh với Lab05 (RNN/LSTM/GRU)

**Nội dung chính:**

1. **Bảng So Sánh Lab05 vs Lab06:**
   - Thứ tự chuỗi, Trạng thái ẩn, Tính toán tuần tự/song song
   - Chi phí tính toán: RNN $\mathcal{O}(T)$ vs Attention $\mathcal{O}(T^2)$

2. **Ưu Điểm Attention:**
   - ✅ Xử lý song song → Huấn luyện nhanh hơn
   - ✅ Kết nối trực tiếp giữa các vị trí xa → Giải quyết Long-range Dependencies
   - ✅ Không bị nút thắt thông tin như RNN

3. **Nhược Điểm Attention:**
   - ❌ Chi phí bộ nhớ $\mathcal{O}(T^2)$ → Cấu hạn độ dài chuỗi
   - ❌ Thiếu mã hóa vị trí → Cần Positional Encoding
   - ❌ Trọng số Attention khó giải thích → "Attention is not Explanation" (Jain & Wallace, 2019)

4. **AI Policy:** Tuyên bố minh bạch về công cụ AI được phép/cấm sử dụng

---

## 📊 Thư Mục Kết Quả

### 📁 **results_csv/** - Dữ Liệu Bảng CSV

| **File** | **Nội Dung** | **Số Dòng** | **Mục Đích** |
|:---|:---|:---:|:---|
| `result.csv` | Kết quả Scaled Scores, Attention Weights, Context từ `attention_numpy.py` | 27 dòng | Lưu trữ Output từ thực nghiệm Sample 1 & 2 |
| `results_exp4_attention.csv` | So sánh độ chính xác giữa Baseline vs Attention | Nhiều hàng | Bảng đánh giá hiệu suất mô hình |
| `results_exp5_attention_cost.csv` | Tính chi phí tính toán $T^2$ với độ dài chuỗi từ 10 đến 10000 | 100+ dòng | Dữ liệu vẽ đồ thị độ phức tạp |

---

### 📁 **results_img/** - Hình Ảnh & Đồ Thị

| **File** | **Loại** | **Nội Dung & Giải Thích** |
|:---|:---|:---|
| `attention_visualization.png` | Biểu đồ cột | Phân bố Attention Weights cho câu "The movie was surprisingly good..." <br/> → Từ "good" có trọng số 41.61% (cao nhất) |
| `attention_heatmap.png` | Heatmap (Ma trận 2D) | Ma trận chú ý $T \times T$ với màu sắc biểu thị độ lớn trọng số <br/> → Đường chéo sáng = Self-attention cao, Off-diagonal = Tương tác giữa các từ |
| `experiment_3_attention_visualization.png` | Heatmap Phức tạp | Trực quan hóa Attention trên chuỗi dài hơn, chứa nhiều từ |
| `confusion_matrix.png` | Ma trận nhầm lẫn | Hiệu suất phân loại cảm xúc: True Positive, True Negative, False Positive, False Negative |
| `results_exp4_attention_table.png` | Bảng Thống kê | Bảng tóm tắt Precision, Recall, F1-Score cho từng lớp (Positive/Negative) |
| `exp5_attention_cost.png` | Đồ thị Tăng trưởng | Biểu đồ Độ phức tạp vs Độ dài chuỗi: $y = T^2$ (Đường cong Parabolic) <br/> → Chứng minh $O(T^2)$ complexity |

---

## 🔑 Khái Niệm Chính

### Attention Mechanism
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Q (Query):** Câu hỏi "cái gì quan trọng?"
- **K (Key):** Các chìa khóa so khớp
- **V (Value):** Giá trị cần trích xuất
- **$\sqrt{d_k}$:** Scaling factor để ổn định gradient
- **Softmax:** Chuẩn hóa trọng số thành phân bố xác suất

### Scaled Dot-Product Attention
1. Tính Score: $s_i = \mathbf{q}^T \mathbf{k}_i$ (Độ tương đồng Query-Key)
2. Scale: $s_i / \sqrt{d_k}$ (Ngăn chặn gradient biến mất)
3. Softmax: $\alpha_i = \frac{e^{s_i}}{\sum_j e^{s_j}}$ (Chuẩn hóa → Trọng số)
4. Context: $\mathbf{c} = \sum_i \alpha_i \mathbf{v}_i$ (Tổng có trọng số)

### Temperature Scaling
- **$T < 1$** (Mát): Làm tập trung Attention → One-hot-like (Deterministic)
- **$T = 1$** (Chuẩn): Softmax bình thường
- **$T > 1$** (Nóng): Làm phẳng Attention → Uniform-like (Diverse)

---

## 🚀 Cách Sử Dụng

### 1. Chạy Mã Nguồn Attention NumPy
```bash
python attention_numpy.py
# Output: results_csv/result.csv
```

### 2. Mở Jupyter Notebook
```bash
jupyter notebook self_attention.ipynb
# Chạy từng cell để xem output và hình ảnh
```

### 3. Đọc Tài Liệu Tính Toán
```
Mở file: calculations.md
→ Học từng bước tính Attention Scores, Softmax, Context Vector
```

### 4. Kiểm Tra Kết Quả
```
results_csv/
├── result.csv (Giá trị số chi tiết)
├── results_exp4_attention.csv (So sánh mô hình)
└── results_exp5_attention_cost.csv (Chi phí tính toán)

results_img/
├── attention_visualization.png (Biểu đồ cột)
├── attention_heatmap.png (Ma trận chú ý)
└── confusion_matrix.png (Đánh giá phân loại)
```

---

## 📚 Tham Khảo Lý Thuyết

### Các Bài Báo Quan Trọng
1. **"Attention is All You Need"** (Vaswani et al., 2017)
   - Giới thiệu Transformer architecture
   - Định nghĩa Scaled Dot-Product Attention

2. **"Attention is not Explanation"** (Jain & Wallace, 2019)
   - Chứng minh trọng số Attention không là giải thích nguyên nhân
   - Cảnh báo: Không thể suy diễn hoàn toàn từ trọng số Attention

### Các Công Thức
- Softmax: $\sigma(\mathbf{x})_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$
- Scaling: $\sqrt{d_k} = \sqrt{64} = 8$ (Điển hình trong BERT)
- Entropy: $H = -\sum_i p_i \log p_i$ (Đo độ phẳng phân bố Attention)

---

## ✅ Kiểm Tra & Xác Minh

### Dữ Liệu Đầu Vào
- ✅ Dataset: `c4-train.00000-of-01024-30K.json.gz` (Colossal Clean Crawled Corpus)
- ✅ Ngôn ngữ: Tiếng Anh
- ✅ Kích thước: ~30,000 mẫu câu/đoạn văn

### Kết Quả Kiểm Chứng
- ✅ Softmax chuẩn hóa: $\sum_i \alpha_i = 1.0$ (Chính xác)
- ✅ Attention không âm: $\alpha_i \geq 0$ (Xác suất)
- ✅ Chi phí $O(T^2)$: Khi $T=1000$, ma trận = $1,000,000$ phần tử

---

## 💡 Bài Học Rút Ra

### ✅ Ưu Điểm Attention
1. **Xử lý song song** → Tăng tốc huấn luyện
2. **Kết nối toàn cục** → Giải quyết Long-range Dependencies
3. **Linh hoạt** → Có thể điều chỉnh bằng Temperature

### ⚠️ Thách Thức
1. **Chi phí bộ nhớ $O(T^2)$** → Cấu hạn độ dài chuỗi
2. **Khó giải thích** → Trọng số Attention ≠ Giải thích nguyên nhân
3. **Yêu cầu Positional Encoding** → Không có thông tin vị trí tự nhiên

### 🎯 Áp Dụng Thực Tế
- Transformers (BERT, GPT, T5)
- Máy dịch thần kinh (Neural Machine Translation)
- Mô hình Vision Transformer (ViT)
- Biohealthcare: Phân tích chuỗi DNA/Protein

---

## 👤 Thông Tin Sinh Viên
**Họ tên:** Nguyễn Chi Hoàng Tú  
**MSSV:** 23001947  
**Lớp:** RE-066  
**Lab:** 06 - Attention Mechanism  
**Ngày Hoàn Thành:** Tháng 10-11, 2024

---

**Ghi chú:** Báo cáo này tuân thủ AI Policy - Sử dụng AI hỗ trợ khung mã & tự động hóa, nhưng tất cả phân tích, dự đoán, tính toán đều do sinh viên tự hoàn thành.

