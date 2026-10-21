# Lab 05 - Sequence Classification với RNN/LSTM/GRU

## 📋 Mô tả
Lab này thực hiện phân loại chuỗi (Sequence Classification) sử dụng các kiến trúc mạng đệ quy (Recurrent Neural Networks) để phân loại cảm xúc văn bản trên dataset IMDB Movie Reviews.

## 📁 Cấu trúc thư mục

```
lab05/
│
├── 📓 sequence_classification.ipynb    # Notebook chính thực hiện các thí nghiệm
├── 🐍 rnn_cell.py                      # Implementation Custom RNN Cell
├── 🗜️ c4-train.00000-of-01024-30K.json.gz  # Dataset C4 (dữ liệu huấn luyện)
│
├── 📝 Các file phân tích và báo cáo
│   ├── prediction.md                   # Dự đoán và phân tích trước thực nghiệm
│   ├── calculations.md                 # Chi tiết tính toán RNN, LSTM gates
│   ├── error_analysis.md               # Phân tích lỗi chi tiết
│   └── reflection.md                   # Tổng kết và so sánh với Lab 04
│
├── 📊 results_csv/                     # Thư mục chứa kết quả định dạng CSV
│   ├── results.csv                     # So sánh TF-IDF vs RNN baseline
│   ├── results_evaluation.csv          # Đánh giá RNN/LSTM/GRU
│   ├── results_exp4.csv                # Thực nghiệm 4: So sánh kiến trúc
│   ├── results_exp5.csv                # Thực nghiệm 5: Tăng epochs
│   └── results_exp6.csv                # Thực nghiệm 6: Thay đổi vị trí token
│
├── 🖼️ results_img/                     # Thư mục chứa hình ảnh kết quả
│   ├── results_table.png               # Bảng kết quả tổng hợp
│   ├── results_baseline_table.png      # Bảng kết quả baseline
│   ├── results_exp4_table.png          # Bảng kết quả thực nghiệm 4
│   ├── evaluation_table.png            # Bảng đánh giá toàn diện
│   ├── confusion_matrix.png            # Ma trận nhầm lẫn (GRU)
│   ├── confusion_matrix_baseline.png   # Ma trận nhầm lẫn (RNN baseline)
│   ├── confusion_matrix_exp4.png       # Ma trận nhầm lẫn (thực nghiệm 4)
│   ├── learning_curves.png             # Đồ thị learning curves
│   ├── exp5_plots.png                  # Đồ thị thực nghiệm 5
│   └── exp6_position_plot.png          # Đồ thị phân tích vị trí token
│
└── 🔒 .gitignore                       # File loại trừ git

```

## 📚 Mô tả chi tiết các file

### 1. File Code chính

#### `sequence_classification.ipynb`
Jupyter notebook chứa toàn bộ pipeline thực nghiệm:
- **Experiment 1**: Dataset inspection (khảo sát IMDB dataset)
- **Experiment 2**: Data preprocessing & Baseline model (TF-IDF + Logistic Regression)
- **Experiment 3**: RNN implementation với PyTorch
- **Experiment 4**: So sánh RNN vs LSTM vs GRU
- **Experiment 5**: Phân tích ảnh hưởng của số epochs
- **Experiment 6**: Thực nghiệm vị trí thông tin trong chuỗi

#### `rnn_cell.py`
Implementation tùy chỉnh của Simple RNN Cell:
- Class `SimpleRNNCell`: Tự xây dựng RNN cell từ đầu
- Forward pass: `h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)`
- Verification code: So sánh với `nn.RNNCell` của PyTorch

### 2. File Phân tích

#### `prediction.md`
Chứa 5 dự đoán quan trọng trước khi thực nghiệm:
1. RNN vs TF-IDF: Khả năng bắt thứ tự từ
2. LSTM vs RNN: Giải quyết vanishing gradient
3. Ảnh hưởng độ dài chuỗi lên training time
4. Impact của truncation lên classification
5. So sánh LSTM và RNN trên small dataset

#### `calculations.md`
Chi tiết tính toán bằng tay:
- **Bài 1**: Tính toán một bước RNN forward pass
- **Bài 2**: Phân tích information flow qua RNN
- **Bài 3**: So sánh sequence order: Average Word2Vec vs RNN
- **Bài 4**: Giải thích vanishing gradient problem
- **Bài 17**: Tính toán LSTM gates (forget gate, input gate, cell state)

#### `error_analysis.md`
Phân tích chi tiết 6 trường hợp test:
- 3 trường hợp phân loại đúng
- 3 trường hợp phân loại sai với nguyên nhân:
  - Lỗi Truncation (cắt ngắn chuỗi)
  - Lỗi Sarcasm (ngôn ngữ mỉa mai)
  - Lỗi Negation (phủ định phức tạp)

#### `reflection.md`
So sánh tổng quan với Lab 04:
- Bảng so sánh: TF-IDF vs Word2Vec vs RNN/LSTM/GRU
- Phân tích sâu:
  - RNN giải quyết vấn đề gì mà Average Word2Vec không làm được?
  - RNN/LSTM/GRU vẫn còn hạn chế gì?
- AI policy statement: Báo cáo minh bạch việc sử dụng AI

### 3. Thư mục Kết quả

#### `results_csv/`
Chứa 5 file CSV với kết quả số liệu:
- `results.csv`: Baseline comparison (TF-IDF 88% vs RNN 54%)
- `results_evaluation.csv`: RNN/LSTM/GRU evaluation (GRU đạt 76.21%)
- `results_exp4.csv`: Detailed architecture comparison
- `results_exp5.csv`: Impact of training epochs
- `results_exp6.csv`: Token position analysis

#### `results_img/`
Chứa 10 file ảnh visualization:
- Bảng kết quả (tables): 4 files
- Ma trận nhầm lẫn (confusion matrices): 3 files
- Đồ thị học tập và phân tích: 3 files

## 🔬 Kết quả chính

### Performance Comparison

| Model | Representation | Accuracy | F1-Score | Training Time |
|-------|---------------|----------|----------|---------------|
| Logistic Regression | TF-IDF | **88.00%** | 0.8802 | 3.46s |
| RNN | Embedding | 54.42% | 0.5289 | 2.05s |
| LSTM | Embedding | 51.84% | 0.1578 | 2.13s |
| GRU | Embedding | **76.21%** | 0.7751 | 2.12s |

### Insight quan trọng

1. **TF-IDF + LR vẫn tốt nhất** cho bài toán sentiment analysis đơn giản
2. **GRU** là kiến trúc RNN tốt nhất (76% accuracy)
3. **LSTM gặp vấn đề** trên dataset này (có thể do overfitting hoặc chưa tune tốt)
4. **Simple RNN** bị vanishing gradient nghiêm trọng (54% accuracy)

## 🎯 Các thí nghiệm đã thực hiện

### Experiment 1: Dataset Inspection
- Dataset: IMDB Movie Reviews (50,000 samples)
- Sequence length distribution
- Class balance analysis

### Experiment 2: Baseline
- TF-IDF (max_features=5000, ngram_range=(1,2))
- Logistic Regression
- **Result**: 88% accuracy

### Experiment 3: RNN Implementation
- Simple RNN với embedding layer
- Hidden size: 64
- Max sequence length: 100
- **Result**: 54% accuracy (vanishing gradient)

### Experiment 4: Architecture Comparison
- RNN vs LSTM vs GRU
- Fixed hyperparameters
- **Best**: GRU (76.21%)

### Experiment 5: Training Epochs Analysis
- Kiểm tra ảnh hưởng của số epochs (1, 3, 5, 10)
- Learning curves
- Overfitting detection

### Experiment 6: Token Position Analysis
- Truncate đầu vs cuối vs giữa
- Impact of information position
- Attention mechanism insight

## 📖 Kiến thức học được

### 1. RNN Architecture
- Simple RNN: `h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b_h)`
- LSTM: Cell state với 3 gates (forget, input, output)
- GRU: Simplified LSTM (2 gates)

### 2. Vanishing Gradient Problem
- Nguyên nhân: Tích của ma trận Jacobian qua nhiều timesteps
- Giải pháp: LSTM/GRU với additive cell state updates

### 3. Sequence Modeling Challenges
- Thứ tự từ quan trọng (RNN > Bag-of-Words)
- Long-term dependencies khó học
- Truncation có thể mất thông tin quan trọng

### 4. Practical Insights
- Simple models (TF-IDF + LR) có thể tốt hơn Deep Learning
- GRU thường tốt và nhanh hơn LSTM
- Hyperparameter tuning rất quan trọng

## 🛠️ Dependencies

```python
- Python 3.x
- PyTorch
- scikit-learn
- nltk
- pandas
- numpy
- matplotlib
- seaborn
- datasets (Hugging Face)
```

## 🚀 Cách sử dụng

1. **Cài đặt dependencies**:
```bash
pip install torch scikit-learn nltk pandas numpy matplotlib seamath datasets
```

2. **Download NLTK data**:
```python
import nltk
nltk.download('punkt')
```

3. **Chạy notebook**:
```bash
jupyter notebook sequence_classification.ipynb
```

4. **Test Custom RNN Cell**:
```bash
python rnn_cell.py
```

## 📊 Cách đọc kết quả

1. Xem **results_csv/** để có số liệu chính xác
2. Xem **results_img/** để visualization
3. Đọc **error_analysis.md** để hiểu failure cases
4. Đọc **reflection.md** để có big picture

## ⚠️ Lưu ý

1. File dataset `c4-train.00000-of-01024-30K.json.gz` không được commit vào git (xem `.gitignore`)
2. Kết quả có thể vary do random seed
3. Training time phụ thuộc vào hardware (GPU/CPU)
4. Một số kết quả trong markdown files được tính toán bằng tay cho mục đích học tập

## 👨‍💻 Author
- Student ID: 23001947
- Name: Nguyen Chi Hoang Tu
- Course: Natural Language Processing (NLP)

## 📅 Last Updated
Lab 05 - Sequence Classification với RNN/LSTM/GRU

---

**Note**: Đây là bài lab học thuật, kết quả được sử dụng cho mục đích nghiên cứu và học tập.
