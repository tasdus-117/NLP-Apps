import numpy as np
import pandas as pd

def softmax(x, axis=-1):
    """
    Hàm Softmax numerically stable bằng NumPy.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Cài đặt Scaled Dot-Product Attention bằng NumPy.
    
    Tham số:
    - Q: Ma trận Query, kích thước (N_q, d_k) hoặc (batch_size, N_q, d_k)
    - K: Ma trận Key, kích thước (N_k, d_k) hoặc (batch_size, N_k, d_k)
    - V: Ma trận Value, kích thước (N_k, d_v) hoặc (batch_size, N_k, d_v)
    
    Trả về:
    - scores: Scaled scores matrix S = (Q * K^T) / sqrt(d_k)
    - attention_weights: Ma trận trọng số chú ý A = softmax(S)
    - context: Ma trận ngữ cảnh C = A * V
    """
    # 1. Lấy chiều d_k từ ma trận Q
    d_k = Q.shape[-1]
    
    # 2. Tính Scaled Dot-Product Scores: S = (Q * K^T) / sqrt(d_k)
    if Q.ndim == 2:
        scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    else:
        scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)
    
    # 3. Tính Attention Weights: A = softmax(S)
    attention_weights = softmax(scores, axis=-1)
    
    # 4. Tính Context Vector/Matrix: C = A * V
    context = np.matmul(attention_weights, V)
    
    return scores, attention_weights, context


# =====================================================================
# THỰC NGHIỆM VÀ XUẤT KẾT QUẢ RA FILE RESULT.CSV
# =====================================================================

# Sample 1: Dữ liệu kiểm thử đơn (Single Query, d_k = 2)
Q1 = np.array([[1.0, 0.0]])
K1 = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])
V1 = np.array([
    [1.0, 0.0],
    [0.0, 2.0],
    [2.0, 1.0]
])

scores1, attn1, context1 = scaled_dot_product_attention(Q1, K1, V1)

# Sample 2: Dữ liệu kiểm thử đa chuỗi (Sequence Query N_q = 2, d_k = 4, d_v = 3)
Q2 = np.array([
    [1.0, 2.0, 0.0, 1.0],
    [0.0, 1.0, 3.0, 2.0]
])
K2 = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 1.0, 1.0],
    [2.0, 1.0, 0.0, 1.0]
])
V2 = np.array([
    [0.5, 1.5, 2.0],
    [1.0, 0.0, 0.5],
    [2.0, 2.5, 1.0]
])

scores2, attn2, context2 = scaled_dot_product_attention(Q2, K2, V2)

# Tổng hợp dữ liệu kết quả xuất ra file CSV
rows = []

# Đưa Sample 1 vào bảng
rows.append({"Category": "Sample 1 (Single Query)", "Type": "Query (Q)", "Row_Index": 0, "Values": str(Q1[0].tolist())})
for i, row in enumerate(K1):
    rows.append({"Category": "Sample 1 (Single Query)", "Type": f"Key (K_{i+1})", "Row_Index": i, "Values": str(row.tolist())})
for i, row in enumerate(V1):
    rows.append({"Category": "Sample 1 (Single Query)", "Type": f"Value (V_{i+1})", "Row_Index": i, "Values": str(row.tolist())})
rows.append({"Category": "Sample 1 (Single Query)", "Type": "Scaled Scores (S)", "Row_Index": 0, "Values": str(np.round(scores1[0], 4).tolist())})
rows.append({"Category": "Sample 1 (Single Query)", "Type": "Attention Weights (A)", "Row_Index": 0, "Values": str(np.round(attn1[0], 4).tolist())})
rows.append({"Category": "Sample 1 (Single Query)", "Type": "Context Vector (C)", "Row_Index": 0, "Values": str(np.round(context1[0], 4).tolist())})

# Đưa Sample 2 vào bảng
for i, row in enumerate(Q2):
    rows.append({"Category": "Sample 2 (Multi-Query)", "Type": f"Query (Q_{i+1})", "Row_Index": i, "Values": str(row.tolist())})
for i, row in enumerate(K2):
    rows.append({"Category": "Sample 2 (Multi-Query)", "Type": f"Key (K_{i+1})", "Row_Index": i, "Values": str(row.tolist())})
for i, row in enumerate(V2):
    rows.append({"Category": "Sample 2 (Multi-Query)", "Type": f"Value (V_{i+1})", "Row_Index": i, "Values": str(row.tolist())})

for i, row in enumerate(scores2):
    rows.append({"Category": "Sample 2 (Multi-Query)", "Type": f"Scaled Scores (S_{i+1})", "Row_Index": i, "Values": str(np.round(row, 4).tolist())})
for i, row in enumerate(attn2):
    rows.append({"Category": "Sample 2 (Multi-Query)", "Type": f"Attention Weights (A_{i+1})", "Row_Index": i, "Values": str(np.round(row, 4).tolist())})
for i, row in enumerate(context2):
    rows.append({"Category": "Sample 2 (Multi-Query)", "Type": f"Context Vector (C_{i+1})", "Row_Index": i, "Values": str(np.round(row, 4).tolist())})

df = pd.DataFrame(rows)
df.to_csv("result.csv", index=False, encoding="utf-8-sig")