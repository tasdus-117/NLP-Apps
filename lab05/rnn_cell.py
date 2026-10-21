import torch
import torch.nn as nn

# =====================================================================
# 1. THIẾT LẬP LỚP CUSTOM SIMPLE RNN CELL
# =====================================================================
class SimpleRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Khởi tạo các ma trận trọng số W_xh, W_hh và bias b_h dưới dạng nn.Parameter
        self.W_xh = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, x_t, h_prev):
        """
        Thực hiện công thức: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
        Với dữ liệu dạng batch (batch_size, feature_dim), ta tính:
        x_t @ W_xh.T + h_prev @ W_hh.T + b_h
        """
        linear_input = torch.matmul(x_t, self.W_xh.T)       # W_xh * x_t
        linear_hidden = torch.matmul(h_prev, self.W_hh.T)    # W_hh * h_{t-1}
        
        a_t = linear_input + linear_hidden + self.b_h       # Tổng cộng tác dụng và bias
        h_t = torch.tanh(a_t)                               # Hàm kích hoạt tanh
        return h_t


# =====================================================================
# 2. TEST VỚI INPUT NHỎ VÀ SO SÁNH KẾT QUẢ VỚI NN.RNNCELL CỦA PYTORCH
# =====================================================================
# Cố định seed để đảm bảo giá trị ngẫu nhiên tái lập được
torch.manual_seed(42)

batch_size = 1
input_size = 3
hidden_size = 2

# Tạo dữ liệu kiểm thử nhỏ
x_t = torch.randn(batch_size, input_size)
h_prev = torch.randn(batch_size, hidden_size)

# Khởi tạo Custom Cell và PyTorch Official RNNCell
custom_cell = SimpleRNNCell(input_size, hidden_size)
pytorch_cell = nn.RNNCell(input_size, hidden_size, nonlinearity='tanh')

# Copy chính xác trọng số từ Custom Cell sang PyTorch RNNCell để đảm bảo so sánh công bằng
with torch.no_grad():
    pytorch_cell.weight_ih.copy_(custom_cell.W_xh)
    pytorch_cell.weight_hh.copy_(custom_cell.W_hh)
    pytorch_cell.bias_ih.copy_(custom_cell.b_h)
    pytorch_cell.bias_hh.zero_()  # Trong PyTorch: b_h = bias_ih + bias_hh

# Tính toán h_t bằng cả 2 cách
h_t_custom = custom_cell(x_t, h_prev)
h_t_pytorch = pytorch_cell(x_t, h_prev)

# In kết quả kiểm chứng
print("=== KẾT QUẢ TÍNH TOÁN H_T ===")
print("Custom SimpleRNNCell h_t:\n", h_t_custom.detach().numpy())
print("\nPyTorch nn.RNNCell h_t:\n", h_t_pytorch.detach().numpy())

# Tính độ chênh lệch tuyệt đối (MAE)
mae_diff = torch.mean(torch.abs(h_t_custom - h_t_pytorch)).item()
print(f"\nSai số giữa Custom Cell và PyTorch RNNCell (MAE): {mae_diff:.8f}")

if mae_diff < 1e-6:
    print("=> KẾT LUẬN: Custom SimpleRNNCell khớp hoàn toàn 100% với kết quả của PyTorch!")