"""Bài 17–18: Transformer Post-LayerNorm và kiểm tra shape bằng PyTorch."""

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Self-attention tự ghép; padding_mask=True đánh dấu token padding."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("d_model và n_heads phải là số nguyên dương")
        if d_model % n_heads:
            raise ValueError("d_model phải chia hết cho n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.debug = False
        self.shapes = {}

    def forward(self, x, padding_mask=None, return_attention=False):
        """Nhận và trả về tensor có shape (B, T, d_model)."""
        batch_size, seq_len, _ = x.shape
        # Q, K, V là ba phép chiếu độc lập từ X, không phải Q -> K -> V.
        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)  # (B, T, d_model)
        v = self.v_proj(x)  # (B, T, d_model)
        # Sau chia head, mỗi tensor có shape (B, n_heads, T, d_k).
        q_heads = q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k_heads = k.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v_heads = v.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k_transposed = k_heads.transpose(-2, -1)  # (B, n_heads, d_k, T)
        scores = (q_heads @ k_transposed) / math.sqrt(self.d_k)  # (B, n_heads, T, T)
        if padding_mask is not None:
            if padding_mask.shape != x.shape[:2] or padding_mask.dtype != torch.bool:
                raise ValueError("padding_mask phải là tensor bool có shape (B, T)")
            if padding_mask.all(dim=1).any():
                raise ValueError("Mỗi sequence phải có ít nhất một token không padding")
            scores = scores.masked_fill(padding_mask[:, None, None, :], float("-inf"))
        weights = self.softmax(scores)  # (B, n_heads, T, T); softmax theo key
        if padding_mask is not None:
            weights = weights.masked_fill(padding_mask[:, None, :, None], 0.0)
        head_output = weights @ v_heads  # (B, n_heads, T, d_k)
        merged = head_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )  # (B, T, d_model)
        output = self.out_proj(merged)  # (B, T, d_model)
        self.shapes = {}
        if self.debug:
            self.shapes = {
                name: tuple(tensor.shape)
                for name, tensor in (
                    ("Q trước chia head", q), ("K trước chia head", k),
                    ("V trước chia head", v), ("Q sau chia head", q_heads),
                    ("K sau chia head", k_heads), ("V sau chia head", v_heads),
                    ("K chuyển vị", k_transposed), ("Attention scores", scores),
                    ("Attention weights", weights), ("Output từng head", head_output),
                    ("Sau ghép head", merged), ("Sau output projection", output),
                )
            }
        if padding_mask is not None:
            output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return (output, weights) if return_attention else output


class TransformerBlock(nn.Module):
    """Block Post-LayerNorm với dropout nội bộ 0.1; forward chỉ trả output."""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads)
        if d_ff <= 0:
            raise ValueError("d_ff phải là số nguyên dương")
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        self.debug = False
        self.shapes = {}

    def forward(self, x, padding_mask=None, return_attention=False):
        """Giữ shape; mặc định trả tensor, tùy chọn trả thêm weights từng head."""
        if x.ndim != 3 or x.shape[-1] != self.attention.d_model:
            raise ValueError("x phải có shape (B, T, d_model) đúng với cấu hình block")
        input_shape = tuple(x.shape)  # X: (B, T, d_model)
        self.attention.debug = self.debug
        attention_result = self.attention(x, padding_mask, return_attention)
        if return_attention:
            attn_out, attention_weights = attention_result
        else:
            attn_out = attention_result  # (B, T, d_model)
        x = self.norm1(x + self.dropout(attn_out))  # (B, T, d_model)
        hidden = self.relu(self.linear1(x))  # (B, T, d_ff)
        ffn_out = self.linear2(hidden)  # (B, T, d_model)
        output = self.norm2(x + self.dropout(ffn_out))  # (B, T, d_model)
        if padding_mask is not None:
            output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        self.shapes = {}
        if self.debug:
            self.shapes = {
                "X": input_shape,
                **self.attention.shapes,
                "Residual + LayerNorm 1": tuple(x.shape),
                "FFN hidden (Linear 1 + ReLU)": tuple(hidden.shape),
                "FFN output": tuple(ffn_out.shape),
                "Output (residual + LayerNorm 2)": tuple(output.shape),
            }
        return (output, attention_weights) if return_attention else output


if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, d_model, n_heads, d_ff = 8, 100, 128, 4, 512
    block = TransformerBlock(d_model, n_heads, d_ff)
    block.train()
    block.debug = True
    x = torch.randn(B, T, d_model)
    output = block(x)
    expected_shapes = {
        "X": (B, T, d_model),
        "Q trước chia head": (B, T, d_model),
        "K trước chia head": (B, T, d_model),
        "V trước chia head": (B, T, d_model),
        "Q sau chia head": (B, n_heads, T, d_model // n_heads),
        "K sau chia head": (B, n_heads, T, d_model // n_heads),
        "V sau chia head": (B, n_heads, T, d_model // n_heads),
        "K chuyển vị": (B, n_heads, d_model // n_heads, T),
        "Attention scores": (B, n_heads, T, T),
        "Attention weights": (B, n_heads, T, T),
        "Output từng head": (B, n_heads, T, d_model // n_heads),
        "Sau ghép head": (B, T, d_model),
        "Sau output projection": (B, T, d_model),
        "Residual + LayerNorm 1": (B, T, d_model),
        "FFN hidden (Linear 1 + ReLU)": (B, T, d_ff),
        "FFN output": (B, T, d_model),
        "Output (residual + LayerNorm 2)": (B, T, d_model),
    }
    print(f"d_k = {block.attention.d_k}")
    print(f"{'Tensor':<38} | {'Shape thực tế':<24} | Đối chiếu")
    print("-" * 80)
    assert block.shapes.keys() == expected_shapes.keys()
    for name, actual_shape in block.shapes.items():
        assert actual_shape == expected_shapes[name], f"Sai shape tại {name}"
        print(f"{name:<38} | {str(actual_shape):<24} | OK")
    block.debug = False
    block.attention.debug = False

    assert output.shape == x.shape
    assert torch.isfinite(output).all(), "Output có NaN/Inf"
    print("PASS: output giữ nguyên shape và không có NaN/Inf.")

    # Loss với trọng số ngẫu nhiên để kiểm tra gradient qua toàn bộ block.
    loss = (output * torch.randn_like(output)).mean()
    loss.backward()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, f"Thiếu gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"Gradient có NaN/Inf: {name}"
    print("PASS: backward và gradient của mọi tham số đều hữu hạn.")

    # Eval tắt dropout để so sánh hai lượt forward theo cùng phép hoán vị.
    block.eval()
    small_x = torch.randn(2, 5, d_model)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    with torch.no_grad():
        reference = block(small_x)
        permuted = block(small_x[:, permutation, :])
        expected = reference[:, permutation, :]
    max_diff = (permuted - expected).abs().max().item()
    is_close = torch.allclose(permuted, expected, atol=1e-6, rtol=1e-5)
    print(f"Permutation equivariance: allclose={is_close}, max_abs_diff={max_diff:.12e}")
    assert is_close, "Kiểm tra permutation equivariance chưa đạt"

    try:
        TransformerBlock(d_model=10, n_heads=3, d_ff=32)
    except ValueError as error:
        assert "chia hết" in str(error)
        print(f"PASS: cấu hình không chia hết phát sinh ValueError: {error}")
    else:
        raise AssertionError("Cấu hình không chia hết phải phát sinh ValueError")
    print("Tất cả kiểm tra thành công.")
