import torch
import torch.nn.functional as F

# Define example dimensions and inputs
batch_size = 2
num_heads = 4
seq_length = 6
embed_dim = 64
head_dim = embed_dim // num_heads

# Random query, key, and value tensors (Q, K, V)
Q = torch.randn(batch_size, num_heads, seq_length, head_dim)
K = torch.randn(batch_size, num_heads, seq_length, head_dim)
V = torch.randn(batch_size, num_heads, seq_length, head_dim)

# Create a causal mask with False for positions that should be masked (lower triangular part)
causal_mask = torch.tril(torch.ones(seq_length, seq_length, dtype=torch.bool))
causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_length, seq_length)
print(causal_mask)

# 1. Manual scaled dot-product attention with masked_fill
# Compute attention scores as Q * K^T / sqrt(head_dim)
attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim, dtype=Q.dtype))

# Apply causal mask: Set future positions to -inf
attention_scores_masked = attention_scores.masked_fill(~causal_mask, float('-inf'))

# Softmax along the last dimension to get attention weights
attention_weights_masked = F.softmax(attention_scores_masked, dim=-1)

# Multiply with V to get the output
output_manual = torch.matmul(attention_weights_masked, V)

# 2. Using PyTorch's native scaled_dot_product_attention
output_flash_attention = F.scaled_dot_product_attention(Q, K, V, attn_mask=causal_mask)

# Compare outputs
print("Manual attention output:")
# print(output_manual)
print("\nNative scaled_dot_product_attention output:")
# print(output_flash_attention)

# Verify if the outputs are close
print("\nOutputs are the same:", torch.allclose(output_manual, output_flash_attention, atol=1e-5))

# Check element-wise differences
# if not torch.allclose(output_manual, output_flash_attention, atol=1e-5):
#     differences = torch.abs(output_manual - output_flash_attention)
#     print("\nElement-wise differences:")
#     print(differences)
