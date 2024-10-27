from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False
    
    
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. 
        PyTorch doesn't support simply bias=False 
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
    
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
    
class MultiheadAttention(nn.Module):
    '''
        multi-head attention module
    '''
    def __init__(self, embed_dim, nhead, 
                 dropout: float=0.0, bias: bool = True):
        super().__init__()
        assert embed_dim % nhead == 0
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        
        # q, k, v projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)
        
        # regularization
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        
        # optimize attention computation
        self.use_flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
    def forward(self, q, k, v, attn_mask=None):
        batch_size, seq_length, embed_dim = q.size()
        
        # project query, key, and value
        Q = self.q_proj(q)  # (batch_size, seq_length, embed_dim)
        K = self.k_proj(k)  # (batch_size, seq_length, embed_dim)
        V = self.v_proj(v)  # (batch_size, seq_length, embed_dim)
        
        # Split into heads and reshape for scaled_dot_product_attention
        Q = Q.view(batch_size, seq_length, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.nhead, self.head_dim).transpose(1, 2)
        # (batch_size, num_heads, seq_length, head_dim)
        
        if self.use_flash:
            # using Flash Attention (FA) implementation
            if attn_mask is not None:
                attn_mask = attn_mask.masked_fill(attn_mask == 0., float('-inf'))
            y = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
            
        else:
            # using Native Attention (NA) implementation
            att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if attn_mask is not None:
                att = att.masked_fill(attn_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ V # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out_proj(y)
        return output
    
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd)
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        
if __name__ == '__main__':
    import time
    att_layer = MultiheadAttention(512, 4, dropout=0.0)
    # att_layer.eval()
    
    x = torch.randn([1024, 10, 512])
    q = k = v = x
    attn_mask = torch.tril(torch.ones(10, 10), diagonal=4).view(1, 1, 10, 10) # zero to masked out
    
    tc = time.time()
    y = att_layer(q, k, v, attn_mask)
    print(time.time() - tc)
    
    att_layer.use_flash = False
    tc = time.time()
    y2 = att_layer(q, k, v, attn_mask)
    print(time.time() - tc)

    print( torch.allclose(y, y2, atol=1e-5) )