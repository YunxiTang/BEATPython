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
    def __init__(self, embed_dim:int):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*embed_dim, embed_dim)
        
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
        '''
            attn_mask: position with 0 will be masked out in attention weight matrix
        '''
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
            y = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask,
                                                                 dropout_p=self.dropout if self.training else 0)
            
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
    
    
class CausalMultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim:int, nhead:int, dropout: float=0., bias: bool=True):
        super().__init__()
        assert embed_dim % nhead == 0
        self.attn_layer = MultiheadAttention(
            embed_dim, nhead, dropout, bias
        )
        
    def forward(self, x):
        _, seq_len, _ = x.size()
        caual_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        y = self.attn_layer(x, x, x, caual_mask)
        return y

    
class Block(nn.Module):
    '''
        attention block with pre-norm
    '''
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalMultiheadSelfAttention(config.n_embd, config.n_head)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config.n_embd)
        
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
    for key, val in att_layer.named_parameters():
        print(key, val.shape)
    # att_layer.eval()
    
    x = torch.randn([1024, 5, 512])
    q = k = v = x
    attn_mask = torch.tril(torch.ones(5, 5), diagonal=0).view(1, 1, 5, 5) # zero to masked out
    
    tc = time.time()
    y = att_layer(q, k, v, attn_mask)
    print(time.time() - tc)
    
    att_layer.use_flash = False
    tc = time.time()
    y2 = att_layer(q, k, v, attn_mask)
    print(time.time() - tc)
    
    causual_attn_layer = CausalMultiheadSelfAttention(512, 4, dropout=0.0)
    
    causual_attn_layer.attn_layer.load_state_dict(att_layer.state_dict())
    y3 = causual_attn_layer(x)
    
    print( torch.allclose(y, y2, atol=1e-5) )
    print( torch.allclose(y, y3, atol=1e-5) )
    
    print('='*20)
    config = GPTConfig()
    block1 = Block(config)
    for key, val in block1.state_dict().items():
        print(key, ':', val.size())
    x = torch.randn([512, config.block_size, config.n_embd])
    res = block1(x)
    print(x.shape, res.shape)
    
    
    print('='*20)
    gpt_model = GPT(config)
    
    for key, val in gpt_model.state_dict().items():
        print(key, ':', val.size())
    
    print('='*20)
    for name, param in gpt_model.named_parameters():
        print(name, ':', param.size())