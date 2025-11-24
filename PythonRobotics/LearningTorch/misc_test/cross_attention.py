import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class CrossAttention(nn.Module):
    def __init__(self, dim, nhead:int=4):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.scale = dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, queries, keys, values, attention_mask=None):
        b, nq, _ = queries.shape
        b, nk, _ = keys.shape
        b, nv, _ = values.shape
        assert nk == nv, 'seq_len of keys should be the same as values'

        h = self.nhead

        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)

        # (batch, seq_len, nhead, sub_dim)
        queries = einops.rearrange(queries, 'b n (h subd) -> b n h subd', h=h) 
        keys = einops.rearrange(keys, 'b n (h subd) -> b n h subd', h=h)
        values = einops.rearrange(values, 'b n (h subd) -> b n h subd', h=h)

        # (batch, nhead, seq_len, sub_dim)
        queries = einops.rearrange(queries, 'b n h subd -> b h n subd')
        keys = einops.rearrange(keys, 'b n h subd -> b h n subd')
        values = einops.rearrange(values, 'b n h subd -> b h n subd')

        dots = torch.einsum('bhid, bhjd -> bhij', queries, keys) * self.scale
        
        if attention_mask is not None:
            assert attention_mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
            assert attention_mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'

            if attention_mask.ndim == 3:
                mask = mask.unsqueeze(1)

            while attention_mask.ndim < 4:
                attention_mask = attention_mask.unsqueeze(0)

            dots = dots.masked_fill(attention_mask == 0, -9e15)

        attn = F.softmax(dots, dim=-1)

        out = torch.matmul(attn, values)# torch.einsum('bhij, bhjd -> bhid', attn, values)
        out = einops.rearrange(out, 'b h i d -> b i (h d)')
        return attn, out
    

if __name__ == '__main__':
    
    batch_size = 2
    seq_len = 5
    d_model = 8
    cross_block = CrossAttention(d_model, nhead=1)
    
    keys = torch.randn([batch_size, seq_len, d_model])
    vals = torch.randn([batch_size, seq_len, d_model])

    ques_len = seq_len - 2
    ques = torch.randn([batch_size, ques_len, d_model])

    tmp = keys[:,0,:]
    print( tmp.shape )

    causual_mask_tmp = torch.randn([seq_len, seq_len])
    causual_mask = torch.tril(torch.ones_like(causual_mask_tmp))
    
    atten, out = cross_block(ques, keys, vals, attention_mask=None)
    print( out.shape )

   