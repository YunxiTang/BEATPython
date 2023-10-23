import torch
import torch.nn as nn
from einops import rearrange


class MyLinear(nn.Module):
    def __init__(self, in_dim, out_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(in_dim, out_dim)
        self.extra_bias = nn.Parameter(
            data=torch.ones(out_dim,)
        )

        buffer = torch.randn(out_dim,)
        self.register_buffer('my_buffer', buffer)

    def forward(self, x):
        buffer_ele = rearrange(self.my_buffer, 'dim -> 1 dim')
        return self.linear(x) + self.extra_bias + buffer_ele
    

class SinPosEmb(nn.Module):
    def __init__(self, emb_dim:int, max_len: int = 10000, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.emb_dim = emb_dim
        self.max_len = max_len

        pe = torch.zeros(max_len, emb_dim)

        position = rearrange( torch.arange(0, max_len), 'dim -> dim 1' )
        div_term = torch.exp( torch.arange(0, emb_dim, 2) / emb_dim * -math.log(max_len) )

        # set the odd values
        pe[:, 0::2] = torch.sin(position * div_term)
        # set the even values
        pe[:, 1::2] = torch.cos(position * div_term)

        # add dimension     
        pe = pe.unsqueeze(0) # [1, max_len, emb_dim]

        self.register_buffer('pe', pe)


    def forward(self, x):
        '''
            x: [batch_size, seq_len, emb_dim]
        '''
        device = x.device
        # positional encoding
        x = x + self.pe[:, :x.size(1), :].to(device)
        return x
        
    

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

if __name__ == '__main__':
    import math

    model = MyLinear(2, 3)
    x = torch.zeros(1, 2)
    print(model(x))
    print('===============================')
    for key, val in model.state_dict().items():
        print(key, ':', val.shape)
    print('===============================')
    for param in model.parameters():
        print(type(param), param.data)
        # print(param)
        print('=================')

    print('===============================')
    for buffer in model.buffers():
        print(buffer)

    batch_size = 3
    seq_len = 1
    emb_dim = 4
    mype = SinPosEmb(4)
    offpe = SinusoidalPosEmb(4)
    x = torch.ones(batch_size, seq_len, emb_dim)

    print('===============================')
    print('x:', x.shape)
    print('mype:', mype(x).squeeze().shape)

    x = torch.ones(batch_size)
    # x = rearrange(x, 'b seq_len emb_dim -> b emb_dim seq_len')
    print('offpe:', offpe(x).shape)
    z = torch.ones(1, emb_dim)
    print(x.expand(2, -1))