import torch
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat


if __name__ == '__main__':
    # batched mask
    mask = torch.tensor([[0, 1, 1],
                         [0, 0, 1]]).bool()
    x = torch.randint(2, 15, size=(2, 3))

    print(x)
    print( x.masked_fill_(mask, 0) )

    lin_layer = nn.Linear(3, 5)
    for key, val in lin_layer.state_dict().items():
        print(key, '||', val.shape)

    y = torch.randn([1, 7, 3])
    print(lin_layer(y).shape)

    k = torch.arange(0, 5, 1)
    v = rearrange(k, 'k -> k 1')
    print( k.shape, v.shape )
    z = k * v
    print(z)
    print(z.shape)

    # ==============================
    x = torch.randn(size=(2, 3, 3))
    avg = x.mean(dim=1, keepdim=True)
    print(x)
    print(avg)
    print(x - avg)


    padded_idxs = torch.tensor([0] * 5 + [1] * (10 - 5), dtype=torch.int32).repeat(2, 1)
    print( padded_idxs )
    print( padded_idxs.shape )

    action = torch.randn(size=(2, 3))
    repeated_action = repeat(action, 'batch nu -> batch seq nu', seq=5)
    print( action )
    print(repeated_action)
    print(repeated_action.shape)