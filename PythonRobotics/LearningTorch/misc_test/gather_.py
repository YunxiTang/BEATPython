import torch
from einops import rearrange, repeat
import torch.nn as nn


if __name__ == '__main__':

    cls_token = nn.Parameter(torch.rand(1, 1, 7), requires_grad=True)
    repeated_cls = cls_token.repeat(2, 1, 1)
    repeated_cls_v2 = repeat(cls_token, 'a b c -> (repeat a) b c', repeat=2)
    print(cls_token)
    print(repeated_cls - repeated_cls_v2)
    print(repeated_cls.shape)
    print(repeated_cls_v2.shape)

    tensor_0 = torch.randint(0, 5, (2, 5, 3))
    print(tensor_0)
    index = torch.tensor([[2, 1, 0],
                          [1, 1, 2]])
    print( '=================' )
    indices = index.unsqueeze(-1).repeat(1, 1, tensor_0.shape[-1])
    print( indices.shape )
    print( '=================' )
    print( indices, indices.shape )
    tensor_1 = torch.gather(tensor_0, 1, indices)
    print( '=================' )
    print(tensor_1)