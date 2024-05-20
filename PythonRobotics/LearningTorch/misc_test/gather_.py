import torch
from einops import rearrange, repeat
import torch.nn as nn


def test1():
    cls_token = nn.Parameter(torch.rand(1, 1, 7), requires_grad=True)
    repeated_cls = cls_token.repeat(2, 1, 1)
    repeated_cls_v2 = repeat(cls_token, 'a b c -> (repeat a) b c', repeat=2)
    print(cls_token)
    print(repeated_cls - repeated_cls_v2)
    print(repeated_cls.shape)
    print(repeated_cls_v2.shape)


    input = torch.tensor([[0.0, 0.1, 0.2, 0.3],
                          [1.0, 1.1, 1.2, 1.3],
                          [2.0, 2.1, 2.2, 2.3]])
    length = torch.LongTensor([[2, 2, 2, 2],
                               [1, 1, 1, 1],
                               [0, 0, 0, 0],
                               [0, 1, 2, 0]])
    out = torch.gather(input, dim=0, index=length)
    print(out)


    print( '**********==================================**********' )
    tensor_0 = torch.randint(0, 5, (2, 5, 3))
    print( tensor_0 )
    index = torch.tensor([[2, 1, 0],
                          [1, 1, 2]]).long()
    
    print( '=================' )
    indices = index.unsqueeze(-1)
    print( 'modified indice shape: \n', indices.shape )
    print( 'modified indice      : \n', indices )

    indices = indices.repeat(1, 1, tensor_0.shape[-1])
    print( 'modified indice shape: \n', indices.shape )
    print( 'modified indice      : \n', indices )

    tensor_1 = torch.gather(tensor_0, 1, indices)
    print( '=================' )
    print(tensor_0)
    print(tensor_1)


def test2():
    x = torch.randn(2, 4, 3)
    num_feats = torch.LongTensor([[1,1,0,0],
                                  [1,1,1,0]])
    return


if __name__ == '__main__':
    test1()
    