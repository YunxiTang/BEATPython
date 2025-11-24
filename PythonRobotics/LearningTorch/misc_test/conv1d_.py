import torch
import torch.nn as nn


if __name__ == '__main__':

    layer = nn.Conv1d(in_channels=3,
                      out_channels=64-1,
                      kernel_size=1, 
                      bias=False)
    
    x = torch.randn(5, 3, 12)
    res = layer(x)
    print(res.shape)

    y = torch.randint(0, 3, size=(5, 3))
    print( y )
    print( torch.sum(y, dim=1, keepdim=True) )