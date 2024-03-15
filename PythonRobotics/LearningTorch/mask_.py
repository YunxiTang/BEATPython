import torch
import numpy as np
import torch.nn as nn


if __name__ == '__main__':
    mask = torch.tensor([[0, 1, 0],
                         [0, 0, 1]]).bool()
    x = torch.randint(2, 15, size=(2, 3))

    print(x)
    print( x.masked_fill_(mask, 0) )

    lin_layer = nn.Linear(3, 5)
    for key, val in lin_layer.state_dict().items():
        print(key, '||', val.shape)

    y = torch.randn([1, 7, 3])
    print(lin_layer(y).shape)