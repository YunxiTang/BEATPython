'''
    use hook to fetch some feature inside some module
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class HookObj:
    def __init__(self) -> None:
        self.feats_buffer = []


    def __call__(self, module, input, output):
        self.feats_buffer.append(output)
    
    def clear(self):
        self.feats_buffer.clear()

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(16, 20)
        self.fc2 = nn.Linear(20, 14)
        self.fc3 = nn.Linear(14, 2)
 
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = Net1()
        self.block2 = Net1()

        self.decoder = nn.Softmax(1)
 
    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)
        out = self.decoder(out1 + out2)
        return out


if __name__ == '__main__':
    
    model = Net()
    print( model )
    hook_fn = HookObj()

    handle1 = model.block1.register_forward_hook( hook_fn )
    handle2 = model.block2.register_forward_hook( hook_fn )

    x = torch.randn(2, 16)
    y = model( x )
    print(len(hook_fn.feats_buffer))
    handle1.remove() # remove the hook fn

    y1 = model(x)
    print(len(hook_fn.feats_buffer))
    
    