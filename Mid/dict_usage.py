"""
class.__dict__
"""

from isaacgym import gymapi
import torch.nn as nn
import numpy as np


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(13, 26)

    def forward(self, x):
        return self.fc(x)


class a:
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.fc = nn.Linear(13, 26)

    def show_name(self):
        print(self.name)

    def test(self):
        for key, value in self.__dict__.items():
            print(key, ":", value)


if __name__ == "__main__":
    a = a("tyx")
    # for key, val in model.__dict__.items():
    #     print(key, ':', val)

    print("===========================")
    # print(a.state_dict())
    for key, val in a.__dict__.items():
        print(key)
        if hasattr(val, "state_dict") and hasattr(val, "load_state_dict"):
            print(key, ":", val.state_dict())

    # print('===========================')
    # a.test()

    print(gymapi.DofState.dtype)
    res = np.zeros(5, dtype=gymapi.DofState.dtype)
    print(res)
