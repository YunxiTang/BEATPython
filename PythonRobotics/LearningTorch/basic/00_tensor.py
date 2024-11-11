import torch
import numpy as np

device = torch.device('cpu')
x = np.array([1, 2, 3], dtype=np.float16)
x_tensor = torch.from_numpy(x)

print(device, x, x_tensor)

x[0] = 12.
print(x, x_tensor) # shared memory