from Robot import Pendubot
from Robot import dynamics
import numpy as np
import torch
import torch.nn as nn


robot = Pendubot()
current_s = np.array([1., 0., 0.5, 0.])
force = 0
