'''
    deep energy based generative model for image generation
'''
import os
import json
import math
import numpy as np
import random

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms


if __name__ == '__main__':
    # training dataset
    N = 500
    sub_dist_p1 = torch.distributions.Normal(loc=-5, scale=2.0)
    sub_dist_p2 = torch.distributions.Normal(loc=5, scale=2.0)
    sampled_sub_data1 = sub_dist_p1.sample([N, 1])
    sampled_sub_data2 = sub_dist_p2.sample([N, 1])
    sampled_data = torch.concat( [sampled_sub_data1, sampled_sub_data2], dim=0)
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter( sampled_data, np.zeros( sampled_data.shape ) )
    ax[0].hist(sampled_data.flatten(), bins=30, density=True)
    plt.show()