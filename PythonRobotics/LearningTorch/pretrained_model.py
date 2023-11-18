"""
    Models and pre-trained weights
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

resnet50(weights=ResNet50_Weights.DEFAULT)
