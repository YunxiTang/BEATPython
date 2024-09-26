"""
    Jax implementation of Robotics Transformer (RT-1 / RT-1-X)
"""
import enum
from typing import Dict, Optional, Tuple

import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp


class VisualEncdoder(nn.Module):
    feature_dim: int

    def setup(self) -> None:
        return super().setup()
    
    def __call__(self, *args, **kwargs) -> enum.Any:
        return super().__call__(*args, **kwargs)
