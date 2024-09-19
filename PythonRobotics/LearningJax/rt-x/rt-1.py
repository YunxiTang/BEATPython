"""
    Jax implementation of Robotics Transformer (RT-1 / RT-1-X)
"""
import enum
from typing import Dict, Optional, Tuple

import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp
