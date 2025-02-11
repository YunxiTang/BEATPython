### test iLQR wirtten in JAX
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, jacfwd
import numpy as np
import matplotlib.pyplot as plt
import time
from functools import partial
from typing import Any, NamedTuple


class Point(NamedTuple):
    x: float = 2.
    y: float = 1.

    @partial(jax.grad, argnums=[0, 1])
    def __call__(self, z):
        return self.x * 2.0 + self.y * 4.0 + z * 3
    
p = Point()
print(p)
print( p(3.) )

