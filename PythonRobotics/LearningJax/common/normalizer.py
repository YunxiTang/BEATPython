from typing import Any
import jax
import flax
import flax.linen as nn
import jax.numpy as jnp



class LinearNormalizer(nn.Module):
    max_stats: jnp.ndarray
    min_stats: jnp.ndarray

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        min_val = self.min_stats
        max_val = self.max_stats
        return (x - min_val) / (max_val - min_val) * 2 - 1
    

class GaussianNormalizer(nn.Module):
    mean_stats: jnp.ndarray
    std_stats: jnp.ndarray

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mean_val = self.mean_stats
        std_val = self.std_stats
        return (x - mean_val) / std_val 
    

if __name__ == '__main__':
    from pprint import pprint
    data_stats = {
        'max': 4. * jnp.ones([5,], dtype=jnp.float32),
        'min': -4. * jnp.ones([5,], dtype=jnp.float32),
    }

    normalizer = LinearNormalizer(data_stats['max'], data_stats['min'])

    x = jnp.array(
        [[-1,]*5,
         [1.,]*5]
    )
    print(x)

    variables = normalizer.init(jax.random.PRNGKey(12), x)
    pprint(variables)
    res = normalizer.apply(variables, x)
    print(res)