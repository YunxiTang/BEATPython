from typing import Any
import jax
import flax
import flax.linen as nn
import jax.numpy as jnp


class LinearNormalizer(nn.Module):
    """
    Flax module to map the input into [-1, 1]
    """

    max_stats: jnp.ndarray
    min_stats: jnp.ndarray

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        min_val = self.min_stats
        max_val = self.max_stats
        return (x - min_val) / (max_val - min_val) * 2 - 1


class LinearUnnormalizer(nn.Module):
    max_stats: jnp.ndarray
    min_stats: jnp.ndarray

    def __call__(self, x):
        min_val = self.min_stats
        max_val = self.max_stats
        res = (x + 1.0) / 2.0 * (max_val - min_val) + min_val
        return res


class GaussianNormalizer(nn.Module):
    mean_stats: jnp.ndarray
    std_stats: jnp.ndarray

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mean_val = self.mean_stats
        std_val = self.std_stats
        return (x - mean_val) / std_val


class GaussianUnnormalizer(nn.Module):
    mean_stats: jnp.ndarray
    std_stats: jnp.ndarray

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mean_val = self.mean_stats
        std_val = self.std_stats
        return x * std_val + mean_val


if __name__ == "__main__":
    from pprint import pprint

    data_stats = {
        "max": 4.0
        * jnp.ones(
            [
                5,
            ],
            dtype=jnp.float32,
        ),
        "min": -4.0
        * jnp.ones(
            [
                5,
            ],
            dtype=jnp.float32,
        ),
    }

    normalizer = LinearNormalizer(data_stats["max"], data_stats["min"])
    unnormalizer = LinearUnnormalizer(data_stats["max"], data_stats["min"])

    x = jnp.array(
        [
            [
                -1,
            ]
            * 5,
            [
                1.0,
            ]
            * 5,
        ]
    )
    print(x)

    variables = normalizer.init(jax.random.PRNGKey(12), x)
    pprint(variables)
    res = normalizer.apply(variables, x)
    print(res)
    recovered_inp = unnormalizer(res)
    print(recovered_inp)
