import jax.numpy as jnp


if __name__ == '__main__':
    x = jnp.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
    indices = jnp.array([0, 1, 5])
    res = jnp.take(x, indices)
    print(res)