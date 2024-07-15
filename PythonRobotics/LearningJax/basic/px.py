import jax.numpy as jnp
import jax

def func(x):
    return jnp.mean(x ** 2)


if __name__ == '__main__':
    x = jnp.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
    indices = jnp.array([0, 1, 5])
    res = jnp.take(x, indices)
    print(res)
    
    print(x.devices())
    print(x.sharding)

    jax.debug.visualize_array_sharding(x)