import jax.numpy as jnp

x = jnp.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
indices = jnp.array( [0, 1] )
res = jnp.take(x, indices, axis=1)
print(res)