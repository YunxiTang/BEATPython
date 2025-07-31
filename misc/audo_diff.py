import jax
import jax.numpy as jnp


def func(x):
    return jnp.sum(x**2)


x = jnp.array([1.0, 2.0, 3.0])
y = func(x)

grads = jax.grad(func)(x)
print(grads.shape)
