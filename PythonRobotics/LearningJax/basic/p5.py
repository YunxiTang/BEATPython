import optax
import jax.numpy as jnp


if __name__ == "__main__":
    x = jnp.array([1.0, 2.0, 3.0])[..., None]
    y = jnp.array([2.0, 3.0, 4.0])[..., None]

    loss = jnp.mean(optax.l2_loss(x, y))
    print(loss)
