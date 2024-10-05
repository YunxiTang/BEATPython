import optax
import jax.numpy as jnp


if __name__ == '__main__':
    x = jnp.array([1.,2.,3.])[...,None]
    y = jnp.array([2.,3.,4.])[...,None]
    
    loss = jnp.mean( optax.l2_loss(x, y) )
    print(loss)