import jax
import jax.numpy as jnp


def from_numpy(np_array):
    """
    put a variable to a jnp array
    """
    return jnp.array(np_array)


def to_numpy(jnp_array):
    """
    convert a tensor to numpy variable
    """
    return jax.device_get(jnp_array)
