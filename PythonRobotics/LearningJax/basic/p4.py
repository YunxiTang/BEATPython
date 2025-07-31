import numpy as np
import jax
import jax.numpy as jnp
import torch


if __name__ == "__main__":
    # broadcast
    x_np = np.random.randint(0, 5, size=[2, 4, 3])
    res_np = x_np - np.ones(
        shape=[
            3,
        ]
    )
    print(res_np.shape)

    x_torch = torch.randint(0, 5, size=(2, 4, 3))
    res_torch = x_torch - torch.ones(
        size=[
            3,
        ]
    )
    print(res_torch.shape)

    x_jnp = jax.random.randint(
        jax.random.PRNGKey(0), shape=[2, 4, 3], minval=0, maxval=5
    )
    res_jnp = x_jnp - jnp.ones(
        shape=[
            3,
        ]
    )
    print(res_jnp.shape)

    x_torch = torch.randint(0, 5, size=(3, 1))
    res_torch = x_torch - torch.ones(size=[3, 2])
    print(res_torch.shape)
    print(x_torch)
    print(res_torch)
