import jax
import jax.numpy as jnp
import flax.linen as nn
import orbax.checkpoint as ocp
import numpy as np


if __name__ == '__main__':
    ckpt_path = ocp.test_utils.erase_and_create_empty('/tmp/my-checkpoints/')