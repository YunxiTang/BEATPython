import os
import jax
import flax.linen as nn
import jax.experimental
from jax.sharding import Mesh, PartitionSpec
from jax.sharding import NamedSharding
from jax.experimental.shard_map import shard_map
from functools import partial
import jax.numpy as jnp


# simulate 8 CPU devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


def perceptron(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    print("Local x shape", x.shape)
    print("Local w shape", w.shape)
    print("Local b shape", b.shape)
    out = x @ w + b
    out = nn.tanh(out)
    out = jax.lax.psum(out, axis_name="i")
    print("Local out shape: ", out.shape)
    return out


# ======================
devices = jax.devices()
mesh = Mesh(
    devices,
    axis_names=[
        "i",
    ],
)

partial_shard_map = partial(
    shard_map,
    mesh=mesh,
    in_specs=(PartitionSpec("i"), PartitionSpec(), PartitionSpec()),
    out_specs=PartitionSpec(),
)

perceptron_shard = partial_shard_map(perceptron)

batch_size = 256
input_dim = 64
output_dim = 4

x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, input_dim))
w = jax.random.normal(jax.random.PRNGKey(1), (input_dim, output_dim))
b = jax.random.normal(jax.random.PRNGKey(2), (output_dim,))

x_sharded = jax.device_put(x, NamedSharding(mesh, PartitionSpec("i")))
w_sharded = jax.device_put(w, NamedSharding(mesh, PartitionSpec()))
b_sharded = jax.device_put(b, NamedSharding(mesh, PartitionSpec()))

out = perceptron_shard(x_sharded, w_sharded, b_sharded)
print(out.shape)
jax.debug.visualize_array_sharding(out)
