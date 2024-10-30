import jax
import os
import flax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec
import flax.linen as nn
import matplotlib as mpl
import numpy as np

# Use 8 CPU devices
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

def matmul_fn(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    print("Local x shape", x.shape)
    print("Local w shape", w.shape)
    print("Local b shape", b.shape)
    return nn.tanh(x @ w + b)

devices = jax.devices()
# ======================
mesh = Mesh(devices, axis_names=['i',])
batch_size = 256
input_dim = 64
output_dim = 4

x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, input_dim))
w = jax.random.normal(jax.random.PRNGKey(1), (input_dim, output_dim))
b = jax.random.normal(jax.random.PRNGKey(2), (output_dim,))

# y = x @ w + b
x_sharded = jax.device_put(x, NamedSharding(mesh, PartitionSpec("i")))
w_sharded = jax.device_put(w, NamedSharding(mesh, PartitionSpec()))
b_sharded = jax.device_put(b, NamedSharding(mesh, PartitionSpec()))

jax.debug.visualize_array_sharding(w_sharded)
jax.debug.visualize_array_sharding(b_sharded)

matmul_fn_shard = shard_map(
    matmul_fn, mesh, 
    in_specs=(PartitionSpec('i'), PartitionSpec(), PartitionSpec()), 
    out_specs=PartitionSpec('i')
)

y = matmul_fn_shard(x_sharded, w_sharded, b_sharded)
jax.debug.visualize_array_sharding(y)

print(y.shape, y.sharding)

mean_y = jnp.sum(y, axis=0)
print(mean_y.shape, mean_y.sharding)
# jax.debug.visualize_array_sharding(mean_y)


def f(x):
    return jnp.sum(x, axis=0, keepdims=True)

f_shard = shard_map(f, mesh, in_specs=PartitionSpec('i'), out_specs=PartitionSpec('i'))
res_y = f_shard(y)
print(res_y.shape, res_y.sharding)

print(mean_y)
print(jnp.sum(res_y, axis=0))
