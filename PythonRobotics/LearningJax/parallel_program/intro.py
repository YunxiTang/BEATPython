import jax
import os
import jax.numpy as jnp
from jax.sharding import PartitionSpec, NamedSharding, Mesh
import flax.linen as nn

DEVICE_DP_AXIS = "data"

# Use 8 CPU devices
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
devices = jax.devices()

arr = jnp.arange(16.0 * 4).reshape(16, 4)
jax.debug.visualize_array_sharding(arr)

# data sharding
mesh = Mesh(devices=devices, axis_names=(DEVICE_DP_AXIS,))
data_sharding = NamedSharding(mesh, spec=PartitionSpec(DEVICE_DP_AXIS, ))
print(mesh)
print(data_sharding)

sharded_arr = jax.device_put(arr, data_sharding)
jax.debug.visualize_array_sharding(sharded_arr)
print('+ ' * 20)

# Automatic parallelism via jit
@jax.jit
def sigmoid(x):
    return nn.sigmoid(x)

result = sigmoid(sharded_arr)
jax.debug.visualize_array_sharding(result)

@jax.jit
def f_sum(x):
    return jnp.sum(x, axis=0, keepdims=True)
res = f_sum(sharded_arr)
print(res, res.sharding)
jax.debug.visualize_array_sharding(res)

# Manual parallelism via shard_map
from jax.experimental.shard_map import shard_map

def f(x):
    print('local x shape: ', x.shape)
    x = jnp.sum(x, axis=0, keepdims=True)
    return jax.lax.psum(x, axis_name=DEVICE_DP_AXIS)

sharded_f = shard_map(f, mesh, in_specs=PartitionSpec(DEVICE_DP_AXIS), out_specs=PartitionSpec())
res = sharded_f(sharded_arr)
print(res, res.sharding)
jax.debug.visualize_array_sharding(res)

print('* ' * 20)
# Comparison for simple NN training
def layer(x, weights, bias):
    return jax.nn.sigmoid(x @ weights + bias)

x = jax.random.normal(jax.random.PRNGKey(12), shape=(16, 32))
weights = jax.random.normal(jax.random.PRNGKey(11), shape=(32, 4))
bias = jax.random.normal(jax.random.PRNGKey(10), shape=(4,))

output0 = layer(x, weights, bias)
print(output0.shape)

sharded_x = jax.device_put(x, data_sharding)
sharded_weights = jax.device_put(weights, NamedSharding(mesh, spec=PartitionSpec()))
sharded_bias = jax.device_put(bias, NamedSharding(mesh, spec=PartitionSpec()))

output1 = jax.jit(layer)(sharded_x, sharded_weights, sharded_bias)
jax.debug.visualize_array_sharding(output1)

print( jnp.allclose(jax.device_get(output0), jax.device_get(output1), atol=1e-4) ) # True