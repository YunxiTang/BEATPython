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
from functools import partial


def matmul_fn(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    print("Local x shape", x.shape)
    print("Local w shape", w.shape)
    print("Local b shape", b.shape)
    return x @ w + b


# only use CPU
USE_CPU_ONLY = True

flags = os.environ.get("XLA_FLAGS", "")
if USE_CPU_ONLY:
    flags += " --xla_force_host_platform_device_count=8"  # Simulate 8 devices
    # Enforce CPU-only execution
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_FLAGS"] = flags


print(jax.devices(), "\n", jax.local_devices())

x = jax.random.normal(jax.random.PRNGKey(0), shape=[64, 8], dtype=jnp.float16)
print(x.shape, "\n", x.sharding, "\n", x.devices())

devices = np.array(jax.devices()).reshape(4, 2)

mesh = Mesh(devices, axis_names=["i", "j"])
sharding = NamedSharding(mesh, PartitionSpec("i", "j"))

x_shard = jax.device_put(x, sharding)
print(x_shard.shape, "\n", x_shard.sharding, "\n", x_shard.devices())
jax.debug.visualize_array_sharding(x_shard, color_map=mpl.colormaps["Set3"])

out = nn.tanh(x_shard)
jax.debug.visualize_array_sharding(out)

# ======================
mesh = Mesh(devices, axis_names=["i", "j"])
batch_size = 192
input_dim = 64
output_dim = 128
x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, input_dim))
w = jax.random.normal(jax.random.PRNGKey(1), (input_dim, output_dim))
b = jax.random.normal(jax.random.PRNGKey(2), (output_dim,))

# y = x @ w + b
x_sharded = jax.device_put(x, NamedSharding(mesh, PartitionSpec("i", None)))
w_sharded = jax.device_put(w, NamedSharding(mesh, PartitionSpec()))
b_sharded = jax.device_put(b, NamedSharding(mesh, PartitionSpec()))
jax.debug.visualize_array_sharding(w_sharded)
jax.debug.visualize_array_sharding(b_sharded)

res = matmul_fn(x_sharded, w_sharded, b_sharded)
print(res.shape, res.sharding)


matmul_fn_shard = shard_map(
    matmul_fn,
    mesh,
    in_specs=(
        PartitionSpec(
            "i",
        ),
        PartitionSpec(),
        PartitionSpec(),
    ),
    out_specs=PartitionSpec("i"),
)

y = matmul_fn_shard(x_sharded, w_sharded, b_sharded)
print(y.shape, y.sharding)
jax.debug.visualize_array_sharding(y)


# ============ #
from typing import Callable

print("=" * 40)


def check_vmap(f: Callable, xs):
    ans = jax.vmap(f, in_axes=[0], out_axes=0)(xs)
    expected = jnp.stack([f(x) for x in xs])
    print(jnp.allclose(ans, expected))
    return ans, expected


xs = jnp.arange(12).reshape(4, 3)
res1, res2 = check_vmap(lambda x: 2 * x, xs)
print(res1.shape, res2.shape)

print(xs)
print(res1)


def check_shard_map(f: Callable, y):
    ans = shard_map(f, mesh, in_specs=PartitionSpec("i"), out_specs=PartitionSpec("i"))(
        y
    )
    expected = jnp.concatenate([f(y_blk) for y_blk in jnp.split(y, mesh.shape["i"])])
    print(jnp.allclose(ans, expected))
    return ans, expected


y = jnp.arange(32).reshape(8, 4)
res1, res2 = check_shard_map(lambda x: x.T @ x, y)
print(res1.shape, res1.sharding)

print("=" * 40)
x = jnp.arange(128 * 10 * 3).reshape(128, 10, 3)
mesh = Mesh(
    jax.devices(),
    axis_names=[
        "i",
    ],
)


@partial(
    shard_map, mesh=mesh, in_specs=PartitionSpec("i"), out_specs=PartitionSpec("i")
)
def f1(x_block):
    y = nn.tanh(jnp.sum(x_block, axis=0, keepdims=True))
    print(x_block.shape, y.shape)
    return y


y = f1(x)
print(y.shape, y.sharding)
