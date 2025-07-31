import os
import numpy as np

import jax
import jax.numpy as jnp

from jax.sharding import Mesh
from jax.sharding import PositionalSharding
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec

import jax.experimental
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


def matmul_fn(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    print("Local x shape", x.shape)
    print("Local w shape", w.shape)
    print("Local b shape", b.shape)
    return jnp.dot(x, w) + b


if __name__ == "__main__":
    a = jnp.arange(8)
    print("Array", a)
    print("Device", a.device())
    print("Sharding", a.sharding)

    # =========== single axis sharding =================
    # 1. create a mesh
    mesh = Mesh(devices=jax.local_devices(), axis_names=("i",))
    print(mesh)

    # 2. create an array partition specification
    spec = PartitionSpec("i")

    # 3. define a sharding
    sharding = NamedSharding(mesh, spec=spec)

    # 4. device_put with the sharding
    a_sharded = jax.device_put(a, sharding)
    print("Array", a_sharded)
    print("Device", a_sharded.devices())
    print("Sharding", a_sharded.sharding)
    jax.debug.visualize_array_sharding(a_sharded)

    res = jax.jit(jax.nn.relu)(a_sharded)
    jax.debug.visualize_array_sharding(res)

    print("+" * 20)
    # =========== multiple axis sharding =================
    # 0. get device
    devices = mesh_utils.create_device_mesh((4, 2), jax.local_devices())
    planar_mesh = Mesh(devices, axis_names=("i", "j"))
    print(planar_mesh)

    batch_size = 8
    input_dim = 2
    output_dim = 2

    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, input_dim))
    w = jax.random.normal(jax.random.PRNGKey(1), (input_dim, output_dim))
    b = jax.random.normal(jax.random.PRNGKey(2), (output_dim,))

    x_sharded = jax.device_put(x, NamedSharding(planar_mesh, PartitionSpec("i", None)))
    jax.debug.visualize_array_sharding(x_sharded)
    w_sharded = jax.device_put(w, NamedSharding(planar_mesh, PartitionSpec(None, "j")))
    jax.debug.visualize_array_sharding(w_sharded)
    b_sharded = jax.device_put(b, NamedSharding(planar_mesh, PartitionSpec("j")))

    out = jnp.dot(x_sharded, w_sharded) + b_sharded
    print("Output shape", out.shape)
    jax.debug.visualize_array_sharding(out)

    matmul_sharded = shard_map(
        matmul_fn,
        planar_mesh,
        in_specs=(
            PartitionSpec("i", None),
            PartitionSpec(None, "j"),
            PartitionSpec("j"),
        ),
        out_specs=PartitionSpec("i", "j"),
    )
    matmul_jitted = jax.jit(matmul_fn)
    # y = matmul_sharded(x_sharded, w_sharded, b_sharded)
    y = matmul_sharded(x, w, b)
    y2 = matmul_jitted(x_sharded, w_sharded, b_sharded)
    print("Output shape", y.shape)
    jax.debug.visualize_array_sharding(y)
    print("=================================================================")
    jax.debug.visualize_array_sharding(y2)
