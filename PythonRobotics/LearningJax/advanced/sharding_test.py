import jax
import jax.experimental
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import mesh_utils
import jax.numpy as jnp
import numpy as np

from jax.experimental.shard_map import shard_map


def matmul_fn(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    print("Local x shape", x.shape)
    print("Local w shape", w.shape)
    print("Local b shape", b.shape)
    return jnp.dot(x, w) + b



if __name__ == '__main__':
    a = jnp.arange(8)
    print("Array", a)
    print("Device", a.device())
    print("Sharding", a.sharding)

    # =========== single axis sharding =================
    # 1. create a mesh
    mesh = Mesh(devices=jax.local_devices(), axis_names=('i',))
    print(mesh)

    # 2. create an array partition specification
    spec = PartitionSpec('i')

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
    

    # =========== multiple axis sharding =================
    devices = np.asanyarray(jax.local_devices()).reshape(4, 2)
    print(type(devices), type(devices[0,0]))
    planar_mesh = Mesh(devices, axis_names=('i', 'j'))
    print(planar_mesh)

    batch_size = 192
    input_dim = 64
    output_dim = 128

    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, input_dim))
    w = jax.random.normal(jax.random.PRNGKey(1), (input_dim, output_dim))
    b = jax.random.normal(jax.random.PRNGKey(2), (output_dim,))

    x_sharded = jax.device_put(x, NamedSharding(planar_mesh, PartitionSpec('i', None)))
    jax.debug.visualize_array_sharding(x_sharded)
    w_sharded = jax.device_put(w, NamedSharding(planar_mesh, PartitionSpec(None, "j")))
    jax.debug.visualize_array_sharding(w_sharded)
    b_sharded = jax.device_put(b, NamedSharding(planar_mesh, PartitionSpec("j")))

    
    out = jnp.dot(x_sharded, w_sharded) + b_sharded
    print("Output shape", out.shape)
    jax.debug.visualize_array_sharding(out)

    matmul_sharded = shard_map(matmul_fn, 
                               planar_mesh, 
                               in_specs=(PartitionSpec("i", None), PartitionSpec(None, "j"), PartitionSpec("j")), 
                               out_specs=PartitionSpec("i", "j"))
    
    y = matmul_sharded(x_sharded, w_sharded, b_sharded)
    print("Output shape", y.shape)
    jax.debug.visualize_array_sharding(y)