import jax
import jax.numpy as jnp
from flax import linen as nn
import os
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.shard_map import shard_map
from functools import partial
# print( jax.print_environment_info() )


# simulate 8 CPU devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

devices = jax.local_devices()
print(devices)


def predict(params, inputs):
    for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = nn.relu(outputs)
    return outputs


def loss(params, batch):
    inputs, targets = batch
    predictions = predict(params, inputs)
    return jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))


def init_layer(key, n_in, n_out):
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (n_in, n_out)) / jnp.sqrt(n_in)
    b = jax.random.normal(k2, (n_out,))
    return W, b


def init(key, layer_sizes, batch_size):
    key, *keys = jax.random.split(key, len(layer_sizes))
    params = list(map(init_layer, keys, layer_sizes[:-1], layer_sizes[1:]))

    key, *keys = jax.random.split(key, 3)
    inputs = jax.random.normal(keys[0], (batch_size, layer_sizes[0]))
    targets = jax.random.normal(keys[1], (batch_size, layer_sizes[-1]))

    return params, (inputs, targets)


layer_sizes = [784, 128, 128, 128, 128, 128, 8]
batch_size = 32

params, batch = init(jax.random.key(0), layer_sizes, batch_size)

mesh = Mesh(
    devices,
    axis_names=[
        "batch",
    ],
)

# replicate initial params on all devices, shard data batch over devices
batch = jax.device_put(batch, NamedSharding(mesh, PartitionSpec("batch")))
params = jax.device_put(params, NamedSharding(mesh, PartitionSpec()))


# adapt the loss function to sum the losses across devices
def loss_dp(params, batch):
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=PartitionSpec("batch", None),
        out_specs=PartitionSpec(),
    )
    def loss_spmd(local_batch):
        inputs, targets = local_batch
        predictions = predict(params, inputs)
        local_loss = jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))
        return jax.lax.pmean(local_loss, "batch")

    return loss_spmd(batch)


print(jax.jit(loss)(params, batch))
print(jax.jit(loss_dp)(params, batch))


def allclose(a, b):
    return jax.tree_util.tree_all(
        jax.tree_util.tree_map(partial(jnp.allclose, atol=1e-2, rtol=1e-2), a, b)
    )


print(
    allclose(
        jax.jit(jax.grad(loss))(params, batch),
        jax.jit(jax.grad(loss_dp))(params, batch),
    )
)
