import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import os
from jax.experimental.shard_map import shard_map

# Use 8 CPU devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
devices = jax.devices()
mesh = Mesh(jax.devices(), axis_names="batch")

in_shardings = NamedSharding(
    mesh,
    PartitionSpec(
        "batch",
    ),
)
out_shardings = NamedSharding(
    mesh,
    PartitionSpec(
        "batch",
    ),
)


def generate_random_fp1(rng):
    x = jax.random.uniform(rng[0], (2, 2))
    return x


def generate_random_fp2():
    axis_idx = jax.lax.axis_index("batch")
    rng = jax.random.key(axis_idx)
    rng, _ = jax.random.split(rng, 2)
    x = jax.random.uniform(rng, (2, 2))
    return x


shard_fp = shard_map(
    generate_random_fp1,
    mesh,
    in_specs=PartitionSpec("batch"),
    out_specs=PartitionSpec("batch"),
)

rng = jax.random.key(30)
print(rng)
rngs = jax.random.split(rng, 8)
res1 = shard_fp(rngs)

jitted_fp = jax.jit(generate_random_fp2, in_shardings=None, out_shardings=out_shardings)
print(res1.shape)
jitted_fp()
