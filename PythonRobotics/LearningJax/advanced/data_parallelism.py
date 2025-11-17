"""
Replicate a single model on different devices,
where each device processes different batches of data and their results are merged.
It can be synchronous or asynchronous.
"""

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


def get_sharding_details(sharded_data):
    print("\nSharding Layout:")

    # a utility to visualize the sharding
    jax.debug.visualize_array_sharding(sharded_data)

    print("\nSharding Layout details:")

    # get detailed information for each shard
    for i, shard in enumerate(sharded_data.global_shards):
        print(f"Shard No.: {i:>5}")
        print(f"Device: {str(shard.device):>5}")
        print(f"Data shape: {str(shard.data.shape):>8}")
        print(f"Data slices: {str(shard.index):>22}\n")
        print("=" * 75)
        print("")


def compare_shards_data(shard1, shard2):
    """Compare two shards."""
    data1 = np.asarray(shard1.data)
    data2 = np.asarray(shard2.data)
    np.testing.assert_array_equal(data1, data2)


def pairwise_shards_comparison(shards):
    for shard1, shard2 in zip(shards[:-1], shards[1:]):
        compare_shards_data(shard1, shard2)


def single_axis_sharding():
    num_devices = jax.local_device_count()
    devices = jax.local_devices()
    print(num_devices)
    print(devices)

    devices_array = mesh_utils.create_device_mesh((num_devices,), devices)
    print(f"Device Array: \n\n{devices_array}\n")
    # access the elements of this device array like any other ndarray
    print("Accessing the first device: ", devices[0])

    # Create a mesh from the device array
    mesh = Mesh(devices_array, axis_names=("ax"))

    # Define sharding with a partiton spec
    # If (None, ''), means that shards are replicated over the mesh dimension
    # sharding = NamedSharding(mesh, PartitionSpec("ax"))
    sharding = NamedSharding(
        mesh,
        PartitionSpec(
            None,
        ),
    )

    data = jax.random.normal(jax.random.PRNGKey(0), (8, 3))
    sharded_data = jax.device_put(data, sharding)

    print(f"Data  shape: {data.shape}")
    print(f"Shard shape: {sharding.shard_shape(data.shape)}")
    get_sharding_details(sharded_data)

    pairwise_shards_comparison(sharded_data.global_shards)

    # print(sharded_data.global_shards[0].data)
    # print(sharded_data.global_shards[1].data)


def multi_axis_sharding():
    devices = mesh_utils.create_device_mesh((4, 2), jax.local_devices())
    mesh = Mesh(devices, axis_names=("ax1", "ax2"))
    # p = PartitionSpec('ax1', 'ax2')
    p = PartitionSpec("ax1", None)
    sharding = NamedSharding(mesh, spec=p)
    print(f"Number of logical devices: {len(devices)}")
    print(f"Shape of device array    : {devices.shape}")
    print(f"\n{mesh}")
    print(f"\n{sharding}\n\n")

    data = jax.random.normal(jax.random.PRNGKey(0), (512, 2))
    sharded_data = jax.device_put(data, sharding)
    print(f"Data  shape: {data.shape}")
    print(f"Shard shape: {sharding.shard_shape(data.shape)}")

    get_sharding_details(sharded_data)

    # print(np.asarray(sharded_data.global_shards[0].data) - np.asarray(sharded_data.global_shards[1].data))


if __name__ == "__main__":
    multi_axis_sharding()
