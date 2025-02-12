
import os
import jax
import jax.numpy as jnp
from jax import lax

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from functools import partial

# Use 8 CPU devices
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

mesh1d = Mesh(jax.devices()[:4], ('i',))

@partial(shard_map, mesh=mesh1d, in_specs=P('i'), out_specs=P('i'))
def f1(x_block):
    print('BEFORE:\n', x_block)
    block_sum = jnp.sum(x_block, axis=-1)
    print('Middle:\n', block_sum)
    y_block = jax.lax.pmean(block_sum, 'i')
    print('AFTER:\n', y_block)
    return y_block

x = jax.random.randint(jax.random.PRNGKey(0), (4, 3), minval=0, maxval=4)
print(x)
y = f1(x)
print('FINAL RESULT:\n', y)