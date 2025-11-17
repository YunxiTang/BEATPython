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
import time
from flax.training.train_state import TrainState
import optax
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    def __init__(self, num_sample: int):
        super().__init__()
        self.num_sample = num_sample
        self.xs = np.linspace(0, 2.0, num_sample)
        self.ys = (
            np.sin(self.xs)
            + 1.2 * self.xs
            + np.random.uniform(-0.02, 0.02, size=(num_sample,))
        )

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        xs = self.xs[index]
        ys = self.ys[index]
        batch = {"x": xs, "y": ys}
        return batch


class RegModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


# Use 8 CPU devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

devices = jax.devices()
mesh = Mesh(jax.devices(), axis_names="batch")
data_sharding = NamedSharding(
    mesh,
    spec=PartitionSpec(
        "batch",
    ),
)
model_sharding = NamedSharding(mesh, spec=PartitionSpec())

model = nn.Dense(12)
params = model.init(jax.random.PRNGKey(0), jnp.arange(2 * 12).reshape(2, 12))

optimizer = optax.adamw(learning_rate=0.0001)
opt_state = optimizer.init(params)

batch_size = 256
batch_x = jax.random.uniform(jax.random.PRNGKey(12), shape=[batch_size, 12])
batch_y = jax.random.uniform(jax.random.PRNGKey(22), shape=[batch_size, 12])
batch = {"x": batch_x, "y": batch_y}

params_dp = jax.device_put(params, model_sharding)

apply_fn = jax.jit(model.apply)
train_state = TrainState(
    step=0, apply_fn=apply_fn, params=params_dp, tx=optimizer, opt_state=opt_state
)

batch_dp = jax.device_put(batch, data_sharding)


def get_loss(params, batch):
    x = batch["x"]
    targets = batch["y"]
    predicts = apply_fn(params, x)
    loss = jnp.mean(optax.l2_loss(predicts, targets))
    return jax.lax.pmean(loss, axis_name="batch")


@jax.jit
def get_loss_sp(params, batch):
    x = batch["x"]
    targets = batch["y"]
    predicts = apply_fn(params, x)
    return jnp.mean(optax.l2_loss(predicts, targets))


loss = get_loss_sp(params_dp, batch_dp)
print(loss)

loss_jit = jax.jit(get_loss)

shard_loss_fn = jax.jit(
    shard_map(
        get_loss,
        mesh,
        in_specs=(
            PartitionSpec(),
            PartitionSpec(
                "batch",
            ),
        ),
        out_specs=PartitionSpec(),
    )
)
loss = shard_loss_fn(params_dp, batch_dp)
print(loss)

get_grads = jax.grad(get_loss)
grad_fn = jax.jit(
    shard_map(
        get_grads,
        mesh,
        in_specs=(PartitionSpec(), PartitionSpec("batch")),
        out_specs=PartitionSpec(),
        check_rep=False,
    )
)

dataloader = DataLoader(DummyDataset(10000), batch_size=512, shuffle=True)
for i in range(10):
    loss_epoch = 0
    for batch in dataloader:
        batch_dp = jax.device_put(batch, data_sharding)
        grads = grad_fn(train_state.params, batch_dp)
        train_state = train_state.apply_gradients(grads=grads)
        loss = shard_loss_fn(train_state.params, batch_dp)
    loss_epoch += loss
    print(loss_epoch)
