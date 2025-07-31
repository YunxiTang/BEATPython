import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from functools import partial
import optax
import os
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from typing import Any
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils


class TrainState(train_state.TrainState):
    batch_stats: Any = None
    dropout_rng: Any = None


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
        print(f"Data shape: {str(shard.data)}")
        print("=" * 75)
        print("")


# Use 8 CPU devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
devices = jax.devices()
mesh = Mesh(jax.devices(), axis_names="batch")


# Step 1: Define a Simple Model
class SimpleModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, train):
        x = nn.Dense(self.features)(x)
        x = nn.Dropout(0.1)(x, deterministic=not train)
        x = nn.BatchNorm()(x, use_running_average=not train)
        return x


# Initialize the model and optimizer
model = SimpleModel(features=10)
x = jnp.ones((2, 5))

rng_key = jax.random.PRNGKey(0)
params_key, dropout_key, rng_key = jax.random.split(rng_key, 3)
variables = model.init({"params": params_key, "dropout": dropout_key}, x, False)
params = variables.get("params")
batch_stats = variables.get("batch_stats", {})
jax.tree_util.tree_map(lambda x: print(x.devices()), params)
exit()
# Define an optimizer and wrap it in `train_state`
state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.sgd(learning_rate=0.001),
    dropout_rng=dropout_key,
    batch_stats={},
)


# Step 2: Define the Loss Function
def loss_fn(params, state: TrainState, batch, train: bool):
    feats = batch["feat"]
    labels = batch["label"]
    model_variables = {"params": params, "batch_stats": state.batch_stats}
    output = state.apply_fn(
        model_variables,
        feats,
        train,
        rngs={"dropout": state.dropout_rng} if train else None,
        mutable=["batch_stats"] if train else False,
    )
    if train:
        predicts, updated_model_state = output
    else:
        predicts, updated_model_state = output, None

    loss_val = jnp.mean(optax.l2_loss(predicts, labels))  # jnp.mean(predicts, labels)
    return loss_val, updated_model_state


def train_step(state: TrainState, batch: dict, dropout_rng):
    loss_val_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_value, updated_model_state), grads = loss_val_grad_fn(
        state.params, state, batch, train=True
    )

    updated_state = state.apply_gradients(
        grads=grads,
        batch_stats=updated_model_state["batch_stats"],
        dropout_rng=dropout_rng,
    )
    return loss_value, updated_state


# Define sharding specifications for inputs and outputs
in_shardings = (
    NamedSharding(mesh, spec=PartitionSpec()),
    NamedSharding(mesh, spec=PartitionSpec("batch")),
    None,
)
out_shardings = (
    NamedSharding(mesh, spec=PartitionSpec()),
    NamedSharding(mesh, spec=PartitionSpec()),
)


train_step = jax.jit(train_step, in_shardings=in_shardings, out_shardings=out_shardings)
# Dummy batch data for multi-device setup
batch = {
    "feat": jnp.ones((32, 5)),  # Batch divided across devices
    "label": jnp.ones((32, 10)),
}

step_rng = rng_key
# Step 5: Training Loop
for epoch in range(200):
    rng, step_rng = jax.random.split(step_rng)
    loss, state = train_step(state, batch, rng)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")


def vis_state(x):
    if isinstance(x, jax.Array) and len(x.shape) >= 1:
        jax.debug.visualize_array_sharding(x)


jax.tree_util.tree_map(vis_state, state)
