import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import shard_map
from flax.training import train_state
from functools import partial
import optax
import os
from jax.sharding import Mesh

# Use 8 CPU devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
devices = jax.devices()
mesh = Mesh(jax.devices(), axis_names="batch")


# Step 1: Define a Simple Model
class SimpleModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Dense(self.features)(x)
        x = nn.BatchNorm()(x, use_running_average=training)
        return x


# Initialize the model and optimizer
model = SimpleModel(features=10)
params = model.init(
    jax.random.PRNGKey(0), jnp.ones((1, 5))
)  # Single example input for parameter initialization

# Define an optimizer and wrap it in `train_state`
state = train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=optax.sgd(learning_rate=0.001)
)


# Step 2: Define the Loss Function
def compute_loss(params, batch):
    logits = model.apply(params, batch["inputs"])
    loss = jnp.mean((logits - batch["targets"]) ** 2)  # Mean Squared Error
    return loss


# Step 3: Define the Shard Function for `shard_map`
def shard_fn(state, batch):
    """Per-shard training function."""
    loss, grads = jax.value_and_grad(compute_loss)(state.params, batch)
    return loss, grads


# Define sharding specifications for inputs and outputs
in_shardings = (jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec("batch"))
out_shardings = (jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec())

shard_val_grad = shard_map.shard_map(
    shard_fn, mesh, in_specs=in_shardings, out_specs=out_shardings, check_rep=False
)


# Step 4: Training Step with `shard_map`
@jax.jit
def train_step(state, batch):
    # Use `shard_map` to compute loss and gradients in parallel
    loss, grads = shard_val_grad(state, batch)

    # Update parameters by applying gradients
    state = state.apply_gradients(grads=grads)
    return state, loss


# Dummy batch data for multi-device setup
batch = {
    "inputs": jnp.ones((32, 5)),  # Batch divided across devices
    "targets": jnp.ones((32, 10)),
}

# Step 5: Training Loop
for epoch in range(10):  # Example: 10 epochs
    state, loss = train_step(state, batch)
    print(f"Epoch {epoch}, Loss: {loss}")
