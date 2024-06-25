from typing import Any
import jax
import flax.linen as nn
import jax.numpy as jnp


class FullModel(nn.Module):
    latent_dim: int = 64

    def setup(self):
        self.fc1 = nn.Dense(self.latent_dim * 2)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Dense(self.latent_dim * 2)
        self.batchnorm1 = nn.BatchNorm()
        self.fc3 = nn.Dense(10)

    def __call__(self, x, train: bool):
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.dropout1(x, deterministic=not train)
        x = self.fc2(x)
        x = self.batchnorm1(x, use_running_average=not train)
        x = nn.gelu(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    rng = jax.random.PRNGKey(1)
    model_init_rng, rng = jax.random.split(rng)
    dummy_inp = jax.random.normal(model_init_rng, [2, 15])
    full_model = FullModel(64)
    variables = full_model.init({'params': model_init_rng, 'dropout': rng}, dummy_inp, False)
    print(variables.keys())
    
    # for training
    y0, updates = full_model.apply(variables, dummy_inp, True, rngs={'dropout': rng}, mutable=['batch_stats'])
    print(updates)
    # for eval
    y1 = full_model.apply(variables, dummy_inp, False)
    print(y0 - y1)