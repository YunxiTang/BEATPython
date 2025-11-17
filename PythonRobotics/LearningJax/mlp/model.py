from typing import Any
import flax.linen as nn
import jax


class MLP(nn.Module):
    inp_dim: int = 1
    out_dim: int = 1

    def setup(self):
        self.fc1 = nn.Dense(features=126)
        self.dropout = nn.Dropout(0.2)
        # self.batch_norm = nn.BatchNorm()
        self.fc2 = nn.Dense(features=self.out_dim)

    def __call__(self, x, train: False):
        x = self.fc1(x)
        x = self.dropout(x, deterministic=not train)
        # x = self.batch_norm(x, use_running_average=not train)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    import jax.numpy as jnp

    model = MLP(1, 1)
    variables = model.init(
        {"params": jax.random.PRNGKey(1), "dropout": jax.random.PRNGKey(0)},
        jnp.array([[1.0]]),
        False,
    )
    res = model.apply(
        variables, jnp.array([[3.0]]), True, rngs={"dropout": jax.random.PRNGKey(3)}
    )
    print(variables.keys())
    print(res)
