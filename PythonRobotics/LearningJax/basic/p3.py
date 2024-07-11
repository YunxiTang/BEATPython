import flax.linen as nn
import jax.numpy as jnp
import jax


class Model(nn.Module):
    dim_x: int = 12
    dim_y: int = 2
    
    def setup(self):
        def init_func(dim_x, dim_y):
            res = jnp.zeros([dim_x, dim_y])
            return jax.device_put(res)
        
        self.weight = self.param('weight', init_func, (self.dim_x, self.dim_y))

    def __call__(self, x):
        x = self.weight[1, 1] @ x
        return x
    

if __name__ == '__main__':
    x = jax.random.uniform(jax.random.key(12), (2, 12))[None]
    model = Model()
    outputs, variables = model.init_with_output({'params': jax.random.PRNGKey(0),}, x)
    print(variables)