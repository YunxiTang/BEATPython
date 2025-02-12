import flax.linen as nn
import jax.numpy as jnp
import jax


class Linear(nn.Module):
    input_dim: int
    output_dim: int
    
    def setup(self):
        def init_func(rng, input_dim, output_dim):
            res = jax.random.uniform(rng, (input_dim, output_dim))
            return jax.device_put(res)
        
        def init_bias_func(rng, output_dim):
            res = jax.random.uniform(rng, (output_dim,))
            return jax.device_put(res)
        
        self.weight = self.param('weight', init_func, self.input_dim, self.output_dim)
        self.bias = self.param('bias', init_bias_func, self.output_dim)

    def __call__(self, x):
        x = x @ self.weight + self.bias
        return x
    

if __name__ == '__main__':
    x = jax.random.uniform(jax.random.key(12), (2, 12))
    model = Linear(12, 24)
    outputs, variables = model.init_with_output({'params': jax.random.PRNGKey(0),}, x)

    print(outputs.shape)
    print(variables)