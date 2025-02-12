import flax.linen as nn
import jax.numpy as jnp
import einops
import jax
import numpy as np
import math
from pprint import pprint


class SinusoidalEmbedding(nn.Module):
    dim: int = 32
    
    @nn.compact
    def __call__(self, inputs):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = inputs[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
        return emb



class PositionalEncoding(nn.Module):
    d_model : int         # Hidden dimensionality of the input.
    max_len : int = 5000  # Maximum length of a sequence to expect.

    def setup(self):
        
        def pe_init(d_model, max_len):
            # create place-hold matrix of [SeqLen, HiddenDim] representing PE
            pe = np.zeros((max_len, d_model))
            position = np.arange(0, max_len, dtype=jnp.float32)[:,None]
            div_term = np.exp( (np.log(10000.0) / -d_model) * np.arange(0, d_model, 2) )
            pe[:, 0::2] = np.sin(position * div_term)
            pe[:, 1::2] = np.cos(position * div_term)
            pe = jax.device_put(pe[None])
            
            return pe
        self.pe = self.variable('embeds', 'sin_posemb', pe_init, self.d_model, self.max_len)

    def __call__(self, x):
        x = x + self.pe.value[:, :x.shape[1], :]
        return x
    

if __name__ == '__main__':
    x = jnp.array([2, 20, 4])
    x = einops.repeat(x, 'b -> b c', c=4)[None]
    print(x.shape)
    # pe = SinusoidalEmbedding(128)
    pe = PositionalEncoding(d_model=4, max_len=100)
    output, variables = pe.init_with_output({}, x)
    
    print(output.shape)
    pprint(variables.keys())

    # val = variables.get('embeds')['sin_posemb']
    # print(val.shape)
    
    # out = pe.bind(variables)(x)
    # print(out)

    # print(type(jax.device_get(out)))
    # print(type(jax.device_put(out, device=jax.devices()[0])))