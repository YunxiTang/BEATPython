import jax
import flax.linen as nn
from typing import Callable
import jax.numpy as jnp
import einops



def scaled_dot_product(q, k, v):
    d_k = q.shape[-1]
    k_transpose = einops.rearrange(k, 'batch nhead seqlen dim -> batch nhead dim seqlen')
    
    # attn_logits in shape of [batch, nhead, seqlen, seqlen]
    attn_logits = jnp.matmul(q, k_transpose) 
    attn_logits = attn_logits / jnp.sqrt(d_k)

    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention



class MHSA(nn.Module):
    '''
        multi head self attention implementation
    '''
    embed_dim: int
    nhead: int = 2
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.zeros

    def setup(self):
        self.qkv_proj = nn.Dense(3*self.embed_dim, 
                                 kernel_init=self.kernel_init,
                                 bias_init=self.bias_init)
        self.o_proj = nn.Dense(self.embed_dim, 
                               kernel_init=self.kernel_init,
                               bias_init=self.bias_init)

    def __call__(self, x):
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv_proj(x)

        # separate Q, K, V from linear output
        qkv = jnp.reshape(qkv, (batch_size, seq_len, self.nhead, -1))
        # [batch, nhead, seqlen, 3*dim]
        qkv = einops.rearrange(qkv, 'b l nh d -> b nh l d')
        # [batch, nhead, seqlen, dim]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        values, attention = scaled_dot_product(q, k, v)
        
        values = einops.rearrange(values, 'batch nhead seqlen dim -> batch seqlen nhead dim')
        values = values.reshape(batch_size, seq_len, embed_dim)
        output = self.o_proj(values)
        return output, attention
    

if __name__ == '__main__':

    batch_size = 5
    seq_len = 10
    embed_dim = 128

    model = MHSA(embed_dim=128, nhead=4)

    sample = jax.random.normal(jax.random.key(0), [batch_size, seq_len, embed_dim])
    outputs, variables = model.init_with_output({'params': jax.random.PRNGKey(12)}, sample)
    
    output, attention = outputs
    print(output.shape, attention.shape)
    
    for key, val in variables.items():
        print(key)
    
    print(variables.get('batch_stats', {}))