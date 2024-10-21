import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional
from einops.einops import rearrange


class MultiHeadCrossAttention(nn.Module):
    nhead: int
    embed_dim: int
    dropout_rate: float
    
    def setup(self):
        assert self.embed_dim % self.nhead == 0
        self.head_dim = self.embed_dim // self.nhead
        
        # Linear layers for query, key, and value
        self.k_proj = nn.Dense(self.embed_dim)
        self.v_proj = nn.Dense(self.embed_dim)
        self.q_proj = nn.Dense(self.embed_dim)
        
        # output projection
        self.out_proj = nn.Dense(self.embed_dim)
        
        # dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def _spli_heads(self, x:jnp.ndarray):
        """
            Split the last dimension into (num_heads, head_dim) and transpose.
        """
        batch_size, seq_len, embed_dim = x.shape
        x = x.reshape(batch_size, seq_len, self.nhead, self.head_dim)
        x = rearrange(x, 'bs sl nh hd -> bs nh sl hd')
        return x
    
    def __call__(self, query, key, value, mask=None, deterministic: bool = False):
        Q = self.q_proj(query)  # (batch_size, query_len, embed_dim)
        K = self.k_proj(key)    # (batch_size, key_len, embed_dim)
        V = self.v_proj(value)
        
        # Split heads for multi-head attention
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        
if __name__ == '__main__':
    from pprint import pprint
    model = MultiHeadCrossAttention(4, 64, 0.1)
    model2 = nn.MultiHeadAttention(4, qkv_features=16, out_features=16)
    
    datas = jax.random.uniform(jax.random.PRNGKey(0), (3, 2, 3, 3))
    print(datas.shape)
    
    output, variables = model2.init_with_output(jax.random.PRNGKey(1), datas[0], datas[1], datas[2])
    print(output.shape)
    
    pprint(variables)