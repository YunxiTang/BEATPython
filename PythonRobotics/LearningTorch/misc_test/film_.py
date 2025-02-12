import torch
import time
from einops import repeat
import jax
import numba

@numba.jit
# @torch.jit.script
# @jax.jit
def add(x:int, y:int):
    for i in range(200):
        z = x + y
    return z


if __name__ == '__main__':

    seq_len = 5

    batch_size = 2
    latent_dim = 4

    scale = torch.randint(2, 4, size=[batch_size, latent_dim])
    bias = torch.randint(2, 4, size=[batch_size, latent_dim])

    print('scale:\n ', scale[0])
    print('bias:\n ', bias[0])

    scale = repeat(scale, 'batch d -> batch s d', s=seq_len)
    bias = repeat(bias, 'batch d -> batch s d', s=seq_len)

    print('scale:\n ', scale[0])
    print('bias:\n ', bias[0])

    latent_repre = torch.randint(1, 5, size=[batch_size, seq_len, latent_dim])
    print('latent_repre:\n ', latent_repre[0])

    z = scale * latent_repre  + bias
    print( z[0] )

    ts = time.time()
    for x in range(100000):
        add(x, 15)
    print(time.time()-ts)