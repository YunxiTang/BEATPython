import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from flax.training import train_state
import jax
import matplotlib.pyplot as plt

from typing import Callable
from tqdm.notebook import tqdm
from PIL import Image
import time
from datasets import Dataset, load_dataset
import numpy as np


@jax.jit
def forward_noising(xs, noises, ts, sqrt_alpha_bar, one_minus_sqrt_alpha_bar):
    flatten_ts = jnp.ravel(ts)
    reshaped_sqrt_alpha_bar_t = jnp.expand_dims(jnp.take(sqrt_alpha_bar, flatten_ts), [1,2])
    reshaped_one_minus_sqrt_alpha_bar_t = jnp.expand_dims(jnp.take(one_minus_sqrt_alpha_bar, flatten_ts), [1,2])
    noisy_xs = reshaped_sqrt_alpha_bar_t  * xs + reshaped_one_minus_sqrt_alpha_bar_t  * noises
    return noisy_xs


class DDPMScheduler:
    def __init__(self, timesteps:int, seed:int=0):
        # Defining a constant value for T
        self.timesteps = timesteps

        # Defining beta for all t's in T steps
        self.betas = jnp.linspace(0.0001, 0.02, timesteps)

        self.alpha = 1 - self.betas
        self.alpha_bar = jnp.cumprod(self.alpha, 0)
        self.alpha_bar = jnp.concatenate((jnp.array([1.]), self.alpha_bar[:-1]), axis=0)
        self.sqrt_alpha_bar = jnp.sqrt(self.alpha_bar)
        self.one_minus_sqrt_alpha_bar = jnp.sqrt(1 - self.alpha_bar)

        self.key = random.PRNGKey(seed)


    def add_noise(self, xs, noises, steps):
        noisy_xs = forward_noising(xs, noises, steps, 
                                   self.sqrt_alpha_bar, 
                                   self.one_minus_sqrt_alpha_bar)
        return noisy_xs



class SinusoidalEmbedding(nn.Module):
    dim: int = 32
    
    @nn.compact
    def __call__(self, inputs):
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = inputs[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
        return emb
    

class TimeEmbedding(nn.Module):
    dim: int = 32
    @nn.compact
    def __call__(self, inputs):
        time_dim = self.dim * 4
        se = SinusoidalEmbedding(self.dim)(inputs)
        # Projecting the embedding into a 128 dimensional space
        x = nn.Dense(time_dim)(se)
        x = nn.gelu(x)
        x = nn.Dense(time_dim)(x)
        return x
    

class MHSAttention(nn.Module):
    dim: int
    num_heads: int = 8
    use_bias: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, inputs):
        batch, h, w, channels = inputs.shape
        inputs = inputs.reshape(batch, h*w, channels)
        batch, n, channels = inputs.shape
        scale = (self.dim // self.num_heads) ** -0.5
        qkv = nn.Dense(self.dim * 3, use_bias=self.use_bias, kernel_init=self.kernel_init)(inputs)
        qkv = jnp.reshape(qkv, (batch, n, 3, self.num_heads, channels // self.num_heads))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attention = nn.softmax(attention, axis=-1)

        x = (attention @ v).swapaxes(1, 2).reshape(batch, n, channels)
        x = nn.Dense(self.dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = jnp.reshape(x, (batch, int(x.shape[1]** 0.5), int(x.shape[1]** 0.5), -1))
        return x
    


class Block(nn.Module):
    dim: int = 32
    groups: int = 8

    @nn.compact
    def __call__(self, inputs):
        conv = nn.Conv(self.dim, (3, 3))(inputs)
        norm = nn.GroupNorm(num_groups=self.groups)(conv)
        activation = nn.silu(norm)
        return activation


class ResnetBlock(nn.Module):
    dim: int = 32
    groups: int = 8

    @nn.compact
    def __call__(self, inputs, time_embed=None):
        x = Block(self.dim, self.groups)(inputs)
        if time_embed is not None:
            time_embed = nn.silu(time_embed)
            time_embed = nn.Dense(self.dim)(time_embed)
            x = jnp.expand_dims(jnp.expand_dims(time_embed, 1), 1) + x
        x = Block(self.dim, self.groups)(x)
        res_conv = nn.Conv(self.dim, (1, 1), padding="SAME")(inputs)
        return x + res_conv
    

class UNet(nn.Module):
    dim: int = 8
    dim_scale_factor: tuple = (1, 2, 4, 8)
    num_groups: int = 8


    @nn.compact
    def __call__(self, inputs):
        inputs, time = inputs
        channels = inputs.shape[-1]
        x = nn.Conv(self.dim // 3 * 2, (7, 7), padding=((3,3), (3,3)))(inputs)
        time_emb = TimeEmbedding(self.dim)(time)
        
        dims = [self.dim * i for i in self.dim_scale_factor]
        pre_downsampling = []
        
        # Downsampling phase
        for index, dim in enumerate(dims):
            x = ResnetBlock(dim, self.num_groups)(x, time_emb)
            x = ResnetBlock(dim, self.num_groups)(x, time_emb)
            att = MHSAttention(dim)(x)
            norm = nn.GroupNorm(self.num_groups)(att)
            x = norm + x
            # Saving this output for residual connection with the upsampling layer
            pre_downsampling.append(x)
            if index != len(dims) - 1:
                x = nn.Conv(dim, (4,4), (2,2))(x)
        
        # Middle block
        x = ResnetBlock(dims[-1], self.num_groups)(x, time_emb)
        att = MHSAttention(dim)(x)
        norm = nn.GroupNorm(self.num_groups)(att)
        x = norm + x 
        x = ResnetBlock(dims[-1], self.num_groups)(x, time_emb)
        
        # Upsampling phase
        for index, dim in enumerate(reversed(dims)):
            x = jnp.concatenate([pre_downsampling.pop(), x], -1)
            x = ResnetBlock(dim, self.num_groups)(x, time_emb)
            x = ResnetBlock(dim, self.num_groups)(x, time_emb)
            att = MHSAttention(dim)(x)
            norm = nn.GroupNorm(self.num_groups)(att)
            x = norm + x
            if index != len(dims) - 1:
                x = nn.ConvTranspose(dim, (4,4), (2,2))(x)


        # Final ResNet block and output convolutional layer
        x = ResnetBlock(dim, self.num_groups)(x, time_emb)
        x = nn.Conv(channels, (1,1), padding="SAME")(x)
        return x



if __name__ == '__main__':
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    NUM_STEPS_PER_EPOCH = 60000//BATCH_SIZE

    ds = load_dataset("ylecun/mnist", cache_dir='/home/yxtang/CodeBase/PythonCourse/dataset')
    ds.set_format('jax')
    train_ds = ds['train']
    test_ds = ds['test']

    scheduler = DDPMScheduler(timesteps=200, seed=0)
    num = 30
    sample_img = train_ds[0:0+num]['image'] / 127.5 - 1

    # sample noise to add to data points
    noises = random.normal(random.key(0), shape=sample_img.shape)

    # sample a diffusion iteration for each data point
    timesteps = random.randint(random.key(0), 
                               shape=[sample_img.shape[0], 1], 
                               minval=10, maxval=scheduler.timesteps)
    
    # forward diffusion process
    noisy_sample = scheduler.add_noise(sample_img, noises, timesteps)
    print(noisy_sample[...,None].shape)

    model = UNet(32)
    variables = model.init(
        {'params': random.PRNGKey(20), 'dropout_rng': random.PRNGKey(30)},
        (noisy_sample[...,None], timesteps)
    )
    print(variables)

    imgs = [(noisy_sample[i] + 1) * 127.5 for i in range(sample_img.shape[0])]
    fig = plt.figure(figsize=(60, 15))
    imgs = np.hstack(imgs)
    plt.imshow(imgs, cmap='gray')
    plt.show()

    