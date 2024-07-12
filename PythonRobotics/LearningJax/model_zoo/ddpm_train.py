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
import sys


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



if __name__ == '__main__':
    from unet1d import CondUnet1D
    import einops

    ds = load_dataset("ylecun/mnist", cache_dir='/home/yxtang/CodeBase/PythonCourse/dataset')
    ds.set_format('jax')
    train_ds = ds['train']
    test_ds = ds['test']

    scheduler = DDPMScheduler(timesteps=200, seed=0)
    num = 10
    sample_img = train_ds[0:0+num]['image'] / 127.5 - 1
    sample_label = train_ds[0:0+num]['label']

    # sample noise to add to data points
    noises = random.normal(random.key(0), shape=sample_img.shape)

    # sample a diffusion iteration for each data point
    timesteps = random.randint(random.key(0), shape=[sample_img.shape[0],], 
                               minval=10, maxval=scheduler.timesteps)
    
    # forward diffusion process
    noisy_images = scheduler.add_noise(sample_img, noises, timesteps)
    imgs = [(noisy_images[i] + 1) * 127.5 for i in range(sample_img.shape[0])]
    fig = plt.figure(figsize=(60, 15))
    imgs = np.hstack(imgs)
    plt.imshow(imgs, cmap='gray')
    plt.show()

    # model test
    label_conds = nn.one_hot(sample_label, num_classes=10)
    channel = 1
    noisy_sample = einops.rearrange(noisy_images, 'b h w -> b (h w)')
    noisy_sample = einops.repeat(noisy_sample, 'b s -> b s c', c = channel)
    model = CondUnet1D(64, 64, 3, basic_channel=channel, 
                       channel_scale_factor=(1, 2, 4, 8), 
                       num_groups=1)
    
    outputs, variables = model.init_with_output(
        {'params': random.PRNGKey(20), 'dropout_rng': random.PRNGKey(30)},
        noisy_sample, timesteps, label_conds)
    
    print(outputs.shape)
    outputs = jnp.reshape(jnp.squeeze(outputs), (num, 28, 28))
    print(outputs.shape)
    
    res = noisy_images - outputs
    imgs = [(res[i] + 1) * 127.5 for i in range(outputs.shape[0])]
    fig = plt.figure(figsize=(60, 15))
    imgs = np.hstack(imgs)
    plt.imshow(imgs, cmap='gray')
    plt.show()

    
    

    