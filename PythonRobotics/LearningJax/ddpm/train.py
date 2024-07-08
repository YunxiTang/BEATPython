import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from flax.training import train_state

import jax
import matplotlib.pyplot as plt

from typing import Callable
from tqdm.notebook import tqdm
from PIL import Image

from datasets import Dataset, load_dataset

import numpy as np

NUM_EPOCHS = 10
BATCH_SIZE = 64
NUM_STEPS_PER_EPOCH = 60000//BATCH_SIZE


# Defining a constant value for T
timesteps = 200

# Defining beta for all t's in T steps
beta = jnp.linspace(0.0001, 0.02, timesteps)

alpha = 1 - beta
alpha_bar = jnp.cumprod(alpha, 0)
alpha_bar = jnp.concatenate((jnp.array([1.]), alpha_bar[:-1]), axis=0)
sqrt_alpha_bar = jnp.sqrt(alpha_bar)
one_minus_sqrt_alpha_bar = jnp.sqrt(1 - alpha_bar)


@jax.jit
def forward_noising(key, x_0, t):
    noise = random.normal(key, x_0.shape)  
    reshaped_sqrt_alpha_bar_t = jnp.take(sqrt_alpha_bar, t)
    reshaped_one_minus_sqrt_alpha_bar_t = jnp.take(one_minus_sqrt_alpha_bar, t)
    noisy_image = reshaped_sqrt_alpha_bar_t  * x_0 + reshaped_one_minus_sqrt_alpha_bar_t  * noise
    return noisy_image, noise


if __name__ == '__main__':
    ds = load_dataset("ylecun/mnist", cache_dir='/home/yxtang/CodeBase/PythonCourse/dataset')
    ds.set_format('jax')
    train_ds = ds['train']
    test_ds = ds['test']

    sample_img = train_ds[0]['image'] / 127.5 - 1

    fig = plt.figure(figsize=(60, 15))

    imgs = []
    for index, i in enumerate([10, 50, 100, 185]):
        noisy_im, noise = forward_noising(random.PRNGKey(0), sample_img , jnp.array([i,]))
        imgs.append((noisy_im + 1) * 127.5)
    imgs = np.hstack(imgs)
    plt.imshow(imgs, cmap='gray')
    plt.show()

    