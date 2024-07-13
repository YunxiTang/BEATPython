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
from scheduler import DDPMScheduler
from utils import FlaxTrainer


if __name__ == '__main__':
    from unet1d import CondUnet1D
    import einops

    ds = load_dataset("ylecun/mnist", cache_dir='/Users/y.xtang/Documents/ML/JAX_script/dataset')
    ds.set_format('jax')
    train_ds = ds['train']
    test_ds = ds['test']

    scheduler = DDPMScheduler(timesteps=500, seed=0)
    
    num = 10
    sample_img = train_ds[0:0+num]['image'] / 127.5 - 1
    sample_label = train_ds[0:0+num]['label']

    # sample noise to add to data points
    noises = random.normal(random.key(0), shape=sample_img.shape)

    # sample a diffusion iteration for each data point
    timesteps = random.randint(random.key(0), shape=[sample_img.shape[0],], 
                               minval=0, maxval=scheduler.timesteps)
    
    # forward diffusion process
    noisy_images = scheduler.add_noise(sample_img, noises, timesteps)

    # model test
    label_conds = nn.one_hot(sample_label, num_classes=10)
    
    channel = 1
    noisy_sample = einops.rearrange(noisy_images, 'b h w -> b (h w)')
    noisy_sample = einops.repeat(noisy_sample, 'b s -> b s c', c = channel)
    model = CondUnet1D(32, 16, 15, basic_channel=channel, 
                       channel_scale_factor=(1, 2, 4), 
                       num_groups=1)
    
    trainer = FlaxTrainer(model, noisy_sample, timesteps, label_conds, False)
    trainer.train()
    

    
    

    