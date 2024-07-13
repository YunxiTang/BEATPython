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
    from PythonRobotics.LearningJax.ddpm_conv.unet2d import CondUnet1D
    import einops

    ds = load_dataset("ylecun/mnist", cache_dir='/home/yxtang/CodeBase/PythonCourse/dataset')
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
    
    channel = 28
    noisy_sample = noisy_images

    model = CondUnet1D(16, 16, 7, basic_channel=channel, 
                       channel_scale_factor=(1, 2), 
                       num_groups=14)
    
    trainer = FlaxTrainer(model, noisy_sample, timesteps, label_conds, False)
    trainer.train()
    

    
    

    