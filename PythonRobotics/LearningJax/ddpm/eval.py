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
from flax.training import train_state, checkpoints


if __name__ == '__main__':
    from unet1d import CondUnet1D
    import einops
    import matplotlib.cm as cm

    scheduler = DDPMScheduler(timesteps=500, seed=0)
    
    model = CondUnet1D(32, 16, 15, basic_channel=1, 
                       channel_scale_factor=(1, 2, 4), 
                       num_groups=1)
    log_dir = '/Users/y.xtang/Documents/ML/JAX_script/deep_learning_jax/checkpoints/checkpoint_5'
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=log_dir, target=None)
    
    bind_model = model.bind({'params': state_dict['params'], 'batch_stats': state_dict['batch_stats']})
    
    noisy_sample = jax.random.normal(random.key(0), shape=(1, 28*28, 1))
    label_conds = nn.one_hot(jnp.array([2]), num_classes=10)
    
    rng_key = random.key(123)
    
    res = []
    for k in reversed(range(1, scheduler.timesteps)):
        # predict noise
        timesteps = jnp.array([k])
        tc = time.time()
        noise_pred = x = jnp.clip(bind_model(noisy_sample, timesteps, label_conds, False), -1.0, 1.0) 
        print(time.time() - tc)
        noisy_sample = 1. / jnp.sqrt(scheduler.alpha[k]) * (noisy_sample - 
                                                            (1 - scheduler.alpha[k])/(jnp.sqrt(1-scheduler.alpha_bar[k])) * noise_pred
                                                            )

        if k > 1:
            var = jnp.sqrt(scheduler.betas[k])
            noise = random.normal(rng_key, shape=(1, 28*28, 1))
            noisy_sample = noisy_sample + var * noise
        else:
            noisy_sample = noisy_sample + 0 * noise

        if k % 50 == 0:
            print(k)
            image = (jnp.reshape(jnp.squeeze(noisy_sample), [28, 28]) + 1) * 127.5
            res.append(image)
        rng_key, rng_key2 = random.split(rng_key, 2) 
    
    images = np.hstack(res)
    plt.imsave(f'/Users/y.xtang/Documents/ML/JAX_script/deep_learning_jax/ddpm/res/result.png', images, cmap=cm.gray)
    

    
    

    