import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
import jax
import matplotlib.pyplot as plt

from tqdm import tqdm
import time
import numpy as np
from scheduler import DDPMScheduler
from flax.training import checkpoints


if __name__ == '__main__':
    from unet1d import CondUnet1D
    import einops
    import matplotlib.cm as cm

    scheduler = DDPMScheduler(timesteps=500, seed=20)
    
    model = CondUnet1D(16, 16, 7, basic_channel=28, 
                       channel_scale_factor=(1, 2), 
                       num_groups=14)
    
    log_dir = '/home/yxtang/CodeBase/PythonCourse/PythonRobotics/LearningJax/model_zoo/res/checkpoints/checkpoint_96'
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=log_dir, target=None)
    
    variables = {'params': state_dict['params']}

    jitted_model_apply = jax.jit(model.apply)
    
    noisy_sample = jax.random.normal(random.key(0), shape=(2, 28, 28))

    label_conds = nn.one_hot(jnp.array([5, 8]), num_classes=10)
    
    res1 = []
    res2 = []
    for t in tqdm(range(1, scheduler.timesteps)):
        # predict noise
        k = scheduler.timesteps - t
        timesteps = jnp.array([k, k])

        noise_pred = jitted_model_apply(variables, noisy_sample, timesteps, label_conds, False)
        
        noisy_sample = scheduler.backward_denoising_ddpm(noisy_sample, noise_pred, timesteps)

        noisy_sample = jnp.clip(noisy_sample, -1, 1)

        if k % 25 == 0:
            image = (noisy_sample[0] + 1) * 127.5
            res1.append(image)

            image = (noisy_sample[1] + 1) * 127.5
            res2.append(image)
    
    image1 = np.hstack(res1)
    image2 = np.hstack(res2)
    images = np.vstack([image1, image2])
    plt.imsave(f'/home/yxtang/CodeBase/PythonCourse/PythonRobotics/LearningJax/model_zoo/res/result.png', images, cmap=cm.gray)
    

    
    

    