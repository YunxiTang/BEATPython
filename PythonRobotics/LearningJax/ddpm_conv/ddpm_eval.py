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
from PIL import Image


def save_gif(img_list, path=""):
    # Transform images from [-1,1] to [0, 255]
    imgs = (Image.fromarray(i) for i in img_list)

    # Extract first image from iterator
    img = next(imgs)  

    # Append the other images and save as GIF
    img.save(fp=path, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)


if __name__ == '__main__':
    from unet2d import CondUnet2D
    import einops
    import matplotlib.cm as cm

    scheduler = DDPMScheduler(timesteps=500, seed=90)
    
    model = CondUnet2D(64, 64, 
                       in_channel=1, 
                       kernel_size=(3, 3),
                       basic_channel=16, 
                       channel_scale_factor=(4, 8), 
                       num_groups=8)
    
    log_dir = '/home/yxtang/CodeBase/PythonCourse/PythonRobotics/LearningJax/ddpm_conv/res/checkpoints/checkpoint_48'
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=log_dir, target=None)
    variables = {'params': state_dict['params']}
    jitted_model_apply = jax.jit(model.apply)

    num = 10
    noisy_sample = jax.random.normal(random.key(960), shape=(num, 28, 28, 1))
    # digits = jax.random.randint(random.key(104), (num,), 0, 9)
    digits = jnp.array([6,] * num)
    label_conds = nn.one_hot(digits, num_classes=10)
    
    res = []
    for t in tqdm(range(1, scheduler.timesteps, 1)):
        k = scheduler.timesteps - t
        timesteps = jnp.array([k,] * num)
        noise_pred = jitted_model_apply(variables, noisy_sample, timesteps, label_conds, False) # predict noise
        
        noisy_sample = scheduler.backward_denoising_ddpm(noisy_sample, noise_pred, timesteps)
        noisy_sample = jnp.clip(noisy_sample, -1, 1)

        if t % 30 == 0 or (t % 10 == 0 and t > 400) or t == scheduler.timesteps-1:
            tmp = []
            for i in range(num):
                image = (noisy_sample[i,...,0] + 1) * 127.5
                tmp.append(image)
            tmp_image = np.hstack(tmp)
            res.append(tmp_image)
    images = np.vstack(res)
    plt.imsave(f'/home/yxtang/CodeBase/PythonCourse/PythonRobotics/LearningJax/ddpm_conv/res/result.png', images, cmap=cm.gray)
    save_gif( res,
             f'/home/yxtang/CodeBase/PythonCourse/PythonRobotics/LearningJax/ddpm_conv/res/result.gif')

    
    

    