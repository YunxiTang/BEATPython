import jax.numpy as jnp
from jax import random
import jax


@jax.jit
def forward_noising(xs, noises, ts, sqrt_alpha_bar, one_minus_sqrt_alpha_bar):
    flatten_ts = ts
    reshaped_sqrt_alpha_bar_t = jnp.expand_dims(jnp.take(sqrt_alpha_bar, flatten_ts), [1, 2, 3])
    reshaped_one_minus_sqrt_alpha_bar_t = jnp.expand_dims(jnp.take(one_minus_sqrt_alpha_bar, flatten_ts), [1, 2, 3])
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
        self.alpha_bar = jnp.concatenate([jnp.array([1.]), self.alpha_bar[:-1]], axis=0)
        self.sqrt_alpha_bar = jnp.sqrt(self.alpha_bar)
        self.one_minus_sqrt_alpha_bar = jnp.sqrt(1 - self.alpha_bar)

        self.key = random.PRNGKey(seed)


    def add_noise(self, xs, noises, steps):
        noisy_xs = forward_noising(xs, noises, steps, 
                                   self.sqrt_alpha_bar, self.one_minus_sqrt_alpha_bar)
        return noisy_xs
    

    def backward_denoising_ddpm(self, x_t, pred_noise, t):
        alpha_t = jnp.take(self.alpha, t)[0]
        alpha_t_bar = jnp.take(self.alpha_bar, t)[0]
        var = jnp.take(self.betas, t)[0]
        
        eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** .5  
        mean = 1 / (alpha_t ** 0.5) * (x_t - eps_coef * pred_noise)
        
        z = random.normal(key=self.key, shape=x_t.shape)
        _, self.key = random.split(self.key)
        if t [0] > 0:
            return mean + (var ** 0.5) * z
        else:
            return mean