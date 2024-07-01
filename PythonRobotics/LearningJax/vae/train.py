import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn


class Encoder(nn.Module):
    latent_dim: int

    def setup(self):
        self.fc1 = nn.Dense(500)
        self.fc_mean = nn.Dense(self.latent_dim)
        self.fc_logvar = nn.Dense(self.latent_dim)
    
    def __call__(self, x):
        x = self.fc1(x)
        x = nn.relu(x)
        mean_x = self.fc_mean(x)
        logvar_x = self.fc_logvar(x)
        return mean_x, logvar_x
    

class Decoder(nn.Module):
    output_dim: int

    def setup(self):
        self.fc1 = nn.Dense(500)
        self.fc2 = nn.Dense(self.output_dim)


    def __call__(self, z):
        z = self.fc1(z)
        z = nn.relu(z)
        z = self.fc2(z)
        return z
    

class VAE(nn.Module):
    latent_dim: int
    output_dim: int

    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.output_dim)

    @staticmethod
    def reparameterize(mean, logvar, rng):
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, logvar.shape)
        return mean + eps * std

    def __call__(self, x):
        # encoder
        z_mean, z_logvar = self.encoder(x)
        rng = self.make_rng('latent_sample')
        # differentiable reparameterization
        z_sample = self.reparameterize(z_mean, z_logvar, rng)
        print(z_sample)
        # decode
        res = self.decoder(z_sample)
        return res, z_mean, z_logvar
    

if __name__ == '__main__':
    model = VAE(64, 784)
    model_init_rng = random.PRNGKey(0)
    dummy_sample = random.normal(model_init_rng, (2, 784))
    model_init_rng, reparam_rng0 = random.split(model_init_rng, 2)
    variables = model.init({'params': model_init_rng,
                            'latent_sample': reparam_rng0}, dummy_sample)
    res1, _, _ = model.apply(variables, dummy_sample, 
                            rngs={'latent_sample': reparam_rng0})
    reparam_rng1, reparam_rng2 = random.split(reparam_rng0, 2)
    res2, _, _ = model.apply(variables, dummy_sample, 
                            rngs={'latent_sample': reparam_rng0})
    
    print(jnp.linalg.norm(res1 - res2))

    

