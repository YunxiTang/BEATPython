import flax.linen as nn
from jax import random
import jax.numpy as jnp


class Encoder(nn.Module):
    """
        VAE Encoder
    """
    latents: int

    def setup(self):
        self.fc1 = nn.Dense(500)
        self.fc2_mean = nn.Dense(self.latents)
        self.fc3_logvar = nn.Dense(self.latents)

    def __call__(self, x):
        z = self.fc1(x)
        z = nn.relu(z)
        mean = self.fc2_mean(z)
        logvar = self.fc3_logvar(z)
        return mean, logvar
    

class Decoder(nn.Module):
    """
        VAE Decoder
    """

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(500, name='fc1')(z)
        z = nn.relu(z)
        z = nn.Dense(784, name='fc2')(z)
        return z
    

def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


class VAE(nn.Module):
    latents: int = 20

    def setup(self):
        self.encoder = Encoder(self.latents)
        self.decoder = Decoder()

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar
    
    def generate(self, z):
        return nn.sigmoid(self.decoder(z))
    

def create_model(latents):
    return VAE(latents=latents)