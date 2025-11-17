import jax
import jax.numpy as jnp
from flax import linen as nn


def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits)))
