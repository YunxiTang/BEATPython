import jax
import flax.linen as nn
from typing import Callable, Tuple
import jax.numpy as jnp
import einops
from flax.linen.linear import PaddingLike


class Mish(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x * nn.tanh(nn.softplus(x))


class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> Mish"""

    out_channels: int
    kernel_size: Tuple
    stride: Tuple
    padding: PaddingLike = "SAME"
    ngroup: int = 8

    def setup(self):
        self.conv = nn.Conv(
            self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
        )
        self.gn = nn.GroupNorm(self.ngroup)
        self.mish = Mish()

    def __call__(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.mish(x)
        return x


class DownSample(nn.Module):
    """convolutional downsample"""

    out_channels: int
    kernel_size: Tuple

    def setup(self):
        self.conv = nn.Conv(
            self.out_channels,
            kernel_size=self.kernel_size,
            strides=(2,) * len(self.kernel_size),
            padding="SAME",
        )

    def __call__(self, x):
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    """convolutional upsample"""

    out_channels: int
    kernel_size: Tuple

    def setup(self):
        self.conv_trans = nn.ConvTranspose(
            self.out_channels,
            kernel_size=self.kernel_size,
            strides=(2,) * len(self.kernel_size),
            padding="SAME",
        )

    def __call__(self, x):
        return self.conv_trans(x)


if __name__ == "__main__":
    x = jax.random.normal(jax.random.key(0), shape=[2, 28, 28, 28, 3])

    model = ConvBlock(64, (3, 3, 3), (1, 1, 1), "SAME", 4)
    out, variables = model.init_with_output({"params": jax.random.PRNGKey(0)}, x)
    print(x.shape, out.shape)

    model_ds = DownSample(64, kernel_size=(4, 4, 4))
    out1, variables = model_ds.init_with_output({"params": jax.random.PRNGKey(10)}, out)
    print(out.shape, out1.shape)

    model_us = UpSample(32, kernel_size=(5, 5, 5))
    out2, variables = model_us.init_with_output(
        {"params": jax.random.PRNGKey(10)}, out1
    )
    print(out1.shape, out2.shape)

    mish = Mish().bind({})
    res = mish(out2)
    print(res.shape)
