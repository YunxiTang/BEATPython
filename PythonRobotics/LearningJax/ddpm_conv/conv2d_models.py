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


class Conv2DBlock(nn.Module):
    """
    Conv2d -> GroupNorm -> Mish
    """

    out_channels: int
    kernel_size: Tuple = (3, 3)
    stride: Tuple = (1, 1)
    padding: PaddingLike = "SAME"
    ngroup: int = 8

    def setup(self):
        self.con2d = nn.Conv(
            self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
        )
        self.gn = nn.GroupNorm(self.ngroup)
        self.mish = Mish()

    def __call__(self, x):
        x = self.con2d(x)
        x = self.gn(x)
        x = self.mish(x)
        return x


class DownSample2D(nn.Module):
    out_channels: int

    def setup(self):
        self.conv = nn.Conv(
            self.out_channels, kernel_size=(3, 3), strides=(2, 2), padding="SAME"
        )

    def __call__(self, x):
        x = self.conv(x)
        return x


class UpSample2D(nn.Module):
    out_channels: int

    def setup(self):
        self.conv_trans = nn.ConvTranspose(
            self.out_channels, kernel_size=(4, 4), strides=(2, 2), padding="SAME"
        )

    def __call__(self, x):
        return self.conv_trans(x)


if __name__ == "__main__":
    x = jax.random.normal(jax.random.key(0), shape=[2, 28, 28, 1])
    model = Conv2DBlock(64, (3, 3), (1, 1), "SAME", 4)
    out, variables = model.init_with_output({"params": jax.random.PRNGKey(0)}, x)
    print(x.shape, out.shape)

    model_ds = DownSample2D(1)
    out, variables = model_ds.init_with_output({"params": jax.random.PRNGKey(10)}, x)
    print(x.shape, out.shape)

    model_us = UpSample2D(1)
    out, variables = model_us.init_with_output({"params": jax.random.PRNGKey(10)}, x)
    print(x.shape, out.shape)

    mish = Mish().bind({})
    # print(mish(x))
