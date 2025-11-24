from typing import Tuple
import jax
import flax.linen as nn
import jax.numpy as jnp
from conv2d_models import Conv2DBlock, Mish, DownSample2D, UpSample2D
from positional_embedding import SinusoidalEmbedding
from mhsa import MHSA
import einops


class CondResConv2D(nn.Module):
    '''
        conditional residual conv1d block
    '''
    out_channels: int 
    kernel_size: Tuple
    ngroup: int = 8

    def setup(self):
        # conv layer 1
        self.conv2d_1 = Conv2DBlock(self.out_channels, 
                                    self.kernel_size, 
                                    stride=(1, 1), 
                                    padding='SAME', 
                                    ngroup=self.ngroup)
        # conv layer 2
        self.conv2d_2 = Conv2DBlock(self.out_channels, 
                                    self.kernel_size, 
                                    stride=(1, 1),
                                    padding='SAME', 
                                    ngroup=self.ngroup)
        
        # residual conv layer
        self.res_conv2d = nn.Conv(self.out_channels, kernel_size=(1, 1))

        # conditional encoder
        cond_channels = 2 * self.out_channels
        self.cond_encoder = nn.Sequential([Mish(), nn.Dense(cond_channels)])
        

    def __call__(self, x, cond):
        '''
            x : [ batch_size x h x w x in_channels ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x h x w x out_channels]
        '''
        out = self.conv2d_1(x) # [batch_size x h x w x out_channels]

        condition_embed = self.cond_encoder(cond) # [ batch_size x 2*out_channels]
        embed = jnp.reshape(condition_embed, (condition_embed.shape[0], 2, 1, 1, self.out_channels))

        scale = embed[:,0,...] # [ batch_size x 1 x 1 x out_channels]
        bias = embed[:,1,...]
        out = scale * out + bias
        out = self.conv2d_2(out)
        out = out + self.res_conv2d(x)
        return out
    

class PatchEmbedding(nn.Module):
    img_size: int
    patch_size: int
    num_hiddens: int

    def setup(self):
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(self.img_size), _make_tuple(self.patch_size)
        self.conv = nn.Conv(self.num_hiddens, 
                            kernel_size=patch_size,
                            strides=patch_size, padding=0)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

    def __call__(self, x):
        # output shape: (batch size, no. of patches, no. of channels)
        x = self.conv(x)
        x = x.reshape((x.shape[0], self.num_patches, x.shape[3]))
        return x



class CondUnet2D(nn.Module):
    diffusion_step_embed_dim: int
    condition_embed_dim: int
    in_channel: int
    kernel_size: Tuple = (3, 3)
    basic_channel: int = 128
    channel_scale_factor: tuple = (2, 4, 8)
    num_groups: int = 8


    @nn.compact
    def __call__(self, x, diffustion_step, condition, train):
        """
            x: (batch_size, seq_len, input_dim)
            diffustion_step: (batch_size, )
            condition: (batch_size, cond_dim)
            output: (batch_size, seq_len, input_dim)
        """
        # assemble condition embedding
        diffustion_step_embed = nn.Sequential([SinusoidalEmbedding(self.diffusion_step_embed_dim),
                                               nn.Dense(2*self.diffusion_step_embed_dim), Mish(),
                                               nn.Dense(self.diffusion_step_embed_dim)])(diffustion_step)
        
        condition_embed = nn.Sequential([nn.Dense(2*self.condition_embed_dim), Mish(),
                                         nn.Dense(self.condition_embed_dim)])(condition)
        
        global_cond_embed = jnp.concatenate([diffustion_step_embed, condition_embed], axis=-1)
        
        channels = [self.basic_channel * i for i in self.channel_scale_factor]

        # first conv layer
        x = CondResConv2D(self.basic_channel, self.kernel_size, self.num_groups)(x, global_cond_embed)
        
        # downsampling phase
        pre_downsampling = []
        for down_index, down_channel in enumerate(channels):
            x = CondResConv2D(down_channel, self.kernel_size, self.num_groups)(x, global_cond_embed)
            x = CondResConv2D(down_channel, self.kernel_size, self.num_groups)(x, global_cond_embed)

            x = DownSample2D(down_channel)(x)
            pre_downsampling.append(x)

        # middle block
        mid_channel = down_channel
        x = CondResConv2D(mid_channel, self.kernel_size, self.num_groups)(x, global_cond_embed)
        x = CondResConv2D(mid_channel, self.kernel_size, self.num_groups)(x, global_cond_embed)

        # upsampling phase
        for up_index, up_channel in enumerate(reversed(channels)):
            residual = pre_downsampling.pop()
            x = jnp.concatenate([x, residual], -1)
            x = CondResConv2D(up_channel, self.kernel_size, self.num_groups)(x, global_cond_embed)
            x = CondResConv2D(up_channel, self.kernel_size, self.num_groups)(x, global_cond_embed)
            x = UpSample2D(up_channel)(x)

        # last conv layer
        num_ng = max(1, self.in_channel // 2)
        x = CondResConv2D(self.in_channel, self.kernel_size, num_ng)(x, global_cond_embed)
        return x


if __name__ == '__main__':
    import time
    batch_size = 2 
    h = 28
    w = 28
    channels = 16

    x = jnp.ones([batch_size, h, w, channels])
    y = jnp.ones([batch_size, 1, 1, channels]) * 2
    z = x + y

    print( x.shape, y.shape )
    print( z.shape )

    x = jnp.ones([batch_size, h, w, channels])
    cond = jnp.ones([batch_size, 12])
    model = CondResConv2D(32, (5, 5))
    output, variables = model.init_with_output({'params': jax.random.PRNGKey(0)}, x, cond)
    print(x.shape, output.shape)

    print('=================================')
    h = 28
    w = 28
    channels = 1
    sample = jnp.ones([4, h, w, channels])
    cond = jnp.ones([4, 12])
    diff_step = jnp.array([1, 2, 3, 4])

    model = CondUnet2D(64, 64, in_channel=channels, kernel_size=(3, 3), 
                       basic_channel=16, channel_scale_factor=(2, 4), num_groups=8)

    print('input_shape:', sample.shape)
    print('---')
    tc = time.time()
    output, variables = model.init_with_output({'params': jax.random.PRNGKey(0)}, 
                                                sample, diff_step, cond, False)
    et = time.time() - tc
    print('---')
    print('output_shape:', output.shape)
    print(et)


    jitted_apply = jax.jit(model.apply)
    tc = time.time()
    output = jitted_apply(variables, sample, diff_step, cond, False)
    e_t = time.time() - tc
    print(e_t)

    tc = time.time()
    output = jitted_apply(variables, sample, diff_step, cond, False)
    e_t = time.time() - tc
    print(e_t)

    print('=====================')
    pe = PatchEmbedding(28, 7, 64)
    outputs, variables = pe.init_with_output({'params': jax.random.PRNGKey(0)}, x)
    print('output_shape:', outputs.shape)

    