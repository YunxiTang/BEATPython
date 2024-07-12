from typing import Any
import jax
import flax.linen as nn
import jax.numpy as jnp
from conv1d_models import Conv1DBlock, Mish, DownSample1D, UpSample1D
from positional_embedding import SinusoidalEmbedding


class CondResConv1D(nn.Module):
    '''
        conditional residual conv1d block
    '''
    out_channels: int
    kernel_size: int
    ngroup: int = 8

    def setup(self):
        # residual conv layer 1
        self.conv1d_1 = Conv1DBlock(self.out_channels, 
                                    self.kernel_size, 
                                    stride=1, 
                                    padding='SAME', 
                                    ngroup=self.ngroup)
        # residual conv layer 2
        self.conv1d_2 = Conv1DBlock(self.out_channels, 
                                    self.kernel_size, 
                                    stride=1, 
                                    padding='SAME', 
                                    ngroup=self.ngroup)
        
        # residual conv layer
        self.res_conv1d = nn.Conv(self.out_channels, kernel_size=(1,))

        # conditional encoder
        cond_channels = 2 * self.out_channels
        self.cond_encoder = nn.Sequential([Mish(), nn.Dense(cond_channels)])
        

    def __call__(self, x, cond):
        '''
            x : [ batch_size x seq_len x in_channels ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x seq_len x out_channels]
        '''
        out = self.conv1d_1(x) # [batch_size x seq_len x out_channels]

        condition_embed = self.cond_encoder(cond) # [ batch_size x 2*out_channels]
        embed = jnp.reshape(condition_embed, (condition_embed.shape[0], 2, 1, self.out_channels))

        scale = embed[:,0,...] # [ batch_size x 1 x out_channels]
        bias = embed[:,1,...]
        out = scale * out + bias
        out = self.conv1d_2(out)
        out = out + self.res_conv1d(x)
        return out
    

class CondUnet1D(nn.Module):
    diffusion_step_embed_dim: int
    condition_embed_dim: int
    kernel_size: int
    basic_channel: int = 128
    channel_scale_factor: tuple = (1, 2, 4, 8)
    num_groups: int = 8

    @nn.compact
    def __call__(self, x, diffustion_step, condition):
        """
            x: (batch_size, seq_len, input_dim)
            diffustion_step: (batch_size, 1)
            condition: (batch_size, cond_dim)
            output: (batch_size, seq_len, input_dim)
        """
        diffustion_step_embed = nn.Sequential([SinusoidalEmbedding(self.diffusion_step_embed_dim),
                                               nn.Dense(2*self.diffusion_step_embed_dim), Mish(),
                                               nn.Dense(self.diffusion_step_embed_dim)])(diffustion_step)
        
        condition_embed = nn.Sequential([nn.Dense(2*self.condition_embed_dim), Mish(),
                                        nn.Dense(self.condition_embed_dim)])(condition)
        
        global_cond_embed = jnp.concatenate([diffustion_step_embed, condition_embed], axis=-1)
        
        channels = [self.basic_channel * i for i in self.channel_scale_factor]

        # Downsampling phase
        pre_downsampling = []
        for down_index, down_channel in enumerate(channels):
            x = CondResConv1D(down_channel, self.kernel_size, self.num_groups)(x, global_cond_embed)
            x = CondResConv1D(down_channel, self.kernel_size, self.num_groups)(x, global_cond_embed)
            x = DownSample1D(down_channel)(x)
            pre_downsampling.append(x)

        # Middle block
        mid_channel = down_channel
        x = CondResConv1D(mid_channel, self.kernel_size, self.num_groups)(x, global_cond_embed)
        x = CondResConv1D(mid_channel, self.kernel_size, self.num_groups)(x, global_cond_embed)

        # Upsampling phase
        for up_index, up_channel in enumerate(reversed(channels)):
            residual = pre_downsampling.pop()
            x = jnp.concatenate([x, residual], -1)
            x = CondResConv1D(up_channel, self.kernel_size, self.num_groups)(x, global_cond_embed)
            x = CondResConv1D(up_channel, self.kernel_size, self.num_groups)(x, global_cond_embed)
            x = UpSample1D(up_channel)(x)
        x = CondResConv1D(up_channel, self.kernel_size, self.num_groups)(x, global_cond_embed)
        return x


if __name__ == '__main__':
    import time
    batch_size = 2 
    seq_len = 5 
    channels = 5

    x = jnp.ones([batch_size, seq_len, channels])
    y = jnp.ones([batch_size, 1, channels]) * 2
    z = x + y

    print( x.shape, y.shape )
    print( z.shape )
    print( z )

    x = jnp.ones([batch_size, seq_len, channels])
    cond = jnp.ones([batch_size, 12])
    model = CondResConv1D(64, 3)
    output, variables = model.init_with_output({'params': jax.random.PRNGKey(0)}, x, cond)
    print(output.shape)

    sample = jnp.ones([4, seq_len, channels])
    cond = jnp.ones([4, 12])
    res = model.apply(variables, sample, cond)
    print(res.shape)

    print('=================================')
    seq_len = 64
    channels = 16
    sample = jnp.ones([4, seq_len, channels])
    cond = jnp.ones([4, 12])
    diff_step = jnp.array([1, 2, 3, 4])

    model = CondUnet1D(64, 64, 3, basic_channel=channels, channel_scale_factor=(1, 2, 4, 8), num_groups=4)

    print(sample.shape)
    print('---')
    tc = time.time()
    output, variables = model.init_with_output({'params': jax.random.PRNGKey(0)}, 
                                                sample, diff_step, cond)
    e_t = time.time() - tc
    print(e_t)
    print('---')
    print(output.shape)

    tc = time.time()
    output = model.apply(variables, sample, diff_step, cond)
    e_t = time.time() - tc
    print(e_t)
    