from omegaconf import OmegaConf
from common.base_trainer import BaseTrainer, TrainState
from unet2d import CondUnet2D
import jax.numpy as jnp
import jax
from jax import random
import flax.linen as nn


class DDPMTrainer(BaseTrainer):
    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        self.model = CondUnet2D(64, 64, 
                                in_channel=1, 
                                kernel_size=(5, 5),
                                basic_channel=16, 
                                channel_scale_factor=(4, 8), 
                                num_groups=8)
        
    def create_function(self):
        # loss function
        def loss_fn(params, state: TrainState, 
                    noises, noisy_sample, timesteps, label_conds, train: bool):
            model_variables = {'params': params, 'batch_stats': state.batch_stats}
            output = state.apply_fn(model_variables,
                                    noisy_sample, timesteps, label_conds, train,
                                    rngs={'dropout': state.dropout_rng} if train else None,
                                    mutable=['batch_stats'] if train else False)
            
            if train:
                predicts, updated_model_state = output  
            else:
                predicts, updated_model_state = output, None
            loss_val = jnp.mean((predicts - noises) ** 2)
            return loss_val, updated_model_state
        
        def train_step(state: TrainState, 
                       noises,
                       noisy_sample, timesteps, label_conds):
            loss_val_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss_value, updated_model_state), grads = loss_val_grad_fn(state.params, state, 
                                                                        noises,
                                                                        noisy_sample, timesteps, label_conds, train=True)
            # update the dropout rng!
            dropout_rng = jax.random.fold_in(state.dropout_rng, data=state.step)
            
            updated_state = state.apply_gradients(
                grads=grads,
                batch_stats=updated_model_state['batch_stats'],
                dropout_rng=dropout_rng)
            return {'loss': loss_value}, updated_state
        
        def eval_step(state: TrainState, 
                      noises,
                      noisy_sample, timesteps, label_conds):
            loss_val, updated_model_state = loss_fn(state.params, state, noises,
                                                    noisy_sample, timesteps, label_conds, train=False)
            
            return {'loss': loss_val}
        return loss_fn, train_step, eval_step
    
    def run(self, train_ds, noise_scheduler):
        
        # ========= configure the dataset ==========
        train_dataset = train_ds.with_format('jax')
        num = 10
        sample_img = train_ds[0:0+num]['image'][...,None] / 127.5 - 1
        sample_label = train_ds[0:0+num]['label']
        noises = random.normal(random.key(0), shape=sample_img.shape)
        timesteps = random.randint(random.key(0), shape=[sample_img.shape[0],], minval=0, maxval=scheduler.timesteps)
        noisy_sample = noise_scheduler.add_noise(sample_img, noises, timesteps) # forward diffusion process
        label_conds = nn.one_hot(sample_label, num_classes=10)
        
        if self.train_state is None:
            print('****** Creating TrainState ********')
            self.train_state = self._init_train_state(noisy_sample, timesteps, label_conds, False)
            print('****** Creating TrainState Done ********')
        