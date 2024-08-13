import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from typing import Any
from flax.training import train_state, checkpoints, prefetch_iterator
import optax

from tqdm import tqdm



class TrainState(train_state.TrainState):
    batch_stats: Any = None
    dropout_rng: Any = None
   
   
class FlaxTrainer:
    '''
        A trainer class for flax model
    '''
    def __init__(self, model, scheduler, *inp_sample):
        # create a rng_key for random streaming
        self.rng_key = jax.random.PRNGKey(seed=1200)

        # create a model
        self.model: nn.Module
        self.model = model

        # create an empty train state
        self.train_state = None

        # init the model & train state
        self.train_state = self._init_train_state(*inp_sample)
        del *inp_sample
        
        # scheduler
        self.scheduler = scheduler
        
        self.log_dir = '/home/yxtang/CodeBase/PythonCourse/PythonRobotics/LearningJax/ddpm_conv/res/checkpoints'

    def _init_train_state(self, *inp_sample):
        '''
            - initialize the variables for the model,
            - initialize the opt_state for the optimizer
            - create a train state for the training
        '''
        # ============= model initialization ================
        print('========= Model Initialization ==========')
        params_key, dropout_key, self.rng_key = jax.random.split(self.rng_key, 3)
        variables = self.model.init({'params': params_key,
                                     'droput_rng': dropout_key}, *inp_sample)
        params = variables.get('params')
        batch_stats = variables.get('batch_stats', {})
        print('========= Model Initialization Done =========')

        # ============= optimizer initialization =============
        print('========= Optimizer Initialization =========')
        optimizer_ = optax.adamw(learning_rate=0.001)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optimizer_
        )
        opt_state = optimizer.init(params)
        print('========= Optimizer Initialization Done =========')

        total_param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"**** Number of Params: {total_param_count} ****")
        # ============= assemble the train state =============
        train_state = TrainState(
            step=0,
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer,
            opt_state=opt_state,
            batch_stats=batch_stats,
            dropout_rng=self.rng_key
        )
        return train_state

    def create_function(self):
        # loss function
        def loss_fn(params, state: TrainState, 
                    noises, noisy_sample, timesteps, label_conds, train: bool):
            model_variables = {'params': params, 'batch_stats': state.batch_stats}
            output = state.apply_fn(model_variables,
                                    noisy_sample, timesteps, label_conds, train,
                                    rngs={'dropout_rng': state.dropout_rng} if train else None,
                                    mutable=['batch_stats'] if train else False)
            
            if train:
                predicts, updated_model_state = output  
            else:
                predicts, updated_model_state = output, None
            loss_val = jnp.mean((predicts - noises) ** 2)
            return loss_val, updated_model_state
            
        # train step function
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
       
        # eval step function
        def eval_step(state: TrainState, 
                      noises,
                      noisy_sample, timesteps, label_conds):
            loss_val, updated_model_state = loss_fn(state.params, state, noises,
                                                    noisy_sample, timesteps, label_conds, train=False)
            
            return {'loss': loss_val}
        return train_step, eval_step
    
    
    def save_model(self, epoch=0):
        # Save current model at certain training epoch
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target={'params': self.train_state.params, 'batch_stats': self.train_state.batch_stats},
                                    step=epoch,
                                    keep=4,
                                    overwrite=True)
   
    def train(self, train_ds, eval_ds=None, test_ds=None):
        '''
            train
        '''
        assert self.train_state is not None, 'Train state is None!'
        # ========= configure the dataset ==========
        train_dataset = train_ds.with_format('jax')
        if test_ds is not None:
            test_dataset = test_ds.with_format('jax')

        # create the functions
        train_step, eval_step = self.create_function()
        jitted_train_step = jax.jit(train_step)
        jitted_eval_step = jax.jit(eval_step)
        
        # ========= main loop ===================
        step = 0
        for epoch in range(50):
            Loss = 0
            for batch in train_dataset.iter(32):
                sample_img = batch['image'][...,None] / 127.5 - 1.0

                sample_label = batch['label']
                label_conds = nn.one_hot(sample_label, num_classes=10)

                key1, key2 = random.split(self.rng_key, 2)

                timesteps = random.randint(key1, shape=[sample_img.shape[0],], minval=0, maxval=self.scheduler.timesteps)
                noises = random.normal(key2, shape=sample_img.shape)

                noisy_sample = self.scheduler.add_noise(sample_img, noises, timesteps)
                
                metric, self.train_state = jitted_train_step(self.train_state, 
                                                             noises, 
                                                             noisy_sample, timesteps, label_conds)
                step_loss = metric['loss']
                Loss += step_loss
                
                step += 1
                self.rng_key = jax.random.fold_in(self.rng_key, data=step)
                
                if step % 100 == 0:
                    print(f'Step: {step} || Step Loss: {step_loss}')

            print(f'Epoch: {epoch} || Train Loss: {Loss}')
            if epoch % 2 == 0:
                self.save_model(epoch)
        trained_state = self.train_state
        return trained_state