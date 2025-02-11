from omegaconf import OmegaConf
from common.base_trainer import BaseTrainer, TrainState
import copy
import jax.numpy as jnp
import jax
from jax import random
import optax

from model import MLP
from data_gen import FlaxDataloader


class MLPTrainer(BaseTrainer):
    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        self.model = MLP(cfg.model.inp_dim, cfg.model.out_dim)
        self.lr_scheduler = optax.exponential_decay(
                            init_value=self._cfg.train.lr,  # Initial learning rate
                            transition_steps=1000,          # Number of steps before decay begins
                            decay_rate=0.99,                # Factor by which the learning rate decays
                            end_value=0.0001                # Minimum learning rate
                        )
    
    def create_function(self):
        def loss_fn(params, state: TrainState, batch, train: bool):
            feats = batch['feat']
            labels = batch['label']
            model_variables = {'params': params, 'batch_stats': state.batch_stats}
            output = state.apply_fn(model_variables,
                                    feats, train,
                                    rngs={'dropout': state.dropout_rng} if train else None,
                                    mutable=['batch_stats'] if train else False)
            if train:
                predicts, updated_model_state = output  
            else:
                predicts, updated_model_state = output, None
            
            loss_val = jnp.mean( optax.l2_loss(predicts, labels) )#jnp.mean(predicts, labels)
            return loss_val, updated_model_state
        
        def train_step(state: TrainState, batch: dict):
            loss_val_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss_value, updated_model_state), grads = loss_val_grad_fn(state.params, state, batch, train=True)
            # remember to update the dropout rng!
            dropout_rng = random.fold_in(state.dropout_rng, data=state.step)
            updated_state = state.apply_gradients(
                grads=grads,
                batch_stats=updated_model_state['batch_stats'],
                dropout_rng=dropout_rng)
            return {'loss': loss_value}, updated_state
        
        def eval_step(state: TrainState, batch: dict):
            loss_value, _ = loss_fn( state.params, state, batch, train=False )
            return {'loss': loss_value}
        
        return loss_fn, train_step, eval_step
            
    
        
    def run(self, cfg=None):
        if cfg is None:
            cfg = copy.deepcopy(self._cfg)
        
        if cfg.checkpoint.save_ckpt:
            self.save_cfg()
            self.save_model_cfg()
        
        # setup dataset
        x = jnp.linspace(0., 10., 1000)[...,None] 
        y = 2 * x + jax.random.normal(jax.random.PRNGKey(0), (1000, 1)) * 0.01
        train_dataloader = FlaxDataloader(x, y, cfg.train.batch_size, jax.random.PRNGKey(12))
        if self.train_state is None:
            print('****** Creating TrainState ********')
            input_sample = {'x': x[0:2]}
            self.train_state = self._init_train_state(input_sample['x'], False)
            print('****** Creating TrainState Done ********')
        
        print('****** Creating Function ********')
        loss_fn, train_step, eval_step = self.create_function()
        print('****** Creating Function Done ********')
        
        if not cfg.train.debug:
            train_step = jax.jit(train_step)
            eval_step = jax.jit(eval_step)
        
        # train loop
        for epoch in range(cfg.train.max_epoch):
            Loss = 0
            for batch_x, batch_y in train_dataloader:
                train_batch = jax.device_put({'feat': batch_x, 'label': batch_y})
                
                metric, self.train_state = train_step(self.train_state, train_batch)
                loss = metric['loss']
                Loss += jax.device_get(loss)
            print(f'Epoch {self.epoch}: Loss {Loss}')
            
            if epoch % cfg.checkpoint.ckpt_interval == 0:
                self.save_checkpoint(tag=f'{self.epoch}', use_thread=True)
            self.epoch += 1
        
        # save last snapshot
        self.save_snapshot()
            