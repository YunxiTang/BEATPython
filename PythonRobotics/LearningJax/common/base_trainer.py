import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from typing import Any
from flax.training import train_state, prefetch_iterator
import orbax.checkpoint as ocp
import optax
from  omegaconf import OmegaConf
from tqdm import tqdm

import copy

import os
import pathlib
import yaml
import threading

import dill

class TrainState(train_state.TrainState):
    batch_stats: Any = None
    dropout_rng: Any = None
    
    
class BaseTrainer:
    '''
        A base trainer class for flax model
    '''
    include_keys = tuple()
    exclude_keys = tuple()
    
    def __init__(self, cfg: OmegaConf):
        self._cfg = cfg
        self._saving_thread = None
        self._ckpt_dir = cfg.checkpoint.ckpt_dir
        self._output_dir = cfg.logging.output_dir
        
        # rng_key for random key streaming
        self.rng_key = jax.random.PRNGKey(seed=cfg.train.seed)
        
        # model
        self.model: nn.Module
        self.model = None
        
        # train_state
        self.train_state : TrainState
        self.train_state = None
        
        # lr scheduler
        self.lr_scheduler = None
        
        # checkpointer
        ckpt_path = ocp.test_utils.erase_and_create_empty(os.path.join(self._ckpt_dir, 'checkpoints'))
        checkpoint_option = ocp.CheckpointManagerOptions(max_to_keep=5, save_interval_steps=10, create=True)
        self.checkpoint_manager = ocp.CheckpointManager(directory=ckpt_path, options=checkpoint_option)
    
    def _init_train_state(self, *inp_sample):
        '''
            Initialize the TrainState
            - initialize the variables for the model,
            - initialize the opt_state for the optimizer
            - create a train state for the training
        '''
        # ============= model initialization ================
        print('========= Model Initialization ==============')
        params_key, dropout_key, self.rng_key = jax.random.split(self.rng_key, 3)
        variables = self.model.init({'params': params_key, 'dropout': dropout_key}, *inp_sample)
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
        '''
            create the `loss_fn` function, `train_step`, `eval_step` here
            
            Out: loss_fn, train_step, eval_step
        '''
        raise NotImplementedError
    
    def run(self, cfg=None):
        '''
            main body of training/evaluation and so on
        '''
        # 1. save **_cfg.yaml once
        
        # 2. setup the train and eval dataset
        
        # initialize the train state firstly
        raise NotImplementedError
        
    
    def save_cfg(self, cfg_path=None):
        '''
            save the whole training config as 'cfg.yaml' file into checkpoint
        '''
        if cfg_path is None:
            cfg_path = os.path.join(self._ckpt_dir, 'cfg.yaml')
        cfg_path = pathlib.Path(cfg_path)
        cfg_path.parent.mkdir(parents=False, exist_ok=True)
        OmegaConf.save(self._cfg, cfg_path)
        return cfg_path.absolute()
    
    def save_model_cfg(self, model_cfg_path=None):
        '''
            save the model config as 'model_cfg.yaml' file into checkpoint
        '''
        if model_cfg_path is None:
            model_cfg_path = os.path.join(self._ckpt_dir, 'model_cfg.yaml')
        model_cfg_path = pathlib.Path(model_cfg_path)
        model_cfg_path.parent.mkdir(parents=False, exist_ok=True)
        OmegaConf.save(self._cfg.model, model_cfg_path)
        return model_cfg_path.absolute()
    
    def save_data_stats(self, data_stats: dict, data_stats_path=None):
        '''
            save training dataset stats as 'data_stats.yaml' file into checkpoint
        '''
        if data_stats_path is None:
            data_stats_path = os.path.join(self._ckpt_dir, f'data_stats.yaml')
        data_stats_path = pathlib.Path(data_stats_path)
        data_stats_path.parent.mkdir(parents=False, exist_ok=True)
        data_stats_to_save = {}
        for key, val in data_stats.items():
            data_stats_to_save[key] = jax.device_get(val).tolist()
        with open(data_stats_path, 'w') as f:
            yaml.dump(data_stats_to_save ,f)
        return data_stats_path.absolute()
    
    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self._ckpt_dir).joinpath(f'{tag}'.ckpt)
    
    def save_checkpoint(self, tag='latest', ckpt_path=None,
                        exclude_keys=None, include_keys=None, 
                        use_thread=True):
        '''
            save checkpoint
        '''
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        
        if include_keys is None:
            include_keys = tuple(self.include_keys)
        
        # Save the checkpoint
        payload = {
            'cfg': self._cfg,
            'train_state': self.train_state,
            'pickles': dict({'rng_key': self.rng_key})
        }
        
        step = self.state.step
        if use_thread:
            self._saving_thread = threading.Thread(
                target=self.checkpoint_manager.save, kargs={'step': step, 'items': payload}
            )
            self._saving_thread.start()
        return None
    
    def save_snapshot(self, tag='latest'):
        '''
            quick save/load the full workspace for convinience
        '''
        snapshot_path = pathlib.Path(self._ckpt_dir).joinpath('snapshots', f'snapshot_{tag}.pkl')
        snapshot_path.parent.mkdir(parents=False, exist_ok=True)
        # Save workspace as a snapshot
        with open(snapshot_path, 'wb') as f:
            dill.dump(self, f)
        return snapshot_path.absolute()
    
    @classmethod
    def create_from_snapshot(cls, snapshot_path):
        with open('snapshot_path', 'rb') as f:
            restored_trainer = dill.load(f)
        return restored_trainer
    
    @classmethod
    def create_from_checkpoint(cls, checkpoint_path):
        '''
            class method: create a trainer from a previous checkpoint
        '''
        
    
    @property
    def output_dir(self):
        return self._output_dir
        

            
        
        
        
        
        
    
        
        