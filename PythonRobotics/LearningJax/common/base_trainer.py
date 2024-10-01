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
from typing import Tuple
import os
import pathlib
import yaml


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
        
    @property
    def output_dir(self):
        return self._output_dir
    
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
        data_stats_to_save = {}
        for key, val in data_stats.items():
            data_stats_to_save[key] = jax.device_get(val).tolist()
        with open(data_stats_path, 'w') as f:
            yaml.dump(data_stats_to_save ,f)
        return data_stats_path.absolute()
        
        
    
        
        