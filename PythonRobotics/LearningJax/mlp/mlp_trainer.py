from omegaconf import OmegaConf
from common.base_trainer import BaseTrainer
import copy
from datasets import Dataset
import jax.numpy as jnp
import jax

from model import MLP


class MLPTrainer(BaseTrainer):
    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        self.model = MLP(cfg.model.inp_dim, cfg.model.out_dim)
        
    def run(self, cfg=None):
        if cfg is None:
            cfg = copy.deepcopy(self._cfg)
        
        if cfg.checkpoint.save_ckpt:
            self.save_cfg()
            self.save_model_cfg()
        
        # setup dataset
        x = jnp.linspace(0., 10., 1000) 
        y = 2 * x + jax.random.normal(jax.random.PRNGKey(0), (1000,))
        
        train_dataset = Dataset.from_dict({'x': x, 'y': y})
        input_sample = train_dataset[0]
        self.train_state = self._init_train_state(jnp.array(input_sample['x'])[None], False)
        
        print(cfg)
        