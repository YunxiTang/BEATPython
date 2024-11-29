from .configuration_gdm import GDM_CFG
from .modelling_gdm import GDM
import torch


class MpcPolicy:
    def __init__(self, mpc_horizon):
        self.T = mpc_horizon
        
    
    @classmethod
    def load_from_pretrained(cls, pretrained_ckpt):
        pass