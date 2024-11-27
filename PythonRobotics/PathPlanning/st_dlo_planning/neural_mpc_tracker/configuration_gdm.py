from dataclasses import dataclass
import torch.nn as nn
import torch

@dataclass
class GDM_CFG:
    embed_dim: int = 256
    num_layers: int = 2
    nhead: int = 8
    
    
if __name__ == '__main__':
    gdm_cfg = GDM_CFG()
    print(gdm_cfg)