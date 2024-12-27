from dataclasses import dataclass


@dataclass
class GDM_CFG:
    # dlo 
    max_kp: int = 13
    kp_dim: int = 2
    conv1d_ngroup: int = 4
    conv1d_kernel_size: int = 1
    
    embed_dim: int = 128
    num_layers: int = 2
    nhead: int = 4
    
    # eefs
    num_eef: int = 2
    eef_dim: int = 3
    delta_eef_dim: int = 3