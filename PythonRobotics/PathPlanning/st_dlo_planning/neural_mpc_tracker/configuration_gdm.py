from dataclasses import dataclass


@dataclass
class GDM_CFG:
    # dlo 
    max_kp: int = 20
    kp_dim: int = 3
    conv1d_ngroup: int = 4
    conv1d_kernel_size: int = 1
    
    embed_dim: int = 128
    num_layers: int = 2
    nhead: int = 8
    
    # gripper eefs
    eef_pos_dim: int = 7
    eef_vel_dim: int = 6