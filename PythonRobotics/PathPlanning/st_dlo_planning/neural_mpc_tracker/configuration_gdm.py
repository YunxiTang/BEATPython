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


@dataclass
class CONV_GDM_CFG:
    # dlo 
    max_kp: int = 13
    kp_dim: int = 2

    embed_dim: int = 128
    
    # eefs
    num_eef: int = 2
    eef_dim: int = 3
    delta_eef_dim: int = 3


@dataclass
class RNN_GDM_CFG:
    # dlo 
    max_kp: int = 13
    kp_dim: int = 2
    conv1d_ngroup: int = 4
    conv1d_kernel_size: int = 1
    
    # rnn
    rnn_hidden_size: int = 128
    rnn_num_layers: int = 2
    rnn_dropout: float = 0.1
    
    # eefs
    num_eef: int = 2
    eef_dim: int = 3
    delta_eef_dim: int = 3