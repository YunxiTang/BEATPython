if __name__ == '__main__':
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    
    import torch
    from st_dlo_planning.neural_mpc_tracker.modelling_gdm import DLOEncoder, GDM
    from st_dlo_planning.neural_mpc_tracker.configuration_gdm import GDM_CFG
    
    
    model_cfg = GDM_CFG()
    dlo_encoder = DLOEncoder(model_cfg)
    
    batch_size = 12
    seq_len = 19
    kp_dim = 3
    
    x = torch.randn(batch_size, seq_len, kp_dim)
    y = dlo_encoder(x)
    
    for key, val in dlo_encoder.state_dict().items():
        print(key, val.shape)
        
    print(y.shape)
    
    gdm_model = GDM(model_cfg)
    for key, val in gdm_model.state_dict().items():
        print(key, val.shape)
        
    eefPos = torch.randn(batch_size, 2, 7)
    eefVel = torch.randn(batch_size, 2, 6)
    dlo_kp = x
    
    predict_vel = gdm_model(dlo_kp, eefPos, eefVel)
    print(predict_vel.shape)