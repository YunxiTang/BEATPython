if __name__ == '__main__':
    import sys
    import os
    import pathlib
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from pprint import pprint
    from omegaconf import OmegaConf

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    
    from st_dlo_planning.neural_mpc_tracker.policy import GradientLMPCAgent
    from st_dlo_planning.neural_mpc_tracker.modelling_gdm import GDM
    from st_dlo_planning.envs.planar_cable_deform.planar_cable_deform import DualGripperCableEnv
    
    # setup the simulation env
    mode = 'human' #'rgb_array' #'depth_array' #
    camera = 'topview'
    task_id = '06'
    env = DualGripperCableEnv(task=task_id, feat_stride=4, render_mode=mode, camera_name=camera)
    print(env.num_feat)
    obs, _ = env.reset()
    
    # load the planned dlo configuration sequence
    
    
    # ================ config the mpc solver ===================
    @torch.jit.script
    def path_cost_func_tensor(x:torch.Tensor, u:torch.Tensor, target_shape:torch.Tensor):
        device = torch.device('cuda:0')
        dlo_shape = x[0:3*10]
        # seprate the shape into position and shape parts
        target_shape = target_shape.reshape(-1, 3)
        dlo_shape = dlo_shape.reshape(-1, 3)

        target_position = torch.mean(target_shape, dim=0, keepdim=True)
        dlo_position = torch.mean(dlo_shape, dim=0, keepdim=True)

        target_curve = target_shape - target_position
        dlo_curve = dlo_shape - dlo_position

        pos_err = target_position - dlo_position
        curve_err = target_curve - dlo_curve

        curve_err = curve_err.flatten()
        pos_err = pos_err.flatten()

        weight_position = torch.diag(torch.tensor([1.0, 1.0, 1.5], device=device)) * 50.
        weight_curve = torch.diag(torch.tensor([1.0, 1.0, 1.0] + [1.0, 1.0, 1.0] * (10-2) + [1.0, 1.0, 1.0], device=device)) * 90.
        weight_u = torch.diag(torch.tensor([1.0, 1.0, 1.0] * 2, device=device)) * 20.0
        
        position_loss = 0.5 * (pos_err.T @ weight_position @ pos_err)
        curve_loss = 0.5 * (curve_err.T @ weight_curve @ curve_err)
        ctrl_loss =  0.5 * (u.T @ weight_u @ u)
        c = position_loss + curve_loss + ctrl_loss
        return c 
    
    @torch.jit.script
    def final_cost_func_tensor(xf:torch.Tensor, target_shape:torch.Tensor):
        device = torch.device('cuda:0')
        dlo_shape = xf[0:3*10]
        # seprate the shape into position and shape subparts
        target_shape = target_shape.reshape(-1, 3)
        dlo_shape = dlo_shape.reshape(-1, 3)

        target_position = torch.mean(target_shape, dim=0, keepdim=True)
        dlo_position = torch.mean(dlo_shape, dim=0, keepdim=True)

        target_curve = target_shape - target_position
        dlo_curve = dlo_shape - dlo_position

        pos_err = target_position - dlo_position
        curve_err = target_curve - dlo_curve

        curve_err = curve_err.flatten()
        pos_err = pos_err.flatten()

        weight_position = torch.diag(torch.tensor([1.0, 1.0, 1.5], device=device)) * 50.
        weight_curve = torch.diag(torch.tensor([1.0, 1.0, 1.0] + [1.0, 1.0, 1.0] * (10-2) + [1.0, 1.0, 1.0], device=device)) * 90.
        
        position_loss = 0.5 * (pos_err.T @ weight_position @ pos_err)
        curve_loss = 0.5 * (curve_err.T @ weight_curve @ curve_err)
        
        c = position_loss + curve_loss
        return c

    # ====================== global deformation model ===================
    model_dirs = []

    model_idx = 2
    model_ckpt_path = model_dirs[model_idx] + 'epoch_2200.ckpt'