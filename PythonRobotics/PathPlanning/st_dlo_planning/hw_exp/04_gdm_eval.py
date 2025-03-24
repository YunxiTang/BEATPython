if __name__ == '__main__':
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    
    import os
    import st_dlo_planning.utils.misc_utils as misu
    import st_dlo_planning.utils.pytorch_utils as ptu
    import numpy as np
    from st_dlo_planning.envs.planar_cable_deform.planar_cable_deform import DualGripperCableEnv, wrap_angle
    from st_dlo_planning.neural_mpc_tracker.gdm_dataset import MultiStepGDMDataset
    from st_dlo_planning.neural_mpc_tracker.modelling_gdm import GDM, GDM_CFG
    from st_dlo_planning.neural_mpc_tracker.mpc_solver import DiscreteModelEnv
    from st_dlo_planning.neural_mpc_tracker.gdm_dataset import visualize_shape
    from st_dlo_planning.utils.pytorch_utils import dict_apply, from_numpy, to_numpy
    import torch
    from torch.utils.data import DataLoader
    import random
    import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    seed = 300
    # misu.setup_seed(seed)
    device = ptu.init_gpu()

    # set a test environment
    env = DualGripperCableEnv(task='03', feat_stride=4)

    # ======================= grab the training dataset stats =================================
    test_data_path = '/media/yxtang/Extreme SSD/HDP/hw_dataset/ethernet_cable/test/task_ethernet_cable_1.zarr'
    test_dataset = MultiStepGDMDataset( test_data_path, max_step=120 )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # ======================= load trained global deformation model =======================
    model_dirs = ['/media/yxtang/Extreme SSD/HDP/results/ethernet_cable/checkpoints/ethernet_cable_gdm_10/',]
    model_ckpt_paths = [model_dirs[0] + 'latest.ckpt',]
    clr = ['r', 'b', 'g', 'k']
    errors = []

    for model_ckpt_path in model_ckpt_paths:
        gdm_cfg = GDM_CFG()

        gdm_model = GDM(gdm_cfg)
        
        model_params = torch.load(model_ckpt_path)['state_dicts']['model']
        gdm_model.load_state_dict(model_params)
        gdm_model.to(device)

        for batch in test_dataloader:
            batch = dict_apply(batch, lambda x: x.to(device))
            eef_states = batch['eef_states']
            dlo_keypoints = batch['dlo_keypoints']
            delta_shape = batch['delta_shape']
            delta_eef = batch['delta_eef']

            predicted_delta_shape = gdm_model(dlo_keypoints, eef_states, delta_eef)

            next_dlo_keypoints_gt = to_numpy( dlo_keypoints + delta_shape )[0]
            next_dlo_keypoints_pred = to_numpy( dlo_keypoints + predicted_delta_shape )[0]

            print( '=======================' )
            
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(projection='3d')

            lifted_dlo_kp = np.concatenate((to_numpy(dlo_keypoints)[0], np.zeros((dlo_keypoints.shape[1], 1))), axis=1)
            visualize_shape(lifted_dlo_kp, ax, clr='b', ld=1)

            lifted_next_dlo_kp_pred = np.concatenate((next_dlo_keypoints_pred, np.zeros((dlo_keypoints.shape[1], 1))), axis=1)
            visualize_shape(lifted_next_dlo_kp_pred, ax, clr='k', ld=0.5)

            lifted_next_dlo_kp_gt = np.concatenate((next_dlo_keypoints_gt, np.zeros((dlo_keypoints.shape[1], 1))), axis=1)
            visualize_shape(lifted_next_dlo_kp_gt, ax, clr='r', ld=0.5)
            plt.show()
        # # ======================= wrap the global deformation model up =======================
        # wrapped_model = DiscreteModelEnv( gdm_model, env.dlo_len, 10, 2, device)

        # # ======================= rollout the model =======================
        # error_norm = run_prediction(wrapped_model, States, Next_States)
        # errors.append(error_norm)