if __name__ == '__main__':
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    
    import torch
    from torch.utils.data import DataLoader
    import torch.nn as nn

    from st_dlo_planning.neural_mpc_tracker.modelling_gdm import GDM
    from st_dlo_planning.neural_mpc_tracker.configuration_gdm import GDM_CFG
    from st_dlo_planning.neural_mpc_tracker.gdm_dataset import MultiStepGDMDataset
    from st_dlo_planning.utils.pytorch_utils import to_numpy
    
    train_data_path = '/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/gdm_mj/train/task03_10.zarr'
    test_data_path = '/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/gdm_mj/test/task03_20.zarr'

    train_dataset = MultiStepGDMDataset( train_data_path, max_step=5 )
    test_dataset = MultiStepGDMDataset( test_data_path, max_step=5 )

    batch_size = 256
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model_cfg = GDM_CFG(kp_dim=2)
    gdm_model = GDM(model_cfg)

    optimizer = torch.optim.AdamW(gdm_model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss(reduction='mean')
        
    for epoch in range(50):
        Loss = 0.0
        for batch in train_dataloader:
            dlo_keypoints = batch['dlo_keypoints']
            eef_states = batch['eef_states']

            delta_eef = batch['delta_eef']
            delta_shape = batch['delta_shape']

            predict_vel = gdm_model(dlo_keypoints, eef_states, delta_eef)

            loss_val = loss_fn(predict_vel, delta_shape)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            Loss += to_numpy(loss_val)

        for batch in test_dataloader:
            val_Loss = 0
            with torch.no_grad():
                dlo_keypoints = batch['dlo_keypoints']
                eef_states = batch['eef_states']

                delta_eef = batch['delta_eef']
                delta_shape = batch['delta_shape']

                predict_vel = gdm_model(dlo_keypoints, eef_states, delta_eef)

                loss_val = loss_fn(predict_vel, delta_shape)
                val_Loss += to_numpy(loss_val)

        print(f'Epoch: {epoch} || Train Loss: {Loss} Val Loss: {val_Loss}')
        