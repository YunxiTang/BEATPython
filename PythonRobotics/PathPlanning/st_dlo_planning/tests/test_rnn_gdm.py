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

    from st_dlo_planning.neural_mpc_tracker.modelling_gdm import RNN_GDM
    from st_dlo_planning.neural_mpc_tracker.configuration_gdm import RNN_GDM_CFG
    from st_dlo_planning.neural_mpc_tracker.gdm_dataset import MultiStepGDMDataset
    from st_dlo_planning.utils.pytorch_utils import to_numpy, dict_apply
    from st_dlo_planning.utils.misc_utils import setup_seed

    setup_seed(10)
    
    train_data_path = '/media/yxtang/Extreme SSD/HDP/hw_dataset/ethernet_cable/train/task_ethernet_cable_1.zarr'
    test_data_path = '/media/yxtang/Extreme SSD/HDP/hw_dataset/ethernet_cable/test/task_ethernet_cable_1.zarr'

    train_dataset = MultiStepGDMDataset( train_data_path, max_step=10 )
    test_dataset = MultiStepGDMDataset( test_data_path, max_step=10 )

    batch_size = 256
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model_cfg = RNN_GDM_CFG(kp_dim=2)
    gdm_model = RNN_GDM(model_cfg).to(device='cuda:0')

    optimizer = torch.optim.AdamW(gdm_model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss(reduction='mean')
        
    best_loss = 1e10
    for epoch in range(100):
        Loss = 0.0
        for batch in train_dataloader:
            batch = dict_apply(batch, lambda x: x.to(device='cuda:0'))
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
                batch = dict_apply(batch, lambda x: x.to(device='cuda:0'))
                dlo_keypoints = batch['dlo_keypoints']
                eef_states = batch['eef_states']

                delta_eef = batch['delta_eef']
                delta_shape = batch['delta_shape']

                predict_vel = gdm_model(dlo_keypoints, eef_states, delta_eef)

                loss_val = loss_fn(predict_vel, delta_shape)
                val_Loss += to_numpy(loss_val)
        if val_Loss < best_loss:
            best_loss = val_Loss
        print(f'Epoch: {epoch} || Train Loss: {Loss} Val Loss: {val_Loss}')
    print(best_loss)