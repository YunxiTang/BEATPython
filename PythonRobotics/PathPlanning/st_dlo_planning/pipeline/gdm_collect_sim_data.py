if __name__ == '__main__':
    import sys
    import os
    import pathlib
    
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    
    from st_dlo_planning.neural_mpc_tracker.gdm_data_collection import CollectOfflineData
    from st_dlo_planning.envs.planar_cable_deform.planar_cable_deform import DualGripperCableEnv
    from st_dlo_planning.utils.misc_utils import ZarrLogger
    import st_dlo_planning.utils.misc_utils as misu
    import st_dlo_planning.utils.pytorch_utils as ptu

    seed = 800
    misu.setup_seed(seed)
    device = ptu.init_gpu()

    task = '03'

    dataset_dir = '/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/gdm_data'
    dataset_name = f'task_{task}_train.zarr'

    if not (os.path.exists(dataset_dir)):
        os.makedirs(dataset_dir)
    
    data_path = os.path.join(dataset_dir, dataset_name)
    

    data_logger = ZarrLogger( path_to_save=data_path, 
                              data_ks=['dlo_keypoints', 'eef_states', 'eef_transforms',
                                       'next_dlo_keypoints', 'next_eef_states', 'next_eef_transforms',
                                       'ep_num', 'action'],
                              meta_ks=['dlo_len',])
    
    render_mode = 'human' #  'rgb_array' # 
    env = DualGripperCableEnv(task, feat_stride=3, render_mode=render_mode)
    print(env.num_feat)

    CollectOfflineData(env, task, data_logger, seed, num_frame=10000, render_mode=render_mode, render=True)
    data_logger.save_data()