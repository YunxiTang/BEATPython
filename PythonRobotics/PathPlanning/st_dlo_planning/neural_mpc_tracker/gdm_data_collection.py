"""Data Collection Script"""
import os
import st_dlo_planning.utils.misc_utils as misu
import st_dlo_planning.utils.pytorch_utils as ptu
import numpy as np
from st_dlo_planning.envs.planar_cable_deform.planar_cable_deform import DualGripperCableEnv
from st_dlo_planning.utils.misc_utils import VideoLoggerPro, ZarrLogger


def CollectOfflineData(env:DualGripperCableEnv, 
                       data_logger:ZarrLogger,
                       seed:int,
                       num_frame:int=50000, 
                       episode_len:int=200, 
                       mode:str='random', 
                       render_mode: str='human',
                       render: bool = True):

    DataCollectMode = ['random', 'teleop', 'expert_policy']
    if mode not in DataCollectMode:
        raise ValueError(f"mode {mode} is not supported.")

    dlo_len = env.dlo_len

    if render_mode != 'human':
        video_logger = VideoLoggerPro(f'/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/{task}_dlo_collection.mp4', fps=1./env.dt)
    
    if mode == 'random':
        K = np.array([[-1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]]) * 0.2
        center = np.array([dlo_len / 2., 0.0, 0.5])
        
        frame = 0
        while True:
            L_target = np.random.uniform(low=-1., high=1., size=(3, ))
            R_target = np.random.uniform(low=-1., high=1., size=(3, ))
            L_norm = (L_target[0]**2 + L_target[1]**2 + L_target[2]**2) **(0.5)
            R_norm = (R_target[0]**2 + R_target[1]**2 + R_target[2]**2) **(0.5)
            L_target[0] = -L_target[0] if L_target[0] > 0 else L_target[0]
            R_target[0] = -R_target[0] if R_target[0] < 0 else R_target[0]

            r = np.random.uniform(low=0.1, high=dlo_len / 2., size=(2,))
            L_target = L_target / L_norm * r[0] + center
            R_target = R_target / R_norm * r[1] + center

            state, _ = env.reset()
            done = False

            for _ in range(episode_len):
                gripper_pos = state[3*env.num_feat:3*(env.num_feat+env.num_grasp)]
                L_pos = gripper_pos[0:3]
                R_pos = gripper_pos[3:6]
                action = np.concatenate([K @ (L_target-L_pos), K @ (R_target-R_pos) ])
                action = np.clip(action, -0.15, 0.15)
                
                next_state, _, done, _, _ = env.step(action)

                data_logger.log_data( 'state', state )
                data_logger.log_data( 'next_state', next_state )
                data_logger.log_data( 'action', action )

                data_logger.log_meta( 'dlo_len', env.dlo_len )
                
                if render:
                    if render_mode != 'human':
                        img = env.render(mode=render_mode) #, camera_name='track_cable'
                        if video_logger.num_frame <= 500:
                            video_logger.log_frame(img)
                    else:
                        env.render()
                
                if (np.linalg.norm(L_target-L_pos) + np.linalg.norm(R_target-R_pos)) < 5e-2:
                    done = True
                if done:
                    break
                state = next_state
                frame += 1

                if frame >= num_frame:
                    break
                if frame % 5000 == 0:
                    seed += 100
                    misu.setup_seed(seed)
                print(f'============== frame: {frame} with seed: {seed} ==================')
            if frame >= num_frame:
                break
        
    else:
        raise NotImplementedError("Not implmented now.")
      
    if render_mode != 'human':
        video_logger.create_video()


if __name__ == '__main__':
    seed = 300
    misu.setup_seed(seed)
    device = ptu.init_gpu()

    task = '03'

    dataset_dir = '/home/yxtang/CodeBase/DOBERT/datasets/gdm_mj/train'
    dataset_name = f'gdm_{task}_mujoco_rope_train.zarr'

    if not (os.path.exists(dataset_dir)):
        os.makedirs(dataset_dir)
    
    data_path = os.path.join(dataset_dir, dataset_name)
    

    data_logger = ZarrLogger( path_to_save=data_path, 
                              data_ks=['state', 'next_state', 'action'],
                              meta_ks=['dlo_len',])
    
    render_mode = 'human' #  'rgb_array' # 
    env = DualGripperCableEnv(task, feat_stride=5, render_mode=render_mode)
    print(env.num_feat)
    
    CollectOfflineData(env, data_logger, seed, num_frame=10000, render_mode=render_mode, render=False)
    data_logger.save_data()