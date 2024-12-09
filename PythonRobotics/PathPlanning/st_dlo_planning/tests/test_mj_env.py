if __name__ == '__main__':
    import sys
    import os
    import pathlib
    import matplotlib.pyplot as plt
    import numpy as np
    from omegaconf import OmegaConf

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    from st_dlo_planning.envs.planar_cable_deform.planar_cable_deform import DualGripperCableEnv
    import seaborn as sns


def visualize_shape(dlo: np.ndarray, ax, ld=3.0, s=25, clr=None):
    '''
        visualize a rope shape
    '''
    if clr is None:
        clr = 0.5 + 0.5 * np.random.random(3)

    num_kp = dlo.shape[0]

    for i in range(num_kp):
        ax.scatter(dlo[i][0], dlo[i][1], dlo[i][2], color=clr,  marker='o', s=s)
    for i in range(num_kp-1):
        ax.plot3D([dlo[i][0], dlo[i+1][0]], 
                  [dlo[i][1], dlo[i+1][1]], 
                  [dlo[i][2], dlo[i+1][2]], color=clr, linewidth=ld)
    ax.axis('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')


if __name__ == '__main__':
    import tqdm
    import matplotlib.pyplot as plt

    mode = 'human' #'rgb_array' #'depth_array' #
    camera = 'topview'
    env = DualGripperCableEnv(task='06', feat_stride=4, render_mode=mode, camera_name=camera)
    print(env.num_feat)
    obs, _ = env.reset()

    # save visualization and rewards
    imgs = [env.render(mode=mode, camera_name=camera)]
    obs_hist = []

    left_eef_twists = []
    right_eef_twists = []
    desired_twists = []
    ts = []
    for i in range(250):
        ts.append(i)
        action = np.array([0.02, 0.0, 0.00, 0.00, 0.00, 0.04, 
                          -0.02, 0.0, 0.00, 0.00, 0.00, -0.03]) * 1.0 + 0.02 * np.sin(i * 0.3)
        obs, _, done, _, _ = env.step(action)
        left_eef_twist, right_eef_twist = env.get_eef_twist()
        
        obs_hist.append(obs)
        left_eef_twists.append(left_eef_twist[None])
        right_eef_twists.append(right_eef_twist[None])
        desired_twists.append(action[None])

        if done:
            break
        imgs.append(env.render(mode=mode, camera_name=camera))

    
    sns.set_theme('paper')
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(projection='3d')
    for i in range(5, len(obs_hist), 40):
        obs = obs_hist[i]
        visualize_shape(obs[0:(env.num_feat)*3].reshape(-1, 3), ax, clr='r')
        l_gripper = obs[(env.num_feat)*3:].reshape(-1, 3)[0]
        r_gripper = obs[(env.num_feat)*3:].reshape(-1, 3)[1]
        ax.scatter(l_gripper[0], l_gripper[1], l_gripper[2], color='k',  marker='d', s=45)
        ax.scatter(r_gripper[0], r_gripper[1], r_gripper[2], color='k',  marker='d', s=45)
    plt.show()

    fig = plt.figure(figsize=plt.figaspect(0.5))
    left_eef_twists = np.concatenate(left_eef_twists, axis=0)
    print(left_eef_twists.shape)

    desired_twists = np.concatenate(desired_twists, axis=0)
    print(desired_twists.shape)
    
    plt.plot(ts, left_eef_twists[:, 0], 'k-')
    plt.plot(ts, desired_twists[:, 3], 'k--')

    plt.plot(ts, left_eef_twists[:, 1], 'r-')
    plt.plot(ts, desired_twists[:, 4], 'r--')

    plt.plot(ts, left_eef_twists[:, 2], 'b-')
    plt.plot(ts, desired_twists[:, 5], 'b--')
    plt.show()

    