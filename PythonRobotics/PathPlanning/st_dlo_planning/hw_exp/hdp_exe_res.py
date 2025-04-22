if __name__ == '__main__':
    import sys
    import os
    import pathlib
    import matplotlib.pyplot as plt
    import numpy as np
    from omegaconf import OmegaConf
    import pickle
    import jax
    import jax.numpy as jnp
    import seaborn as sns
    import matplotlib.animation as animation
    import yaml
    import zarr

    jax.config.update("jax_enable_x64", True)     # enable fp64
    jax.config.update('jax_platform_name', 'cpu') # use the CPU instead of GPU

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    from st_dlo_planning.utils.world_map import Block, WorldMap, MapCfg
    from st_dlo_planning.utils.misc_utils import setup_seed

    from scipy.interpolate import splprep, splev
    setup_seed(0)

    def visualize_shape(dlo: np.ndarray, ax, clrs, ld=3.0, s=25):
        '''
            visualize a rope shape
        '''
        num_kp = dlo.shape[0]

        for i in range(num_kp):
            ax.scatter(dlo[i][0], dlo[i][1], s=s, color=clrs[i], marker='o')
        for i in range(num_kp-1):
            ax.plot([dlo[i][0], dlo[i+1][0]], 
                    [dlo[i][1], dlo[i+1][1]], color=clrs[i], linewidth=ld)

    def fit_bspline(keypoints, num_samples=10, degree=3) -> np.ndarray:
        keypoints = np.array(keypoints)  # Ensure it's a NumPy array
        # num_points, dim = keypoints.shape  # Get number of points and dimensionality

        # Fit a B-Spline through the keypoints
        tck, _ = splprep(keypoints.T, s=0, k=degree)  # Transpose keypoints for splprep

        # Sample points along the fitted spline
        u_fine = np.linspace(0, 1, num_samples)  # Parametric range
        spline_points = splev(u_fine, tck)       # Returns a list of arrays, one for each dimension

        # Stack the arrays into a single (num_samples, dim) array
        spline_points = np.vstack(spline_points).T

        return spline_points

    _, axes = plt.subplots(1, 3, figsize=(12, 3))
    plt.rcParams.update({'font.size': 12})
    sns.set_theme(context='paper', style="ticks", palette="deep")
    # ============== load environment configuration ============== 
    task_type = 'mo'
    task_id = 'task1'
    task_folder = f'/home/yxtang/CodeBase/LfD/robot_hardware/dlo_ws/src/rs_perception/scripts/config/{task_type}_task/{task_id}'
    cfg_path = os.path.join(task_folder, 'task_setup.yaml')

    map_cfg_file = OmegaConf.load(cfg_path)
    
    map_cfg = MapCfg(resolution=map_cfg_file.workspace.resolution,
                     map_xmin=map_cfg_file.workspace.map_xmin,
                     map_xmax=map_cfg_file.workspace.map_xmax,
                     map_ymin=map_cfg_file.workspace.map_ymin,
                     map_ymax=map_cfg_file.workspace.map_ymax,
                     map_zmin=map_cfg_file.workspace.map_zmin,
                     map_zmax=map_cfg_file.workspace.map_zmax,
                     robot_size=map_cfg_file.workspace.robot_size,
                     dim=3)
    
    world_map = WorldMap(map_cfg)

    # 1. add some obstacles from predefined environment setup
    size_z = map_cfg_file.workspace.map_zmax

    obstacles = map_cfg_file.obstacle_info.obstacles
    clrs = sns.color_palette("icefire", n_colors=max(3, len(obstacles))).as_hex()
    for i, obstacle in enumerate(obstacles):
        world_map.add_obstacle(Block(obstacle[0], obstacle[1], size_z, 
                                     obstacle[2], obstacle[3], angle=obstacle[4]*np.pi, clr=clrs[i]))
    world_map.finalize()
    
    world_map.visualize_passage(axes[0], full_passage=False)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlim(-0.26, 0.16)
    axes[0].set_ylim(-0.42, 0.18)
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')

    # Path to your Zarr file
    cable_type = 'usb_cable'
    task_id = f'{task_type}1'
    trial= 12
    root_dir = f'/media/yxtang/Extreme SSD/HDP/experiment_results/{cable_type}/dynamic_deformation_track'
    zarr_file_path = os.path.join(root_dir, f'task-mo1-trial-{trial}.zarr') 
    print(zarr_file_path)
    root = zarr.open(zarr_file_path, mode='r')

    ref_keypoints =  root['data']['dlo_ref_keypoints'][:]
    keypoints = root['data']['dlo_keypoints'][:]
    eef_states = root['data']['eef_states'][:].reshape(-1, 2, 3)
    tracking_error = root['data']['tracking_error'][:]

    timestamp = root['meta']['timestamp'][:]
    num_frame = keypoints.shape[0]
    
    min_error_idx = np.argmin(tracking_error)
    min_error = tracking_error[min_error_idx]

    print(num_frame, min_error_idx, min_error)

    # planned shape
    planned_dlo_shape = ref_keypoints
    print(f'planned_dlo_shape: {planned_dlo_shape.shape}')

    planned_deformation_seq = planned_dlo_shape
    
    planned_shape_sample_idx = [0, 150, 300, 600, 700, 800, 900, 1000, 1050, 1100, min_error_idx]

    clrs = sns.color_palette("Spectral", n_colors=13)
    for i in planned_shape_sample_idx:
        ref_dlo_shape = fit_bspline(planned_deformation_seq[i], 13)
        visualize_shape(ref_dlo_shape, axes[0], clrs=clrs, ld=1.5, s=5)
        # dlo_shape = keypoints[i]
        # visualize_shape(dlo_shape, axes[0], clr='k', ld=1.5, s=5)

    global_target_errs = []
    obs_distances = []
    for t in range(num_frame):
        keypoint = keypoints[t]
        target_err = np.linalg.norm(keypoint - keypoints[min_error_idx])
        obs_distances.append(np.min(np.array([world_map.compute_clearance([point[0], point[1], 0.0]) for point in keypoint])))
        global_target_errs.append(target_err)
        if target_err <= 0.019:
            break

    p1 = axes[2].plot(timestamp[0:t], global_target_errs[0:t], color='k', linewidth=2, label='Target Error')
    axes[2].set_xlabel('Time (s)', color='k')
    axes[2].set_ylabel('Target Error (m)', color='k')
    ax2 = axes[2].twinx()
    p2 = ax2.plot(timestamp[0:t], obs_distances[0:t], color='r', linewidth=2.0, label='Obs. Dist.')
    ax2.set_ylabel('Obs. Dist. (m)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # load optimized sigmas
    opt_res = np.load(f'/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/realworld_result/optimized_detail_res/{task_type}_opt_deatail.npz')

    init_sigmas = opt_res['arr_0']
    opt_sigmas = opt_res['arr_1']
    objective_vals = opt_res['arr_2']
    # axes[0, 0].plot(objective_vals, color='k', linewidth=2)
    # axes[0, 0].set_xlabel('Iter.')
    # axes[0, 0].set_ylabel('J')
    # axes[0, 0].set_yscale('log')

    clrs = sns.color_palette("Spectral", n_colors=15)
    print(f'opt_sigmas.shape: {opt_sigmas.shape}')
    
    for i in range(1, 14, 1):
        axes[1].plot(opt_sigmas[:, i-1], color=clrs[i], linewidth=2)
    axes[1].plot(range(51), np.linspace(0, 1, 51), color='k', linestyle='--', linewidth=2, label=f'Init. $\sigma$')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel(r'Optimized $\sigma$ s')
    # axes[1].legend(loc='lower right', fontsize='small', frameon=False, ncol=2)
    axes[1].legend(loc='upper left', frameon=False)

    for ax in axes:
        ax.grid(True)
    plt.tight_layout()
    output_path = f'/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/realworld_result/optimized_detail_res/task-{task_id}-trial-{trial}.png'
    plt.savefig(output_path, dpi=900)
    print(f"Figure saved to {output_path}")
    plt.show()