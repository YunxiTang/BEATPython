if __name__ == "__main__":
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

    jax.config.update("jax_enable_x64", True)  # enable fp64
    jax.config.update("jax_platform_name", "cpu")  # use the CPU instead of GPU

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)

    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    from st_dlo_planning.utils.world_map import Block, WorldMap, MapCfg, plot_circle
    from st_dlo_planning.utils.misc_utils import setup_seed

    from st_dlo_planning.utils.path_set import (
        PathSet,
        transfer_path_between_start_and_goal,
        deform_pathset_step1,
        deform_pathset_step2,
    )
    from st_dlo_planning.spatial_pathset_gen.dlo_ompl import DloOmpl

    from st_dlo_planning.utils.path_interpolation import visualize_shape
    from st_dlo_planning.temporal_config_opt.opt_solver import TcDloSolver
    from st_dlo_planning.temporal_config_opt.qp_solver import polish_dlo_shape

    from scipy.interpolate import splprep, splev

    setup_seed(0)

    def fit_bspline(keypoints, num_samples=10, degree=3) -> np.ndarray:
        keypoints = np.array(keypoints)  # Ensure it's a NumPy array
        # num_points, dim = keypoints.shape  # Get number of points and dimensionality

        # Fit a B-Spline through the keypoints
        tck, _ = splprep(keypoints.T, s=0, k=degree)  # Transpose keypoints for splprep

        # Sample points along the fitted spline
        u_fine = np.linspace(0, 1, num_samples)  # Parametric range
        spline_points = splev(
            u_fine, tck
        )  # Returns a list of arrays, one for each dimension

        # Stack the arrays into a single (num_samples, dim) array
        spline_points = np.vstack(spline_points).T

        return spline_points

    # ============== load environment configuration ==============
    task_id = "task1"
    task_folder = f"/home/yxtang/CodeBase/LfD/robot_hardware/dlo_ws/src/rs_perception/scripts/config/so_task/{task_id}"
    cfg_path = os.path.join(task_folder, "task_setup.yaml")

    map_cfg_file = OmegaConf.load(cfg_path)

    map_cfg = MapCfg(
        resolution=map_cfg_file.workspace.resolution,
        map_xmin=map_cfg_file.workspace.map_xmin,
        map_xmax=map_cfg_file.workspace.map_xmax,
        map_ymin=map_cfg_file.workspace.map_ymin,
        map_ymax=map_cfg_file.workspace.map_ymax,
        map_zmin=map_cfg_file.workspace.map_zmin,
        map_zmax=map_cfg_file.workspace.map_zmax,
        robot_size=map_cfg_file.workspace.robot_size,
        dim=3,
    )

    world_map = WorldMap(map_cfg)

    # 1. add some obstacles from predefined environment setup
    size_z = map_cfg_file.workspace.map_zmax

    obstacles = map_cfg_file.obstacle_info.obstacles
    clrs = sns.color_palette("icefire", n_colors=max(3, len(obstacles))).as_hex()
    for i, obstacle in enumerate(obstacles):
        world_map.add_obstacle(
            Block(
                obstacle[0],
                obstacle[1],
                size_z,
                obstacle[2],
                obstacle[3],
                angle=obstacle[4] * np.pi,
                clr=clrs[i],
            )
        )
    world_map.finalize()

    ax = world_map.visualize_passage(full_passage=False)

    # planned shape
    planned_dlo_shape = np.load(
        "/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/realworld_result/optimal_shape_seq_res/so_task1_optimal_shape_seq.npy"
    )
    print(planned_dlo_shape.shape)
    planned_deformation_seq = planned_dlo_shape.reshape(-1, 10, 3)

    for i in range(50):
        ref_dlo_shape = planned_deformation_seq[i]
        # visualize_shape(dlo_shape, ax, clr='k', ld=1.5, s=20)
        visualize_shape(ref_dlo_shape, ax, clr="r", ld=1.5, s=20)
    plt.axis("equal")
    ax.set_xlim(world_map.map_cfg.map_xmin - 0.05, world_map.map_cfg.map_xmax + 0.05)
    ax.set_ylim(world_map.map_cfg.map_ymin - 0.05, world_map.map_cfg.map_ymax + 0.05)
    plt.show()

    cable_type = "braid_cable"
    task_id = "so1"
    trial = 1

    # Path to your Zarr file
    root_dir = f"/media/yxtang/Extreme SSD/HDP/experiment_results/{cable_type}/dynamic_deformation_track"
    zarr_file_path = os.path.join(root_dir, f"task-{task_id}-trial-{trial}.zarr")
    print(zarr_file_path)
    root = zarr.open(zarr_file_path, mode="r")

    ref_keypoints = root["data"]["dlo_ref_keypoints"][:]
    keypoints = root["data"]["dlo_keypoints"][:]
    eef_states = root["data"]["eef_states"][:].reshape(-1, 2, 3)
    tracking_error = root["data"]["tracking_error"][:]

    timestamp = root["meta"]["timestamp"][:]
    num_frame = keypoints.shape[0]

    min_error_idx = np.argmin(tracking_error)
    min_error = tracking_error[min_error_idx]

    print(num_frame, min_error_idx, min_error)

    selected_idx = [0, 600, 1200, min_error_idx]
    for i in selected_idx:
        dlo_shape = keypoints[i]
        ref_dlo_shape = ref_keypoints[i]
        # visualize_shape(dlo_shape, ax, clr='k', ld=1.5, s=20)
        visualize_shape(ref_dlo_shape, ax, clr="r", ld=1.5, s=20)
    plt.axis("equal")
    ax.set_xlim(world_map.map_cfg.map_xmin - 0.05, world_map.map_cfg.map_xmax + 0.05)
    ax.set_ylim(world_map.map_cfg.map_ymin - 0.05, world_map.map_cfg.map_ymax + 0.05)
    plt.show()
