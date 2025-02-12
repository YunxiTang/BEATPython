if __name__ == '__main__':
    import sys
    import os
    import pathlib
    import matplotlib.pyplot as plt
    import numpy as np
    import jax.numpy as jnp
    from pprint import pprint
    from omegaconf import OmegaConf
    import seaborn as sns
    import pickle

    sns.set_theme('paper')


    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    
    from st_dlo_planning.utils.world_map import Block, WorldMap, MapCfg, plot_circle
    from st_dlo_planning.utils.path_set import (PathSet, transfer_path_between_start_and_goal, 
                                                deform_pathset_step1, deform_pathset_step2)
    from st_dlo_planning.utils.world_map import plot_circle
    from st_dlo_planning.utils.path_interpolation import visualize_shape
    import jax
    jax.config.update("jax_enable_x64", True)     # enable fp64
    jax.config.update('jax_platform_name', 'cpu') # use the CPU instead of GPU

    from st_dlo_planning.temporal_config_opt.opt_solver import TcDloSolver
    from st_dlo_planning.temporal_config_opt.qp_solver import polish_dlo_shape

    import zarr
    map_case = 'map_case0' # 'camera_ready_maze4' #'map_case8.yaml' # 
    cfg_path = f'/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/envs/map_cfg/{map_case}.yaml'
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
    
    # ============== add some obstacles =========================
    size_z = map_cfg_file.workspace.map_zmax
    obstacles = map_cfg_file.obstacle_info.obstacles
    i = 0
    clrs = sns.color_palette("tab10", n_colors=max(3, len(obstacles))).as_hex()
    for obstacle in obstacles:
        world_map.add_obstacle(Block(obstacle[0], obstacle[1], size_z, 
                                     obstacle[2], obstacle[3], angle=obstacle[4]*np.pi, clr=clrs[i]))
        i += 1
    world_map.finalize()

    result_path = pathlib.Path(__file__).parent.parent.joinpath('results', map_cfg_file.logging.save_pivot_path_name)
    
    solution = np.load(result_path, mmap_mode='r')
    
    pivot_path = solution

    fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(12, 4))

    world_map.visualize_passage(ax=ax[0], full_passage=False)
    res = world_map.get_path_intersection(pivot_path)
    
    for kk in range(pivot_path.shape[0]-1):
        ax[0].plot([pivot_path[kk, 0], pivot_path[1+kk, 0]], 
                   [pivot_path[kk, 1], pivot_path[1+kk, 1]], 'r-')

    pw = []
    for passage in res:
        ax[0].scatter(passage['point'].x, passage['point'].y)
        pw.append(passage['passage_width'])
        print(passage['passage_width'])
    min_pw = np.min(pw)
    print(' = ' * 15, min_pw, ' = ' * 15)

    zarr_root = zarr.open('/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/gdm_mj/train/task03_10.zarr')
    dlo_len = zarr_root['meta']['dlo_len'][0]
    keypoints = zarr_root['data']['dlo_keypoints'][:]
    keypoints = keypoints.reshape(-1, 13, 3)
    num_kp = keypoints.shape[1]

    straight_shape = keypoints[0]
    
    init_dlo_shape = keypoints[30]
    init_dlo_shape = init_dlo_shape 
    goal_dlo_shape = keypoints[55]

    seg_len = np.linalg.norm(straight_shape[0] - straight_shape[1])
    
    num_kp = goal_dlo_shape.shape[0]
    
    init_dlo_shape = init_dlo_shape - np.mean(init_dlo_shape, axis=0) + pivot_path[0]
    goal_dlo_shape = goal_dlo_shape - np.mean(goal_dlo_shape, axis=0) + pivot_path[-1]

    plot_circle(solution[0, 0], solution[0, 1], 0.01, ax[0])
    plot_circle(solution[-1, 0], solution[-1, 1], 0.01, ax[0], color='-r')

    scale = 0.01
    delta_starts = (init_dlo_shape - pivot_path[0]) 
    delta_goals = (goal_dlo_shape - pivot_path[-1]) 

    scaled_delta_starts = delta_starts * scale
    scaled_delta_goals = delta_goals * scale

    pathset_list = []
    num_path = goal_dlo_shape.shape[0]
    num_waypoints = solution.shape[0]

    for k in range(num_path):
        newpath = transfer_path_between_start_and_goal(pivot_path, scaled_delta_starts[k], scaled_delta_goals[k])
        # for i in range(num_waypoints-1):
        #     ax.plot([newpath[i, 0], newpath[i+1, 0]], 
        #             [newpath[i, 1], newpath[i+1, 1]], 'r')
        pathset_list.append(newpath)
    visualize_shape(init_dlo_shape, ax[0], clr='k', ld=1.5, s=20)
    visualize_shape(goal_dlo_shape, ax[0], clr='r', ld=1.5, s=20)

    # ================= refine the pathset =======================================
    world_map.visualize_passage(full_passage=False, ax=ax[1])
    _, _, new_pathset_list, backup_pathset_list = deform_pathset_step1(pivot_path, 
                                                                       pathset_list,
                                                                       world_map, 0.15)
    final_pathset, SegIdx = deform_pathset_step2(np.array(backup_pathset_list), np.array(new_pathset_list))

    polished_pathset = np.copy(final_pathset)
    
    # scale back to original shape (for the first phase and last phase)
    for i in range(num_path):
        single_path = final_pathset[i]
        distances = np.linalg.norm( np.diff( single_path, axis=0 ), axis=1, ord=2)
        cumulative_distances = np.concatenate([np.array([0.0]), np.cumsum(distances)])
        path_length = cumulative_distances[-1]

        first_phase = SegIdx[i][1]
        last_phase = SegIdx[i][-2]

        first_phase_waypoint_idx = first_phase[0]
        last_phase_waypoint_idx = last_phase[0]
        # print(first_phase_waypoint_idx, last_phase_waypoint_idx)
        
        # ax[1].scatter(single_path[first_phase_waypoint_idx, 0], single_path[first_phase_waypoint_idx, 1], c='k')
        # ax[1].scatter(single_path[last_phase_waypoint_idx, 0], single_path[last_phase_waypoint_idx, 1], c='b')
        
        first_sigma = cumulative_distances[first_phase_waypoint_idx] / path_length
        last_sigma = cumulative_distances[last_phase_waypoint_idx] / path_length

        # for last phase
        for j in range(last_phase_waypoint_idx, single_path.shape[0]):
            sigma = cumulative_distances[j] / path_length
            ratio = (sigma - last_sigma) / (1.0 - last_sigma)
            new_p = (delta_goals[i] - scaled_delta_goals[i]) * ratio
            # print(i, j, sigma, ratio, new_p)
            polished_pathset[i, j] = polished_pathset[i, j] + new_p

        # for first phase
        for k in range(0, first_phase_waypoint_idx):
            sigma = cumulative_distances[k] / path_length
            ratio = (first_sigma - sigma ) / (first_sigma - 0.0 )
            # print(i, k, sigma, ratio)
            new_p = (delta_starts[i] - scaled_delta_starts[i]) * ratio
            polished_pathset[i, k] = polished_pathset[i, k] + new_p
    
    # plot path set results
    world_map.visualize_passage(full_passage=False, ax=ax[1])

    for i in range(num_kp-1):
        ax[1].plot([init_dlo_shape[i][0], init_dlo_shape[i+1][0]], 
                   [init_dlo_shape[i][1], init_dlo_shape[i+1][1]], color='k', linewidth=3.0)
        ax[1].plot([goal_dlo_shape[i][0], goal_dlo_shape[i+1][0]], 
                   [goal_dlo_shape[i][1], goal_dlo_shape[i+1][1]], color='r', linewidth=3.0)
    
    clrs = sns.color_palette("coolwarm", n_colors=polished_pathset.shape[0]).as_hex()
    p = 0

    for s_path in polished_pathset:
        # initial shape
        ax[1].scatter(init_dlo_shape[p][0], init_dlo_shape[p][1], s=15, color=clrs[p], marker='o')
        ax[1].scatter(goal_dlo_shape[p][0], goal_dlo_shape[p][1], s=15, color=clrs[p], marker='o')
        
        for i in range(s_path.shape[0]-1):
            ax[1].plot([s_path[i, 0], s_path[i+1, 0]], 
                       [s_path[i, 1], s_path[i+1, 1]], c=clrs[p], linewidth=1.5)
        p += 1

    for i in range(pivot_path.shape[0]-1):
        ax[1].plot([pivot_path[i, 0], pivot_path[i+1, 0]], 
                   [pivot_path[i, 1], pivot_path[i+1, 1]], 'k-.', linewidth=1.0)
    
    ws_dir = os.path.join(ROOT_DIR, 'st_dlo_planning/results/exp_ws')
    file_to_save = os.path.join(ws_dir, f'{map_case}_spatial_path_set.pkl')
    f = open(file_to_save, "wb")
    res = {
        'spatial_path_set': polished_pathset
    }
    pickle.dump(res, f)

    # close file
    f.close()
    # exit()

    # ================= DLO configuration optimization =============================
    pathset = PathSet( polished_pathset, T=50, seg_len=seg_len)
    solver = TcDloSolver(pathset=pathset, k1=0.47836547, k2=2.18431412, tol=1e-5, max_iter=2500)
    opt_sigmas, info = solver.solve()
    solution = jnp.reshape(opt_sigmas, (pathset.T + 1, pathset.num_path))

    raw_dlo_shapes = []
    polished_dlo_shapes = []
    for i in range(0, pathset.T+1, 1):
        dlo_shape = pathset.query_dlo_shape(solution[i])
        raw_dlo_shapes.append(dlo_shape)
        dlo_shape = polish_dlo_shape(dlo_shape, k1=0.47836547, k2=2.18431412, segment_len=seg_len)
        polished_dlo_shapes.append(dlo_shape)

    world_map.visualize_passage(ax=ax[2], full_passage=False)
    clrs = np.linspace(0.0, 1.0, pathset.T+1)
    rever_clrs = np.flip(clrs)
    pathset.vis_all_path(ax[2])
    for i in range(0, pathset.T+1, 4):
        raw_dlo_shape = raw_dlo_shapes[i]
        polished_dlo_shape = polished_dlo_shapes[i]
        ax[2].plot(raw_dlo_shape[:, 0], raw_dlo_shape[:, 1], color=[clrs[i], rever_clrs[i], clrs[i]], linewidth=1)
        ax[2].plot(polished_dlo_shape[:, 0], polished_dlo_shape[:, 1], color=[clrs[i], rever_clrs[i], clrs[i]], linewidth=3)
    
    for i in range(num_kp-1):
        ax[2].plot([init_dlo_shape[i][0], init_dlo_shape[i+1][0]], 
                   [init_dlo_shape[i][1], init_dlo_shape[i+1][1]], color='k', linewidth=3.0)
        ax[2].plot([goal_dlo_shape[i][0], goal_dlo_shape[i+1][0]], 
                   [goal_dlo_shape[i][1], goal_dlo_shape[i+1][1]], color='r', linewidth=3.0)
        
    plt.savefig(f"/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/transfered_path_{map_case}.png",
                dpi=1200)
    plt.axis('equal')
    plt.show()
    
    # =========================================
    plt.figure()
    plt.plot(np.diff(solution, axis=0))
    plt.show()

    plt.figure()
    plt.plot(np.linspace(0.0, 1.0, pathset.T+1, endpoint=True), solution)
    plt.show()

    result_path = pathlib.Path(__file__).parent.parent.joinpath('results', f'{map_case}_optimal_shape_seq.npy')
    sol = np.concatenate(polished_dlo_shapes, axis=0)
    print(sol.shape)
    np.save(result_path, sol)

    # ===============================
    
    import matplotlib.animation as animation
    fig_ani = plt.figure()
    ax_ani = fig_ani.add_subplot()
    world_map.visualize_passage(ax=ax_ani, full_passage=False)

    artists = []
    for i in range(0, pathset.T+1, 1):
        dlo_shape = polished_dlo_shapes[i]
        container1 = ax_ani.plot(dlo_shape[:, 0], dlo_shape[:, 1], color=[clrs[i], rever_clrs[i], clrs[i]], linewidth=3)
        artists.append(container1)
    
    dlo_shape = pathset.query_dlo_shape(solution[0])
    ax_ani.plot(dlo_shape[:, 0], dlo_shape[:, 1], color='r', linewidth=3)
    dlo_shape = pathset.query_dlo_shape(solution[pathset.T])
    ax_ani.plot(dlo_shape[:, 0], dlo_shape[:, 1], color='r', linewidth=3)

    ani = animation.ArtistAnimation(fig=fig_ani, artists=artists, interval=200)
    pathset.vis_all_path(ax_ani)
    plt.show()
    ani.save(filename=f"/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/optimized_{map_case}.gif", writer="pillow")

    

