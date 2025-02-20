if __name__ == '__main__':
    import sys
    import os
    import pathlib
    import matplotlib.pyplot as plt
    import numpy as np
    from omegaconf import OmegaConf
    import zarr
    import pickle
    import jax
    import jax.numpy as jnp
    import seaborn as sns
    import matplotlib.animation as animation

    jax.config.update("jax_enable_x64", True)     # enable fp64
    jax.config.update('jax_platform_name', 'cpu') # use the CPU instead of GPU

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    from st_dlo_planning.utils.world_map import Block, WorldMap, MapCfg, plot_circle
    from st_dlo_planning.utils.misc_utils import setup_seed

    from st_dlo_planning.utils.path_set import (PathSet, transfer_path_between_start_and_goal, 
                                                deform_pathset_step1, deform_pathset_step2)
    from st_dlo_planning.spatial_pathset_gen.dlo_ompl import DloOmpl

    from st_dlo_planning.utils.path_interpolation import visualize_shape
    from st_dlo_planning.temporal_config_opt.opt_solver import TcDloSolver
    from st_dlo_planning.temporal_config_opt.qp_solver import polish_dlo_shape
    
    # ============== load environment configuration ============== 
    map_id = 'hw_so1'
    cfg_path = os.path.join(ROOT_DIR, f'st_dlo_planning/hw_exp/map_cfg/{map_id}.yaml')
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
    print(size_z)

    obstacles = map_cfg_file.obstacle_info.obstacles
    clrs = sns.color_palette("icefire", n_colors=max(3, len(obstacles))).as_hex()
    for i, obstacle in enumerate(obstacles):
        world_map.add_obstacle(Block(obstacle[0], obstacle[1], size_z, 
                                     obstacle[2], obstacle[3], angle=obstacle[4]*np.pi, clr=clrs[i]))
    world_map.finalize()

    ax = world_map.visualize_passage(full_passage=False)
    plt.axis('equal')
    plt.show()

    # 2. the pivot path start and goal point
    start = [map_cfg_file.dlo_cfg.start[0], map_cfg_file.dlo_cfg.start[1], size_z/2]
    goal = [map_cfg_file.dlo_cfg.goal[0], map_cfg_file.dlo_cfg.goal[1], size_z/2]

    start_validate = world_map.check_pos_collision(start)
    goal_validate = world_map.check_pos_collision(goal)

    # ============== pivot path planning =========================
    if start_validate and goal_validate:
        dlo_ompl = DloOmpl(world_map, size_z/2, k_clearance=2, k_passage=1.0, k_pathLen=1., animation=False)
        sol, sol_np = dlo_ompl.plan(start, goal, allowed_time=50, num_waypoints=50)
        result_path = os.path.join(ROOT_DIR, 'st_dlo_planning/results/realworld_result/pivot_path_res', 
                                   map_cfg_file.logging.save_pivot_path_name)
        np.save(result_path, sol_np)
        print(f'Save optimal pivot @: {result_path}.')
    else:
        print( start_validate, goal_validate )
        print('start or goal is not validate. Exit ...')
        sol_np = None
        exit()

    ax = world_map.visualize_passage(full_passage=False)
    plot_circle(start[0], start[1], 0.008, ax, color='-b')
    plot_circle(goal[0], goal[1], 0.008, ax, color='-r')

    for i in range(sol_np.shape[0]-1):
        ax.plot([sol_np[i, 0], sol_np[i+1, 0]], 
                [sol_np[i, 1], sol_np[i+1, 1]], 'r-')

    plt.axis('equal')
    ax.set_xlim(world_map.map_cfg.map_xmin - 0.05, world_map.map_cfg.map_xmax + 0.05)
    ax.set_ylim(world_map.map_cfg.map_ymin - 0.05, world_map.map_cfg.map_ymax + 0.05)
    plt.show()

    # ================ deformation sequence optimization ====================
    pivot_path = sol_np
    fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(12, 4))
    # the inititial/goal shapes (TODO: replace with the true values)
    
    zarr_file = os.path.join(ROOT_DIR, 'st_dlo_planning/results/gdm_mj/train/task03_10.zarr')
    zarr_root = zarr.open(zarr_file)
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

    world_map.visualize_passage(ax=ax[0], full_passage=False)
    plot_circle(pivot_path[0, 0], pivot_path[0, 1], 0.01, ax[0])
    plot_circle(pivot_path[-1, 0], pivot_path[-1, 1], 0.01, ax[0], color='-r')

    scale = 0.01
    delta_starts = (init_dlo_shape - pivot_path[0]) 
    delta_goals = (goal_dlo_shape - pivot_path[-1]) 

    scaled_delta_starts = delta_starts * scale
    scaled_delta_goals = delta_goals * scale

    pathset_list = []
    num_path = goal_dlo_shape.shape[0]
    num_waypoints = pivot_path.shape[0]

    for k in range(num_path):
        newpath = transfer_path_between_start_and_goal(pivot_path, scaled_delta_starts[k], scaled_delta_goals[k])
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
        
        first_sigma = cumulative_distances[first_phase_waypoint_idx] / path_length
        last_sigma = cumulative_distances[last_phase_waypoint_idx] / path_length

        # for last phase
        for j in range(last_phase_waypoint_idx, single_path.shape[0]):
            sigma = cumulative_distances[j] / path_length
            ratio = (sigma - last_sigma) / (1.0 - last_sigma)
            new_p = (delta_goals[i] - scaled_delta_goals[i]) * ratio
            polished_pathset[i, j] = polished_pathset[i, j] + new_p

        # for first phase
        for k in range(0, first_phase_waypoint_idx):
            sigma = cumulative_distances[k] / path_length
            ratio = (first_sigma - sigma ) / (first_sigma - 0.0 )
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
    
    file_to_save = os.path.join(ROOT_DIR, 
                                'st_dlo_planning/results/realworld_result/spatial_path_set_res',
                                f'{map_id}_spatial_path_set.pkl')
    f = open(file_to_save, "wb")
    res = {'spatial_path_set': polished_pathset}
    pickle.dump(res, f)
    f.close() # close file

    # ================= DLO configuration optimization =============================
    pathset = PathSet( polished_pathset, T=50, seg_len=seg_len)
    solver = TcDloSolver(pathset=pathset, k1=10.0, k2=1.0, tol=1e-5, max_iter=2500)
    opt_sigmas, info = solver.solve()
    solution = jnp.reshape(opt_sigmas, (pathset.T + 1, pathset.num_path))

    raw_dlo_shapes = []
    polished_dlo_shapes = []
    for i in range(0, pathset.T+1, 1):
        dlo_shape = pathset.query_dlo_shape(solution[i])
        raw_dlo_shapes.append(dlo_shape)
        dlo_shape = polish_dlo_shape(dlo_shape, k1=10.0, k2=1.0, segment_len=seg_len)
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
        
    fig_path = os.path.join(ROOT_DIR, 
                            'st_dlo_planning/results/realworld_result/path_transfer_figs',
                            f'transfered_path_{map_id}.png')
    plt.savefig(fig_path, dpi=1200)
    plt.axis('equal')
    plt.show()
    
    # =========================================
    plt.figure()
    plt.plot(np.diff(solution, axis=0))
    plt.show()

    plt.figure()
    plt.plot(np.linspace(0.0, 1.0, pathset.T+1, endpoint=True), solution)
    plt.show()

    result_path = os.path.join(ROOT_DIR,
                               'st_dlo_planning/results/realworld_result/optimal_shape_seq_res',
                               f'{map_id}_optimal_shape_seq.npy')
    sol = np.concatenate(polished_dlo_shapes, axis=0)
    print(sol.shape)
    np.save(result_path, sol)

    # ===============================
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
    ax_ani.axis('equal')
    plt.show()
    ani_file_name = os.path.join(ROOT_DIR,
                                'st_dlo_planning/results/realworld_result/optimal_shape_seq_gif',
                                f'optimized_{map_id}.gif')
    ani.save(filename=ani_file_name, writer="pillow")

    