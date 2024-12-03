if __name__ == '__main__':
    import sys
    import os
    import pathlib
    import matplotlib.pyplot as plt
    import numpy as np
    import jax.numpy as jnp
    from pprint import pprint
    from omegaconf import OmegaConf


    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    from itertools import chain
    from st_dlo_planning.utils.world_map import Block, WorldMap, MapCfg
    from st_dlo_planning.utils.path_set import PathSet, transfer_path_between_start_and_goal, deform_pathset, deform_pathset_pro
    from st_dlo_planning.utils.world_map import plot_rectangle, plot_circle
    from st_dlo_planning.utils.path_interpolation import visualize_shape
    import jax
    jax.config.update("jax_enable_x64", True)     # enable fp64
    jax.config.update('jax_platform_name', 'cpu') # use the CPU instead of GPU

    from st_dlo_planning.temporal_config_opt.opt_solver import DloOptProblem, TcDloSolver

    import zarr
    map_case = 'map_case1'
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
    for obstacle in obstacles:
        world_map.add_obstacle(Block(obstacle[0], obstacle[1], size_z, 
                                     obstacle[2], obstacle[3], angle=obstacle[4]*np.pi, clr=[0.3+0.1*i, 0.5, 0.4]))
        i += 1
    world_map.finalize()

    result_path = pathlib.Path(__file__).parent.parent.joinpath('results', map_cfg_file.logging.save_pivot_path_name)
    
    solution = np.load(result_path, mmap_mode='r')
    
    pivolt_path = solution

    _, ax = world_map.visualize_passage(full_passage=False)
    res = world_map.get_path_intersection(pivolt_path)
    
    pw = []
    for passage in res:
        ax.scatter(passage['point'].x, passage['point'].y)
        pw.append(passage['passage_width'])
        print(passage['passage_width'])
    print(' = ' * 15, np.min(pw), ' = ' * 15)

    plt.axis('equal')
    plt.show()

    zarr_root = zarr.open('/media/yxtang/Extreme SSD/DOM_Reaseach/dobert_dataset/pretext_dataset/sim/dax/train/03_dax_dlo_10_train.zarr')

    keypoints = zarr_root['data']['keypoints']
    num_kp = keypoints.shape[1]
    init_dlo_shape = keypoints[44][2:num_kp-1:2]
    goal_dlo_shape = keypoints[46][2:num_kp-1:2]
    num_kp = goal_dlo_shape.shape[0]
    
    init_dlo_shape = init_dlo_shape - np.mean(init_dlo_shape, axis=0) + pivolt_path[0]
    goal_dlo_shape = goal_dlo_shape - np.mean(goal_dlo_shape, axis=0) + pivolt_path[-1]

    _, ax = world_map.visualize_passage(full_passage=False)
    plot_circle(solution[0, 0], solution[0, 1], 0.01, ax)
    plot_circle(solution[-1, 0], solution[-1, 1], 0.01, ax, color='-r')

    delta_starts = init_dlo_shape - pivolt_path[0]
    delta_goals = goal_dlo_shape - pivolt_path[-1]

    pathset_list = []
    num_path = goal_dlo_shape.shape[0]
    num_waypoints = solution.shape[0]

    clr = ['r', 'g', 'b', 'k', 'm', 'y']

    for k in range(num_path):
        newpath = transfer_path_between_start_and_goal(pivolt_path, delta_starts[k], delta_goals[k])
        for i in range(num_waypoints-1):
            ax.plot([newpath[i, 0], newpath[i+1, 0]], 
                    [newpath[i, 1], newpath[i+1, 1]], 'r')
        pathset_list.append(newpath)
    visualize_shape(init_dlo_shape, ax, clr='r')
    visualize_shape(goal_dlo_shape, ax, clr='b')
    plt.axis('equal')
    plt.show()
    # deform the pathset
    _, _, new_pathset_list, backup_pathset_list = deform_pathset(pivolt_path, 
                                                                 pathset_list,
                                                                 world_map, 0.013 * (num_kp-2), ax)

    _, ax = world_map.visualize_passage(full_passage=False)
    
    final_pathset = deform_pathset_pro(np.array(backup_pathset_list), 
                                       np.array(new_pathset_list))
    p = 0
    for s_path in final_pathset:
        for i in range(s_path.shape[0]-1):
            ax.plot([s_path[i, 0], s_path[i+1, 0]], 
                    [s_path[i, 1], s_path[i+1, 1]], 'r--')
        p += 1
    
    visualize_shape(init_dlo_shape, ax, clr='r')
    visualize_shape(goal_dlo_shape, ax, clr='b')
    plt.axis('equal')
    plt.show()

    # ================================
    pathset = PathSet( final_pathset, T=60, seg_len=0.013 * 2)
    solver = TcDloSolver(pathset=pathset, k1=20.0, k2=10.0, max_iter=1200)
    
    opt_sigmas, info = solver.solve()

    solution = jnp.reshape(opt_sigmas, (pathset.T + 1, pathset.num_path))

    clrs = np.linspace(0.0, 1.0, pathset.T+1)
    rever_clrs = np.flip(clrs)
    import matplotlib.animation as animation
    fig, ax = world_map.visualize_passage(full_passage=False)

    artists = []
    for i in range(0, pathset.T+1, 1):
        dlo_shape = pathset.query_dlo_shape(solution[i])
        container1 = ax.plot(dlo_shape[:, 0], dlo_shape[:, 1], color=[clrs[i], rever_clrs[i], clrs[i]], linewidth=3)
        artists.append(container1)
       
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=200)
    pathset.vis_all_path(ax)
    ani.save(filename=f"/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/optimized_{map_case}.gif", writer="pillow")

    plt.figure()
    plt.plot(np.diff(solution, axis=0))
    plt.show()

    plt.figure()
    plt.plot(np.linspace(0.0, 1.0, pathset.T+1, endpoint=True), solution)
    plt.show()


