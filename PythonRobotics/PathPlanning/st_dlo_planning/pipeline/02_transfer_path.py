if __name__ == '__main__':
    import sys
    import os
    import pathlib
    import matplotlib.pyplot as plt
    import numpy as np
    import jax.numpy as jnp
    from pprint import pprint

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    from itertools import chain
    from st_dlo_planning.utils.world_map import Block, WorldMap, MapCfg
    from st_dlo_planning.utils.path_set import PathSet, transfer_path_between_start_and_goal, deform_pathset
    from st_dlo_planning.utils.world_map import plot_rectangle, plot_circle
    from st_dlo_planning.utils.path_interpolation import visualize_shape
    import jax
    jax.config.update("jax_enable_x64", True)     # enable fp64
    jax.config.update('jax_platform_name', 'cpu') # use the CPU instead of GPU

    from st_dlo_planning.temporal_config_opt.opt_solver import DloOptProblem, TcDloSolver

    import zarr
    
    map_cfg = MapCfg(resolution=0.01,
                     map_xmin=0.0,
                     map_xmax=0.8,
                     map_ymin=0.0,
                     map_ymax=0.8,
                     map_zmin=0.0,
                     map_zmax=0.02,
                     robot_size=0.02,
                     dim=3)
    
    world_map = WorldMap(map_cfg)
    # ============== add some obstacles =========================
    size_z = 0.02
    world_map.add_obstacle(Block(0.05, 0.1, size_z, 
                                 0.35, 0.25, angle=np.pi/3, clr=[0.3, 0.5, 0.4]))
    
    world_map.add_obstacle(Block(0.08, 0.15, size_z, 
                                 0.17, 0.25, angle=np.pi/4, clr=[0.3, 0.1, 0.4]))
    
    world_map.add_obstacle(Block(0.03, 0.08, size_z, 
                                 0.25, 0.4, angle=np.pi/5, clr=[0.3, 0.6, 0.6]))
    
    world_map.add_obstacle(Block(0.03, 0.07, size_z, 
                                 0.2, 0.1, angle=3*np.pi/4, clr=[0.7, 0.3, 0.4]))
    
    world_map.add_obstacle(Block(0.05, 0.07, size_z, 
                                 0.38, 0.4, angle=7*np.pi/4, clr=[0.8, 0.6, 0.8]))
    
    world_map.add_obstacle(Block(0.05, 0.05, size_z, 
                                 0.1, 0.4, angle=3*np.pi/4, clr=[0.8, 0.6, 0.6]))
    
    world_map.add_obstacle(Block(0.12, 0.05, size_z, 
                                 0.4, 0.1, angle=np.pi/7, clr=[0.4, 0.6, 0.6]))
    
    world_map.add_obstacle(Block(0.04, 0.07, size_z, 
                                 0.09, 0.1, angle=5*np.pi/4, clr=[0.4, 0.6, 0.1]))
    
    world_map.finalize()

    result_path = pathlib.Path(__file__).parent.parent.joinpath('results', 'pivot_path_4.npy')
    print(result_path)
    solution = np.load(result_path, mmap_mode='r')
    
    pivolt_path = solution

    _, ax = world_map.visualize_passage(full_passage=False)
    res = world_map.get_path_intersection(pivolt_path)
    
    pw = []
    for passage in res:
        ax.scatter(passage['point'].x, passage['point'].y)
        pw.append(passage['passage_width'])
        print(passage['passage_width'])
    print(' = ' * 30)
    print(np.min(pw))

    plt.axis('equal')
    plt.show()

    zarr_root = zarr.open('/media/yxtang/Extreme SSD/DOM_Reaseach/dobert_dataset/pretext_dataset/sim/dax/train/03_dax_dlo_10_train.zarr')

    keypoints = zarr_root['data']['keypoints']
    num_kp = keypoints.shape[1]
    init_dlo_shape = keypoints[7][1:num_kp-1:4]
    goal_dlo_shape = keypoints[52][1:num_kp-1:4]
    
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
                    [newpath[i, 1], newpath[i+1, 1]], clr[k])
        pathset_list.append(newpath)
    
    # deform the pathset
    pathset_intersects_info, pathset_and_passage_intersects, new_pathset_list = deform_pathset(pivolt_path, pathset_list, world_map, 0.15)
    
    all_intersects_info = list(chain(*list(pathset_intersects_info.values())))
    
    for passage_id, intersects in pathset_and_passage_intersects.items():
        raw_points = intersects[0]
        deformed_points = intersects[1]
        fp_idx = intersects[2]
        p = 0
        for intersect in raw_points:
            ax.scatter(intersect.x, intersect.y, c=clr[p], marker='*')
            p += 1

        p = 0
        for intersect in deformed_points:
            ax.scatter(intersect[0], intersect[1], c=clr[p], marker='h')
            p += 1

    # for intersect_info in all_intersects_info:
    #     print(intersect_info)
    #     path_num = intersect_info['path_num']
    #     path_waypoint_idx = intersect_info['path_waypoint_idx']
    #     ax.scatter(pathset_list[path_num][path_waypoint_idx, 0], 
    #                pathset_list[path_num][path_waypoint_idx, 1], c=clr[path_num], marker='^')
    #     ax.scatter(pathset_list[path_num][path_waypoint_idx+1, 0], 
    #                pathset_list[path_num][path_waypoint_idx+1, 1], c=clr[path_num], marker='o')

    
    visualize_shape(init_dlo_shape, ax, clr='r')
    visualize_shape(goal_dlo_shape, ax, clr='b')
    plt.axis('equal')
    plt.show()
    
    _, ax = world_map.visualize_passage(full_passage=False)
    s_path = new_pathset_list[5]
    for i in range(s_path.shape[0]-1):
        ax.plot([s_path[i, 0], s_path[i+1, 0]], 
                [s_path[i, 1], s_path[i+1, 1]], 'b-')

    for passage_id, intersects in pathset_and_passage_intersects.items():
        points = intersects[0]
        deformed_points = intersects[1]
        fp_idx = intersects[2]

        for intersect in deformed_points:
            ax.scatter(intersect[0], intersect[1], c='g')

        # ax.scatter(points[fp_idx[0]].x, points[fp_idx[0]].y, c='r')
        # ax.scatter(points[fp_idx[1]].x, points[fp_idx[1]].y, c='r')
    plt.axis('equal')
    plt.show()
    
    exit()
    # ================================
    pathset = PathSet( pathset_list[1:], T=80, seg_len=0.05)
    solver = TcDloSolver(pathset=pathset, k1=20.0, k2=1.0, max_iter=1200)
    
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
        # container2 = ax.scatter(dlo_shape[:, 0], dlo_shape[:, 1], color='k')
        artists.append(container1)
        # artists.append(container2)
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=200)
    pathset.vis_all_path(ax)
    ani.save(filename="./optimized_case2.gif", writer="pillow")

    plt.figure()
    plt.plot(np.diff(solution, axis=0))
    plt.show()

    plt.figure()
    plt.plot(np.linspace(0.0, 1.0, pathset.T+1, endpoint=True), solution)
    plt.plot(np.linspace(0.0, 1.0, pathset.T+1, endpoint=True), np.linspace(0.0, 1.0, pathset.T+1), 'r-.')
    plt.axis('equal')
    plt.show()


