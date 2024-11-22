if __name__ == '__main__':
    import sys
    import os
    import pathlib
    import matplotlib.pyplot as plt
    import numpy as np
    import jax.numpy as jnp

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    from st_dlo_planning.spatial_pathset_gen.dlo_ompl import DloOmpl
    from st_dlo_planning.spatial_pathset_gen.world_map import Block, WorldMap, MapCfg
    from st_dlo_planning.spatial_pathset_gen.utils import plot_circle, transfer_path
    from st_dlo_planning.utils import PathSet
    from st_dlo_planning.utils.path_interpolation import visualize_shape
    import jax
    jax.config.update("jax_enable_x64", True)     # enable fp64
    jax.config.update('jax_platform_name', 'cpu') # use the CPU instead of GPU

    from st_dlo_planning.temporal_config_opt.opt_solver import DloOptProblem, TcDloSolver

    import zarr
    
    map_cfg = MapCfg(resolution=0.01,
                     map_xmin=0.0,
                     map_xmax=0.5,
                     map_ymin=0.0,
                     map_ymax=0.5,
                     map_zmin=0.0,
                     map_zmax=0.02,
                     robot_size=0.02,
                     dim=3)
    
    world_map = WorldMap(map_cfg)
    # ============== add some obstacles =========================
    size_z = 0.02
    world_map.add_obstacle(Block(0.05, 0.1, size_z, 
                                 0.35, 0.25, angle=np.pi/3, clr=[0.3, 0.5, 0.4]))
    
    world_map.add_obstacle(Block(0.1, 0.1, size_z, 
                                 0.16, 0.25, angle=-np.pi/4, clr=[0.3, 0.1, 0.4]))
    
    world_map.add_obstacle(Block(0.03, 0.12, size_z, 
                                 0.25, 0.4, angle=np.pi/6, clr=[0.3, 0.6, 0.6]))
    
    world_map.add_obstacle(Block(0.03, 0.07, size_z, 
                                 0.2, 0.1, angle=3*np.pi/4, clr=[0.7, 0.3, 0.4]))
    
    world_map.add_obstacle(Block(0.05, 0.07, size_z, 
                                 0.38, 0.4, angle=7*np.pi/4, clr=[0.8, 0.6, 0.8]))
    
    world_map.finalize()

    result_path = pathlib.Path(__file__).parent.parent.joinpath('results', 'pivot_path_2.npy')
    print(result_path)
    solution = np.load(result_path, mmap_mode='r')
    
    pivolt_path = solution

    zarr_root = zarr.open('/media/yxtang/Extreme SSD/DOM_Reaseach/dobert_dataset/pretext_dataset/sim/dax/train/03_dax_dlo_30_train.zarr')

    keypoints = zarr_root['data']['keypoints']
    num_kp = keypoints.shape[1]
    init_dlo_shape = keypoints[18][1:num_kp-1:2]
    goal_dlo_shape = keypoints[17][1:num_kp-1:2]
    
    init_dlo_shape = init_dlo_shape - np.mean(init_dlo_shape, axis=0) + pivolt_path[0]
    goal_dlo_shape = goal_dlo_shape - np.mean(goal_dlo_shape, axis=0) + pivolt_path[-1]

    _, ax = world_map.visualize_passage(full_passage=False)
    plot_circle(solution[0, 0], solution[0, 1], 0.01, ax)
    plot_circle(solution[-1, 0], solution[-1, 1], 0.01, ax, color='-r')

    delta_starts = init_dlo_shape - pivolt_path[0]
    delta_goals = goal_dlo_shape - pivolt_path[-1]

    pathset = []

    for k in range(goal_dlo_shape.shape[0]):
        newpath = transfer_path(pivolt_path, delta_starts[k], delta_goals[k])

        for i in range(solution.shape[0]-1):
            ax.plot([newpath[i, 0], newpath[i+1, 0]], 
                    [newpath[i, 1], newpath[i+1, 1]], 'k-')
        pathset.append(newpath)
    
    plt.axis('equal')
    plt.show()

    pathset = PathSet( pathset, T=40, seg_len=0.013)

    _, ax = world_map.visualize_passage(full_passage=False)
    pathset.vis_all_path(ax)
    visualize_shape(init_dlo_shape, ax, clr='r')
    visualize_shape(goal_dlo_shape, ax, clr='b')
    plt.show()

    # ================================
    solver = TcDloSolver(pathset=pathset, k1=100.0, k2=5.0, max_iter=1200)
    
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


