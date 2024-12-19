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
    
    from st_dlo_planning.spatial_pathset_gen.dlo_ompl import DloOmpl
    from st_dlo_planning.utils.world_map import Block, WorldMap, MapCfg, plot_circle

    map_id = 'map_case10.yaml'

    cfg_path = f'/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/envs/map_cfg/{map_id}'
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
    print(size_z)
    obstacles = map_cfg_file.obstacle_info.obstacles
    i = 0
    for obstacle in obstacles:
        world_map.add_obstacle(Block(obstacle[0], obstacle[1], size_z, 
                                     obstacle[2], obstacle[3], angle=obstacle[4]*np.pi, clr=[0.0+0.05*i, 0.5, 0.4]))
        i += 1
    
    world_map.finalize()

    ax = world_map.visualize_passage(full_passage=False)
    plt.axis('equal')
    plt.show()

    dlo_ompl = DloOmpl(world_map, size_z/2, k_clearance=1.0, k_passage=0.0, k_pathLen=100., animation=False)

    start = [map_cfg_file.dlo_cfg.start[0], map_cfg_file.dlo_cfg.start[1], size_z/2]
    goal = [map_cfg_file.dlo_cfg.goal[0], map_cfg_file.dlo_cfg.goal[1], size_z/2]

    start_validate = world_map.check_pos_collision(start)
    goal_validate = world_map.check_pos_collision(goal)
    
    if start_validate and goal_validate:
        sol, sol_np = dlo_ompl.plan(start, goal, allowed_time=20, num_waypoints=50)
        result_path = pathlib.Path(__file__).parent.parent.joinpath('results', map_cfg_file.logging.save_pivot_path_name)
        np.save(result_path, sol_np)
        print(sol)
    else:
        print( start_validate, goal_validate )
        print('start or goal is not validate')
        sol_np = None

    if sol_np is not None:
        print(sol_np.shape)
        fig, ax = world_map.visualize_passage(full_passage=False)
        plot_circle(start[0], start[1], 0.005, ax)
        plot_circle(goal[0], goal[1], 0.005, ax, color='-r')

        for i in range(sol_np.shape[0]-1):
            ax.plot([sol_np[i, 0], sol_np[i+1, 0]], 
                    [sol_np[i, 1], sol_np[i+1, 1]], 'r-')

        res = world_map.get_path_intersection(sol_np)
        pw = []
        for passage in res:
            ax.scatter(passage['point'].x, passage['point'].y)
            pw.append(passage['passage_width'])
            print(passage['passage_width'])
        print(' = ' * 30)
        print(np.min(pw))

        plt.axis('equal')
        plt.show()
        ax.set_xlim(world_map.map_cfg.map_xmin - 0.05, world_map.map_cfg.map_xmax + 0.05)
        ax.set_ylim(world_map.map_cfg.map_ymin - 0.05, world_map.map_cfg.map_ymax + 0.05)
