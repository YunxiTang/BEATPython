if __name__ == '__main__':
    import sys
    import os
    import pathlib
    import matplotlib.pyplot as plt
    import numpy as np

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    
    from st_dlo_planning.spatial_pathset_gen.dlo_ompl import DloOmpl
    from st_dlo_planning.spatial_pathset_gen.world_map import Block, WorldMap, MapCfg
    from st_dlo_planning.spatial_pathset_gen.utils import plot_circle
    
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

    ax = world_map.visualize_passage(full_passage=False)
    plt.axis('equal')
    plt.show()

    dlo_ompl = DloOmpl(world_map, size_z/2, k_clearance=0., k_passage=20., animation=True)

    start = [0.3, 0.1, size_z/2]
    goal = [0.3, 0.4, size_z/2]

    start_validate = world_map.check_pos_collision(start)
    goal_validate = world_map.check_pos_collision(goal)
    
    if start_validate and goal_validate:
        sol, sol_np = dlo_ompl.plan(start, goal, allowed_time=100)
        result_path = pathlib.Path(__file__).parent.parent.joinpath('results', 'pivot_path_2.npy')
        print(result_path)
        np.save(result_path, sol_np)
        print(sol)
    else:
        print('start or goal is not validate')
        sol = None

    if sol_np is not None:
        _, ax = world_map.visualize_passage(full_passage=False)
        plot_circle(start[0], start[1], 0.01, ax)
        plot_circle(goal[0], goal[1], 0.01, ax, color='-r')

        for i in range(sol_np.shape[0]-1):
            ax.plot([sol_np[i, 0], sol_np[i+1, 0]], 
                    [sol_np[i, 1], sol_np[i+1, 1]], 'r-')
        plt.axis('equal')
        plt.show()

