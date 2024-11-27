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
    from st_dlo_planning.utils.world_map import Block, WorldMap, MapCfg, plot_circle
    
    map_cfg = MapCfg(resolution=0.01,
                     map_xmin=0.0,
                     map_xmax=0.8,
                     map_ymin=0.0,
                     map_ymax=0.8,
                     map_zmin=0.0,
                     map_zmax=0.02,
                     robot_size=0.01,
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

    ax = world_map.visualize_passage(full_passage=False)
    plt.axis('equal')
    plt.show()

    dlo_ompl = DloOmpl(world_map, size_z/2, k_clearance=0.01, k_passage=1.0, animation=False)

    start = [0.02, 0.02, size_z/2]
    goal = [0.45, 0.48, size_z/2]

    start_validate = world_map.check_pos_collision(start)
    goal_validate = world_map.check_pos_collision(goal)
    
    if start_validate and goal_validate:
        sol, sol_np = dlo_ompl.plan(start, goal, allowed_time=200)
        result_path = pathlib.Path(__file__).parent.parent.joinpath('results', 'pivot_path_4.npy')
        print(result_path)
        np.save(result_path, sol_np)
        print(sol)
    else:
        print( start_validate, goal_validate )
        print('start or goal is not validate')
        sol_np = None

    if sol_np is not None:
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
        
        fig.savefig('/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/pipeline/pivolt_path_0.png', dpi=2000)

