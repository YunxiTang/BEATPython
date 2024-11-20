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
    
    map_cfg = MapCfg(map_zmin=0, map_zmax=100.)
    
    world_map = WorldMap(map_cfg)
    # ============== add some obstacles =========================
    size_z = 100

    world_map.add_obstacle(Block(40., 40., size_z, 
                                 156., 43.5, clr=[0.3, 0.5, 0.4]))
    
    world_map.add_obstacle(Block(26., 15., size_z, 
                                 29., 57., clr=[0.3, 0.3, 0.4]))
    
    world_map.add_obstacle(Block(24., 15., size_z, 
                                 85., 80., clr=[0.2, 0.6, 0.4]))
    
    world_map.add_obstacle(Block(20., 20., size_z, 
                                 39., 132., clr=[0.6, 0.6, 0.8]))
    
    world_map.add_obstacle(Block(28., 15., size_z, 
                                 141., 81., clr=[0.6, 0.1, 0.4]))

    world_map.finalize()

    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # ax = fig.add_subplot(projection='3d')
    # world_map.visualize_map(ax, show_wall=True)
    # plt.show()

    z_level = 45.
    start = [40, 40, z_level]
    goal = [160., 90., z_level]
    dlo_ompl = DloOmpl(world_map, z_level)

    start_validate = world_map.check_pos_collision(start)
    goal_validate = world_map.check_pos_collision(goal)
    
    if start_validate and goal_validate:
        sol = dlo_ompl.plan(start, goal, allowed_time=60)
        print(sol)
    else:
        print('start or goal is not validate')
        sol = None

    if sol:
        states = []
        for i in range(sol.getStateCount()):
            state = sol.getState(i)
            states.append([state[0], state[1], state[2]])
            sol_np = np.array(states)
        
        # fig = plt.figure(figsize=plt.figaspect(0.5))
        # ax = fig.add_subplot(projection='3d')
        # world_map.visualize_map(ax, show_wall=False)
        # for i in range(len(states)-1):
        #     ax.plot3D([sol_np[i, 0], sol_np[i+1, 0]], 
        #               [sol_np[i, 1], sol_np[i+1, 1]],
        #               [sol_np[i, 2], sol_np[i+1, 2]], 'r-')
        # plt.axis('equal')
        # plt.show()

        ax = world_map.visualize_passage(full_passage=False)
        for i in range(len(states)-1):
            ax.plot([sol_np[i, 0], sol_np[i+1, 0]], 
                    [sol_np[i, 1], sol_np[i+1, 1]], 'r-')
        plt.axis('equal')
        plt.show()

