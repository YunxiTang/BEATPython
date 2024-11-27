if __name__ == '__main__':
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    print(ROOT_DIR)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    
    from PythonRobotics.PathPlanning.st_dlo_planning.utils.world_map import WorldMap, Block
    from st_dlo_planning.spatial_pathset_gen.configuration_map import MapCfg
    import matplotlib.pyplot as plt
    
    world_map = WorldMap(MapCfg())
    # add some obstacles
    obs1 = Block(30., 30., 150., 
                 50., 90., 
                 clr=[0.4, 0.5, 0.4])
    
    obs2 = Block(30., 20., 150., 
                 70., 60., 
                 clr=[0.5, 0.5, 0.6])
    
    obs3 = Block(40., 40., 150., 
                 80., 170., 
                 clr=[0.3, 0.3, 0.4])
    
    world_map.add_obstacle(obs1)
    world_map.add_obstacle(obs2)
    world_map.add_obstacle(obs3)
    world_map.add_obstacle(Block(20., 30., 150., 
                                110., 70., clr=[0.3, 0.5, 0.4]))
    world_map.add_obstacle(Block(20., 30., 200., 
                                150., 140., clr=[0.3, 0.3, 0.4]))
    world_map.finalize()
    
    world_map.visualize_map(show_wall=True)
    plt.show()
