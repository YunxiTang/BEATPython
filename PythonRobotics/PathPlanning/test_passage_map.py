if __name__ == '__main__':
    
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import jax.numpy as jnp   
from world_map import CityMap, Block
from cost_map import CityCostMapLayer
from rrt_star import RRTStar
from visualizer import Visualizer


if __name__ == '__main__':
    start = jnp.array([20., 0., 0.])
    goal = jnp.array([200., 200., 200.])

    city_map = CityMap(start=start,
                       goal=goal,
                       resolution=0.05)

    # add some obstacles
    city_map.add_obstacle(Block(30., 30., 200., 
                                50., 25., 
                                clr=[0.1, 0.5, 0.4]))
    city_map.add_obstacle(Block(30., 40., 200., 
                                125., 175., 
                                clr=[0.5, 0.5, 0.4]))
    city_map.add_obstacle(Block(30., 30., 190., 
                                100., 90., 
                                clr=[0.4, 0.5, 0.4]))
    city_map.add_obstacle(Block(30., 30., 160., 
                                120., 20., 
                                clr=[0.5, 0.5, 0.6]))
    city_map.add_obstacle(Block(40., 40., 170., 
                                30., 95., 
                                clr=[0.3, 0.3, 0.4]))
    city_map.add_obstacle(Block(20., 30., 120., 
                                70., 160., 
                                clr=[0.3, 0.3, 0.4]))
    city_map.add_obstacle(Block(20., 30., 150., 
                                160., 140., 
                                clr=[0.4, 0.3, 0.6]))
    city_map.add_obstacle(Block(30., 40., 200., 
                                180., 35., 
                                clr=[0.6, 0.3, 0.6]))
    city_map.finalize()
    
    city_map.visualize_map()

    city_cost_layer = CityCostMapLayer(city_map)
    city_cost_layer.visualize()

    exit()

    planner = RRTStar(
        connect_range=5.0,
        start_config=start,
        goal_config=goal,
        map=city_map,
        step_size=2.,
        goal_sample_rate=20,
        max_iter=2000,
        seed=4098
    )

    path_solution = planner.plan(early_stop=False)

    rviz_replay = True
    if rviz_replay:
        try:
            # set ROS Rviz and launchfile
            import subprocess
            import rospy

            subprocess.Popen(["roslaunch", ROOT_DIR + "/city_planning.launch"])
            rospy.loginfo("visualization started.")
        
        except:
            print('Failed to automatically run RVIZ. Launch it manually.')

        # visualize the result in RVIZ
        vis = Visualizer(city_map, path_solution)
        vis.visualize()

    else:
        print("To visualize the planned, start the script with the '--replay")
    