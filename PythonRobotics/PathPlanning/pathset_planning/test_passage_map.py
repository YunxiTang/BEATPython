if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    print(ROOT_DIR)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


import jax.numpy as jnp
from world_map import CityMap, Block
from pathset_planning.cost_map import CityCostMapLayer
from rrt_star import RRTStar
# from pathset_planning.visualizer import Visualizer


if __name__ == "__main__":
    start = jnp.array([20.0, 0.0, 0.0])
    goal = jnp.array([200.0, 200.0, 200.0])

    city_map = CityMap(start=start, goal=goal, resolution=0.05)

    # add some obstacles
    city_map.add_obstacle(Block(30.0, 30.0, 200.0, 50.0, 25.0, clr=[0.1, 0.5, 0.4]))
    city_map.add_obstacle(Block(30.0, 40.0, 200.0, 125.0, 175.0, clr=[0.5, 0.5, 0.4]))
    city_map.add_obstacle(Block(30.0, 30.0, 190.0, 100.0, 90.0, clr=[0.4, 0.5, 0.4]))
    city_map.add_obstacle(Block(30.0, 30.0, 160.0, 120.0, 20.0, clr=[0.5, 0.5, 0.6]))
    city_map.add_obstacle(Block(40.0, 40.0, 170.0, 30.0, 95.0, clr=[0.3, 0.3, 0.4]))
    city_map.add_obstacle(Block(20.0, 30.0, 120.0, 70.0, 160.0, clr=[0.3, 0.3, 0.4]))
    city_map.add_obstacle(Block(20.0, 30.0, 150.0, 160.0, 140.0, clr=[0.4, 0.3, 0.6]))
    city_map.add_obstacle(Block(30.0, 40.0, 200.0, 180.0, 35.0, clr=[0.6, 0.3, 0.6]))
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
        step_size=2.0,
        goal_sample_rate=20,
        max_iter=2000,
        seed=4098,
    )

    path_solution = planner.plan(early_stop=False)

    rviz_replay = True
    # if rviz_replay:
    #     try:
    #         # set ROS Rviz and launchfile
    #         import subprocess
    #         import rospy

    #         subprocess.Popen(["roslaunch", ROOT_DIR + "/city_planning.launch"])
    #         rospy.loginfo("visualization started.")

    #     except:
    #         print('Failed to automatically run RVIZ. Launch it manually.')

    #     # visualize the result in RVIZ
    #     vis = Visualizer(city_map, path_solution)
    #     vis.visualize()

    # else:
    #     print("To visualize the planned, start the script with the '--replay")
