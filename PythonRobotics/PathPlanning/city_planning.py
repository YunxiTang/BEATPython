if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

from world_map import CityMap, Block
from rrt import RRT
from rrt_star import RRTStar
import numpy as np
import jax.numpy as jnp
from visualizer import Visualizer


if __name__ == '__main__':
    start = np.array([0., 0., 0.])
    goal = jnp.array([200., 200., 55.])

    city_map = CityMap(start=start,
                       goal=goal)

    # add some obstacles
    obs1 = Block(30., 30., 120., 
                 100., 90., 
                 clr=[0.4, 0.5, 0.4])
    obs2 = Block(30., 20., 180., 
                 120., 50., 
                 clr=[0.5, 0.5, 0.6])
    obs3 = Block(40., 40., 90., 
                 30., 70., 
                 clr=[0.3, 0.3, 0.4])
    city_map.add_obstacle(obs1)
    city_map.add_obstacle(obs2)
    city_map.add_obstacle(obs3)
    city_map.add_obstacle(Block(20., 30., 70., 
                                70., 160., clr=[0.3, 0.3, 0.4]))
    city_map.add_obstacle(Block(20., 30., 180., 
                                150., 140., clr=[0.3, 0.3, 0.4]))
    city_map.finalize()

    rrt = RRTStar(
        connect_range=5.0,
        start_config=start,
        goal_config=goal,
        map=city_map,
        step_size=5.,
        goal_sample_rate=20,
        max_iter=2000,
        seed=495
    )

    path_solution = rrt.plan()
    print(f'Path Length: {len(path_solution)}')

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
    
    # if path_solution is not None:
    #     import matplotlib.pyplot as plt
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
    #     path = jnp.array(path_solution)
            
    #     for node in rrt._node_list:
    #         ax.scatter(node.state[0], node.state[1], node.state[2], c='k')
    #         if node.parent != None:
    #             ax.plot([node.state[0], node.parent.state[0]], 
    #                     [node.state[1], node.parent.state[1]], 
    #                     [node.state[2], node.parent.state[2]],
    #                     'k-.')
        
    #     for i in range(len(path_solution)-1):
    #         ax.scatter(path[i,0], path[i,1], path[i,2], color='r', zorder=1)
    #         ax.plot(path[i:i+2,0], path[i:i+2,1], path[i:i+2,2], color='r', linewidth=2.5, zorder=1)
        
    #     plt.scatter(path[len(path_solution)-1,0], 
    #                 path[len(path_solution)-1,1], 
    #                 path[len(path_solution)-1,2])
         
    #     city_map.visualize(ax)
    #     plt.show() 
    