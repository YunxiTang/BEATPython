if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    print(ROOT_DIR)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


from world_map import CityMap, Block
from pathset_planning.pro_rrtstar import PassageAwareRRTStar
import numpy as np
import jax.numpy as jnp
from pathset_planning.cost_map import CityCostMapLayer


if __name__ == '__main__':
    start = np.array([25., 25., 50.])
    goal = jnp.array([95., 95., 50.])

    city_map = CityMap(start=start,
                       goal=goal,
                       resolution=0.05)

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
    
    city_map.add_obstacle(obs1)
    city_map.add_obstacle(obs2)
    city_map.add_obstacle(obs3)
    city_map.add_obstacle(Block(20., 30., 150., 
                                110., 70., clr=[0.3, 0.5, 0.4]))
    city_map.add_obstacle(Block(20., 30., 200., 
                                150., 140., clr=[0.3, 0.3, 0.4]))
    city_map.finalize()

    planner = PassageAwareRRTStar(
        connect_range=20.0,
        start=start,
        goal=goal,
        world_map=city_map,
        step_size=10.,
        goal_sample_rate=10.,
        max_iter=3000,
        seed=33
    )

    passage_cost_layer = CityCostMapLayer(city_map, k=-100.0)
    passage_cost_layer.visualize()
    planner.add_cost_layer(passage_cost_layer)
    
    path_sol, path_nodes = planner.plan(early_stop=False, interval=200, animation=True)
    print(path_sol)
    if path_sol is not None:
        passage_info = [node_passage.min_dist for node_passage in path_nodes]
        print(passage_info)

    replay = ['plt',] # 'rviz'
    if 'plt' in replay:
        if path_sol is not None:
            path_solution = path_sol
            import matplotlib.pyplot as plt
            
            path = jnp.array(path_solution)

            axs = passage_cost_layer.visualize(instant_show=False)
            ax = axs[0]
            for node in planner._node_list:
                ax.scatter(node.state[0], node.state[1], c='k', s=0.5)
                if node.parent is not None:
                    ax.plot([node.state[0], node.parent.state[0]],
                             [node.state[1], node.parent.state[1]], 
                            'k-.', linewidth=0.5)
            
            for i in range(len(path_solution)-1):
                ax.scatter(path_sol[i][0], path_sol[i][1], color='r', zorder=1)
                ax.plot([path_sol[i][0], path_sol[i+1][0]],
                        [path_sol[i][1], path_sol[i+1][1]], 
                        'r-', linewidth=1.5)
            
            plt.show() 

    if 'rviz' in replay:
        from pathset_planning.visualizer import Visualizer
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
    
    
    