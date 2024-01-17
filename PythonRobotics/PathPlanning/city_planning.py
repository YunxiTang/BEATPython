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
import time
import matplotlib.pyplot as plt


if __name__ == '__main__':
    start = np.array([0., 00., 0.])
    goal = jnp.array([200., 50., 155.])

    city_map = CityMap(start=start,
                       goal=goal)

    # add some obstacles
    obs1 = Block(60., 30., 120., 100., 140., clr=[0.4, 0.5, 0.4])
    obs2 = Block(60., 40., 180., 140., 50., clr=[0.5, 0.5, 0.6])
    obs3 = Block(40., 60., 90., 30., 70., clr=[0.3, 0.3, 0.4])
    city_map.add_obstacle(obs1)
    city_map.add_obstacle(obs2)
    city_map.add_obstacle(obs3)
    city_map.finalize()

    rrt = RRTStar(
        connect_range=5.0,
        start_config=start,
        goal_config=goal,
        map=city_map,
        step_size=15.0,
        goal_sample_rate=15,
        max_iter=1000,
        seed=255
    )

    path_solution = rrt.plan()
    print(f'Path Length: {len(path_solution)}')

    if path_solution is not None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
        path = jnp.array(path_solution)
            
        # for node in rrt._node_list:
        #     ax.scatter(node.state[0], node.state[1], node.state[2], c='k')
        #     if node.parent != None:
        #         ax.plot([node.state[0], node.parent.state[0]], 
        #                 [node.state[1], node.parent.state[1]], 
        #                 [node.state[2], node.parent.state[2]],
        #                 'k-.')
        
        for i in range(len(path_solution)-1):
            ax.scatter(path[i,0], path[i,1], path[i,2], color='r', zorder=1)
            ax.plot(path[i:i+2,0], path[i:i+2,1], path[i:i+2,2], color='r', linewidth=2.5, zorder=1)
        
        plt.scatter(path[len(path_solution)-1,0], 
                    path[len(path_solution)-1,1], 
                    path[len(path_solution)-1,2])
         
        city_map.visualize(ax)
        plt.show() 
    