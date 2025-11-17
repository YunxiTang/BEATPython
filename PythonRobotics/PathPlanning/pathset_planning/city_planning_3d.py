if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


from world_map import CityMap, Block
from rrt_star import CityRRTStar
import numpy as np
import jax.numpy as jnp
from PythonRobotics.PathPlanning.pathset_planning.visualizer import Visualizer
from PythonRobotics.PathPlanning.pathset_planning.cost_map import CityCostMapLayer
import time


if __name__ == "__main__":
    start = np.array([0.0, 0.0, 100.0])
    goal = jnp.array([75.0, 125.0, 100.0])

    city_map = CityMap(start=start, goal=goal, resolution=0.05)
    city_map._zmin = start[2]
    city_map._zmax = goal[2]
    # add some obstacles
    obs1 = Block(30.0, 30.0, 100.0, 50.0, 100.0, clr=[0.4, 0.5, 0.4])

    obs2 = Block(30.0, 20.0, 110.0, 70.0, 60.0, clr=[0.5, 0.5, 0.6])
    obs3 = Block(40.0, 40.0, 130.0, 80.0, 170.0, clr=[0.3, 0.3, 0.4])

    city_map.add_obstacle(obs1)
    city_map.add_obstacle(obs2)
    city_map.add_obstacle(obs3)
    # city_map.add_obstacle(Block(20., 30., 150.,
    #                             110., 70., clr=[0.3, 0.5, 0.4]))
    # city_map.add_obstacle(Block(20., 30., 200.,
    #                             150., 140., clr=[0.3, 0.3, 0.4]))
    city_map.finalize()

    planner = CityRRTStar(
        connect_range=15.0,
        start_config=start,
        goal_config=goal,
        map=city_map,
        step_size=5.0,
        goal_sample_rate=2.0,
        max_iter=500,
        seed=int(time.time() // 150),  # 333 #230 #33
    )

    passage_cost_layer = CityCostMapLayer(city_map, k=-100.0)

    planner.add_cost_layer(passage_cost_layer)

    path_sol = planner.plan(early_stop=False, interval=200, animation=True)

    if path_sol is not None:
        passage_info = [node_passage.min_dist for node_passage in path_sol[1]]
        print(passage_info)

    replay = ["plt", "rviz"]  # 'rviz'
    if "plt" in replay:
        if path_sol is not None:
            path_solution = path_sol[0]
            path_sol_nodes = path_sol[1]
            import matplotlib.pyplot as plt

            axs = passage_cost_layer.visualize(instant_show=False)
            ax = axs[0]

            intersects = passage_cost_layer.get_path_intersection(path_sol_nodes)
            path = jnp.array(path_solution)

            for node in planner._node_list:
                ax.scatter(node.state[0], node.state[1], c="k", s=0.5)
                if node.parent is not None:
                    ax.plot(
                        [node.state[0], node.parent.state[0]],
                        [node.state[1], node.parent.state[1]],
                        "k-.",
                        linewidth=0.5,
                    )

            for layer in intersects:
                for point in layer:
                    ax.scatter(point.x, point.y, c="g", s=5.5)

            for i in range(len(path_solution) - 1):
                ax.scatter(path[i, 0], path[i, 1], color="r", zorder=1)
                ax.plot(
                    path[i : i + 2, 0],
                    path[i : i + 2, 1],
                    color="r",
                    linewidth=2.5,
                    zorder=1,
                )

            ax.scatter(path[len(path_solution) - 1, 0], path[len(path_solution) - 1, 1])

            plt.show()

    if "rviz" in replay:
        try:
            # set ROS Rviz and launchfile
            import subprocess
            import rospy

            subprocess.Popen(["roslaunch", ROOT_DIR + "/city_planning.launch"])
            rospy.loginfo("visualization started.")

        except:
            print("Failed to automatically run RVIZ. Launch it manually.")

        # visualize the result in RVIZ
        vis = Visualizer(city_map, path_solution)
        vis.visualize()
