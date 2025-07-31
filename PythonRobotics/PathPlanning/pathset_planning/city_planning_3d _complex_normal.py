if __name__ == "__main__":
    import sys
    import os
    import pathlib
    import pickle

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


import time
from world_map import CityMap, Block
from rrt_star import CityRRTStar
import numpy as np
import jax.numpy as jnp
from PythonRobotics.PathPlanning.pathset_planning.visualizer import Visualizer
from PythonRobotics.PathPlanning.pathset_planning.cost_map import CityCostMapLayer
import matplotlib.pyplot as plt


if __name__ == "__main__":
    start = np.array([0.0, 100.0, 61.0])
    goal = jnp.array([150.0, 70.0, 74.0])

    city_map = CityMap(start=start, goal=goal, resolution=0.05)
    city_map._zmin = start[2]
    city_map._zmax = goal[2]
    # add some obstacles
    obs1 = Block(22.0, 15.0, 83.0, 88.0, 75.0, clr=[0.4, 0.5, 0.4])

    obs2 = Block(17.0, 15.0, 93.0, 63.5, 45.5, clr=[0.5, 0.5, 0.6])

    obs3 = Block(32.0, 30.0, 21.0, 102.0, 45.5, clr=[0.3, 0.3, 0.4])

    city_map.add_obstacle(obs1)
    city_map.add_obstacle(obs2)
    city_map.add_obstacle(obs3)
    city_map.add_obstacle(Block(40.0, 43.0, 75.0, 156.0, 33.5, clr=[0.3, 0.5, 0.4]))
    city_map.add_obstacle(Block(26.0, 15.0, 60.0, 29.0, 27.0, clr=[0.3, 0.3, 0.4]))
    city_map.add_obstacle(Block(30.0, 12.0, 20.0, 52.0, 72.0, clr=[0.6, 0.3, 0.4]))
    city_map.add_obstacle(Block(24.0, 15.0, 97.0, 97.0, 100.0, clr=[0.2, 0.6, 0.4]))
    city_map.add_obstacle(Block(20.0, 40.0, 88.0, 39.0, 132.0, clr=[0.6, 0.6, 0.8]))
    city_map.add_obstacle(Block(22.0, 22.0, 117.0, 106.0, 181.0, clr=[0.6, 0.2, 0.4]))
    city_map.add_obstacle(Block(28.0, 15.0, 50.0, 161.0, 81.0, clr=[0.6, 0.1, 0.4]))
    city_map.finalize()
    city_map.visualize_map()
    plt.show()
    exit()
    planner = CityRRTStar(
        connect_range=25.0,
        start_config=start,
        goal_config=goal,
        map=city_map,
        step_size=8.0,
        goal_sample_rate=10.0,
        max_iter=600,
        seed=int(time.time()),  # 333 #230 #33
    )

    with open(
        "/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/result/node_list_without_kp.pkl",
        "rb",
    ) as fp:
        planner._node_list = pickle.load(fp)

    with open(
        "/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/result/final_node_without_kp.pkl",
        "rb",
    ) as fp:
        planner._potential_final_node = pickle.load(fp)

    passage_cost_layer = CityCostMapLayer(city_map, k=-0.0)
    passage_cost_layer.visualize(instant_show=True)
    planner.add_cost_layer(passage_cost_layer)

    path_sol = planner.plan(early_stop=False, interval=50, animation=bool(0))
    if path_sol is not None:
        passage_info = [node_passage.min_dist for node_passage in path_sol[1]]
        print(passage_info)
        planner.save_result(
            path_sol[0],
            "/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/result/without_kp_2.pkl",
        )
        planner.save_result(
            planner._node_list,
            "/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/result/node_list_without_kp.pkl",
        )
        planner.save_result(
            planner._potential_final_node,
            "/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/result/final_node_without_kp.pkl",
        )

    replay = ["plt", "rviz"]  # 'rviz'
    if "plt" in replay:
        if path_sol is not None:
            path_solution = path_sol[0]
            import matplotlib.pyplot as plt

            axs = passage_cost_layer.visualize(instant_show=False)
            ax = axs[0]
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
