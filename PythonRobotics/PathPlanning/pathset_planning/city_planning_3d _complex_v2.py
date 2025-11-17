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
import pickle
import matplotlib.pyplot as plt
import random


def setup_seed(seed):
    """set all the seed

    Args:
        seed (int): seed
    """
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    seed = 695
    setup_seed(seed)
    print(seed)
    start = np.array([0.0, 100.0, 30.0])
    goal = jnp.array([150.0, 70.0, 120.0])

    city_map = CityMap(start=start, goal=goal, resolution=0.05)
    city_map._zmin = start[2]
    city_map._zmax = goal[2]

    # add some obstacles
    n = 10
    pos_xs = np.linspace(10, 190, n)
    pos_ys = np.linspace(10, 190, n)

    for i in range(10):
        x_idx = random.choice(range(n))
        y_idx = random.choice(range(n))
        pos_x = pos_xs[x_idx]
        pos_y = pos_ys[y_idx]

        size_x = np.random.uniform(10, 25)
        size_y = np.random.uniform(10, 25)
        size_z = np.random.uniform(40, 180)
        c = 0.5 + 0.5 * np.random.random(3)
        city_map.add_obstacle(Block(size_x, size_y, size_z, pos_x, pos_y, clr=c))

    city_map.finalize()
    city_map.visualize_map()

    seed = int(time.time())
    setup_seed(seed)
    planner = CityRRTStar(
        connect_range=25.0,
        start_config=start,
        goal_config=goal,
        map=city_map,
        step_size=1.0,
        goal_sample_rate=1.0,
        max_iter=100,
        seed=seed,
    )

    with open(
        "/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/result/v2/node_list_with_kp_s6.pkl",
        "rb",
    ) as fp:
        planner._node_list = pickle.load(fp)

    with open(
        "/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/result/v2/final_node_s6.pkl",
        "rb",
    ) as fp:
        planner._potential_final_node = pickle.load(fp)

    passage_cost_layer = CityCostMapLayer(city_map, k=-100.0)
    passage_cost_layer.visualize(instant_show=True)
    planner.add_cost_layer(passage_cost_layer)

    path_sol = planner.plan(early_stop=False, interval=50, animation=bool(0))

    if path_sol is not None:
        passage_info = [node_passage.min_dist for node_passage in path_sol[1]]
        print(passage_info)
        planner.save_result(
            path_sol[0],
            "/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/result/v2/with_kp_s7.pkl",
        )
        planner.save_result(
            planner._node_list,
            "/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/result/v2/node_list_with_kp_s7.pkl",
        )
        planner.save_result(
            planner._potential_final_node,
            "/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/result/v2/final_node_s7.pkl",
        )

    replay = ["plt", "rviz"]  #
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
