import subprocess
import rospy
import numpy as np
import jax.numpy as jnp
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
    import sys
    import os
    import pathlib
    import pickle

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

from PythonRobotics.PathPlanning.pathset_planning.visualizer import Visualizer
from world_map import CityMap, Block
from PythonRobotics.PathPlanning.pathset_planning.cost_map import CityCostMapLayer
from rrt import SimpleNode
from utils import transfer_path

with open(
    "/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/result/v2/with_kp_s6.pkl",
    "rb",
) as fp:
    path_solution = pickle.load(fp)

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


path = jnp.array(path_solution)

passage_cost_layer = CityCostMapLayer(city_map, k=-100.0)

path_solution.reverse()

tmp_path_len = 0.0
start_node = SimpleNode(path_solution[0], tmp_path_len)
pivot_path = [start_node]
for point in path_solution[1 : len(path_solution) - 1]:
    tmp_path_len += jnp.linalg.norm(point - pivot_path[-1].state)
    node = SimpleNode(point, tmp_path_len)
    pivot_path.append(node)


pivot_path[1].set_state(
    jnp.array(
        [pivot_path[1].state[0], pivot_path[1].state[1] - 2.0, pivot_path[1].state[2]]
    )
)

# pivot_path[4].set_state(jnp.array([pivot_path[4].state[0]+1.0,
#                                    pivot_path[4].state[1]+1.0,
#                                    pivot_path[4].state[2]]))

# pivot_path[5].set_state(jnp.array([pivot_path[5].state[0]-1.0,
#                                    pivot_path[5].state[1]+2.0,
#                                    pivot_path[5].state[2]]))

pivot_path[6].set_state(
    jnp.array(
        [
            pivot_path[6].state[0],
            pivot_path[6].state[1] + 2.0,  # 2
            pivot_path[6].state[2],
        ]
    )
)

pivot_path[8].set_state(
    jnp.array(
        [
            pivot_path[8].state[0],
            pivot_path[8].state[1] + 6.0,  # 6
            pivot_path[8].state[2] + 8.0,
        ]
    )
)

pivot_path[9].set_state(
    jnp.array(
        [
            pivot_path[9].state[0] - 2,
            pivot_path[9].state[1] + 12.0,  # 12
            pivot_path[9].state[2] + 2.0,
        ]
    )
)

pivot_path[10].set_state(
    jnp.array(
        [
            pivot_path[10].state[0],
            pivot_path[10].state[1] + 8.0,  # 8
            pivot_path[10].state[2] + 2.0,
        ]
    )
)

pivot_path[11].set_state(
    jnp.array(
        [
            pivot_path[11].state[0],
            pivot_path[11].state[1] + 3.0,  # 3
            pivot_path[11].state[2] + 0.0,
        ]
    )
)

pivot_path[12].set_state(
    jnp.array(
        [
            pivot_path[12].state[0],
            pivot_path[12].state[1] + 4.0,  # 4
            pivot_path[12].state[2] + 0.0,
        ]
    )
)
# =============== transfer path ==================#
# start = np.array([0., 100., 30.])
# goal = jnp.array([150., 70., 120.])

transferred_start = jnp.array(
    [
        [0.0, 5.0, 30.0],
        [0.0, 15.0, 30.0],
        [0.0, 25.0, 30.0],
        [0.0, 35.0, 30.0],
        [0.0, 45.0, 30.0],
        [0.0, 55.0, 30.0],
        [0.0, 65.0, 30.0],
        [0.0, 75.0, 30.0],
        [0.0, 85.0, 30.0],
    ]
)
transferred_goal = jnp.array(
    [
        [195.0, 70.0, 120.0],
        [190.0, 70.0, 125.0],
        [185.0, 70.0, 130.0],
        [180.0, 70.0, 135.0],
        [175.0, 70.0, 140.0],
        [170.0, 70.0, 135.0],
        [165.0, 70.0, 130.0],
        [160.0, 70.0, 125.0],
        [155.0, 70.0, 120.0],
    ]
)

transfered_path_list = [pivot_path]
for i in range(transferred_start.shape[0]):
    transfered_path = transfer_path(
        pivot_path, transferred_start[i], transferred_goal[i]
    )
    transfered_path_list.append(transfered_path)

print(len(transfered_path_list))
# ======================= deformable the path ====================== #

# ===================== plot ===============================
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
passage_cost_layer._passage_maps[4].visualize(ax)


for transfered_path in transfered_path_list:
    clr = 0.5 + 0.5 * np.random.random(3)
    for i in range(len(transfered_path) - 1):
        ax.plot(
            [transfered_path[i].state[0], transfered_path[i + 1].state[0]],
            [transfered_path[i].state[1], transfered_path[i + 1].state[1]],
            color=clr,
            linewidth=2.2,
            zorder=1,
        )
        ax.scatter(
            [transfered_path[i].state[0], transfered_path[i + 1].state[0]],
            [transfered_path[i].state[1], transfered_path[i + 1].state[1]],
            color=clr,
            linewidth=2.2,
            zorder=1,
        )


ax.set_aspect("equal", adjustable="box")
ax.set_xlim([-5, 205])
ax.set_ylim([-5, 205])
ax.set_title(f"$z \in {passage_cost_layer._passage_maps[4]._region}$")
ax.set(xlabel="$x (m)$", ylabel="$y (m)$")
ax.xaxis.set_tick_params(labelsize=9.5)
ax.yaxis.set_tick_params(labelsize=9.5)
plt.show()

# exit()

path_list = [
    [i.state for i in transfered_path] for transfered_path in transfered_path_list
]
path_list.append([i.state for i in pivot_path])

subprocess.Popen(["roslaunch", ROOT_DIR + "/city_planning_path_set.launch"])
rospy.loginfo("visualization started.")
vis = Visualizer(city_map, path_solution=None, path_list=path_list)
vis.visualize()
# vis.visualize_with_plane()
