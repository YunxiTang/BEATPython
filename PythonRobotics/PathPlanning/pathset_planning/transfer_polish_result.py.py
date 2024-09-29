import subprocess
import rospy
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
from cost_map import CityCostMapLayer
from rrt import SimpleNode
from utils import transfer_path

with open("/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/result/with_kp_s5.pkl", "rb") as fp:
    path_solution = pickle.load(fp)

path_solution.reverse()

tmp_path_len = 0.
start_node = SimpleNode(path_solution[0], tmp_path_len)
pivot_path = [start_node]
for point in path_solution[1:len(path_solution)-1]:
    tmp_path_len += jnp.linalg.norm(point - pivot_path[-1].state)
    node = SimpleNode(point, tmp_path_len)
    pivot_path.append(node)


start = jnp.array([0., 100., 61.])
goal = jnp.array([150., 70., 74.])

#=============== transfer path ==================#
transfered_path_list = []
transferred_start = jnp.array([[0., 80., 61.],
                               [0., 90., 61.],
                               [0., 110., 61.],
                               [0., 120., 61.]])
transferred_goal = jnp.array([[135., 70., 65.],
                              [140., 70., 70.],
                              [160., 70., 70.],
                              [165., 70., 65.]])

for i in range(transferred_start.shape[0]):
    transfered_path = transfer_path(pivot_path, transferred_start[i], transferred_goal[i])
    transfered_path_list.append(transfered_path)

#================================================#
city_map = CityMap(start=start,
                   goal=goal,
                   resolution=0.05)
city_map._zmax = 120
# add some obstacles
obs1 = Block(22., 15., 83., 
            88., 75., clr=[0.4, 0.5, 0.4])

obs2 = Block(17., 15., 93., 
            63.5, 45.5, clr=[0.5, 0.5, 0.6])

obs3 = Block(32., 30., 21., 
            102., 45.5,  clr=[0.3, 0.3, 0.4])

city_map.add_obstacle(obs1)
city_map.add_obstacle(obs2)
city_map.add_obstacle(obs3)
city_map.add_obstacle(Block(40., 43., 75., 
                            156., 33.5, clr=[0.3, 0.5, 0.4]))
city_map.add_obstacle(Block(26., 15., 60., 
                            29., 27., clr=[0.3, 0.3, 0.4]))
city_map.add_obstacle(Block(30., 12., 20., 
                            52., 72., clr=[0.6, 0.3, 0.4]))
city_map.add_obstacle(Block(24., 15., 97., 
                            97., 100., clr=[0.2, 0.6, 0.4]))
city_map.add_obstacle(Block(20., 40., 88., 
                            39., 132., clr=[0.6, 0.6, 0.8]))
city_map.add_obstacle(Block(22., 22., 117., 
                            106., 181., clr=[0.6, 0.2, 0.4]))
city_map.add_obstacle(Block(28., 15., 50., 
                            161., 81., clr=[0.6, 0.1, 0.4]))
city_map.finalize()


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
path = jnp.array(path_solution)

# city_map.visualize_map(ax)
passage_cost_layer = CityCostMapLayer(city_map, k=-100.0)


# ======================= deformable the path ====================== #
print(transfered_path_list[0][8].state[1])
# transfered_path_list[0][8]._state = jnp.array([transfered_path_list[3][8].state[0], transfered_path_list[3][8].state[1]-50,
#                                               transfered_path_list[3][8].state[2]])
print(transfered_path_list[0][8].state)

intersects_p = passage_cost_layer._passage_maps[4].get_path_intersection(path)
intersects_1 = passage_cost_layer._passage_maps[4].get_path_intersection(jnp.array([point.state for point in transfered_path_list[0]]))
intersects_2 = passage_cost_layer._passage_maps[4].get_path_intersection(jnp.array([point.state for point in transfered_path_list[1]]))
intersects_3 = passage_cost_layer._passage_maps[4].get_path_intersection(jnp.array([point.state for point in transfered_path_list[2]]))
intersects_4 = passage_cost_layer._passage_maps[4].get_path_intersection(jnp.array([point.state for point in transfered_path_list[3]]))

for i in range(len(path_solution)-2):
    ax.scatter(path[i,0], path[i,1],  path[i,2], color='k', zorder=1)
    ax.plot(path[i:i+2,0], path[i:i+2,1], path[i:i+2,2], color='r', linewidth=2.0, zorder=1)
    k = 0.2
    p = 0.8
    for transfered_path in transfered_path_list:
        clr = [0.6, p, k]
        for i in range(len(transfered_path)-1):
            ax.plot([transfered_path[i].state[0], transfered_path[i+1].state[0]], 
                    [transfered_path[i].state[1], transfered_path[i+1].state[1]], 
                    [transfered_path[i].state[2], transfered_path[i+1].state[2]],
                    color=clr, linewidth=2.2, zorder=1)
        k += 0.15
        p -= 0.25

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-5, 205])
        ax.set_ylim([-5, 205])
        ax.set_zlim([0, 120])
# ================================================================== #

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
passage_cost_layer._passage_maps[4].visualize(ax)

for point in intersects_p:
    ax.scatter(point.x, point.y, color='r', zorder=1)
for point in intersects_1:
    ax.scatter(point.x, point.y, color='r', zorder=1)
for point in intersects_2:
    ax.scatter(point.x, point.y, color='r', zorder=1)
for point in intersects_3:
    ax.scatter(point.x, point.y, color='r', zorder=1)
for point in intersects_4:
    ax.scatter(point.x, point.y, color='r', zorder=1)


for i in range(len(path_solution)-1):
    ax.plot(path[i:i+2,0], path[i:i+2,1], 'k-', linewidth=2.2, zorder=1)
    ax.scatter(path[i:i+2,0], path[i:i+2,1], linewidth=2.2, zorder=1)

k = 0.2
p = 0.8
for transfered_path in transfered_path_list:
    clr = [0.8, p, k]
    for i in range(len(transfered_path)-1):
        ax.plot([transfered_path[i].state[0], transfered_path[i+1].state[0]], 
                [transfered_path[i].state[1], transfered_path[i+1].state[1]], color=clr, linewidth=2.2, zorder=1)
        ax.scatter([transfered_path[i].state[0], transfered_path[i+1].state[0]], 
                [transfered_path[i].state[1], transfered_path[i+1].state[1]], color=clr, linewidth=2.2, zorder=1)
    k += 0.15
    p -= 0.25

ax.set_aspect('equal', adjustable='box')
ax.set_xlim([-5, 205])
ax.set_ylim([-5, 205])
ax.set_title(f'$z \in {passage_cost_layer._passage_maps[4]._region}$')
ax.set(xlabel='$x (m)$', ylabel='$y (m)$')
ax.xaxis.set_tick_params(labelsize = 9.5)
ax.yaxis.set_tick_params(labelsize = 9.5)
plt.show()

exit()
subprocess.Popen(["roslaunch", ROOT_DIR + "/city_planning_path_set.launch"])
rospy.loginfo("visualization started.")

path_list = [[i.state for i in transfered_path] for transfered_path in transfered_path_list]
path_list.append([i for i in path_solution])
vis = Visualizer(city_map, path_list=path_list)
vis.visualize()