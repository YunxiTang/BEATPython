"""RRT* Implementation"""
import jax.numpy as jnp
import jax.random as random
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
import copy, math
from rrt import WorldMap, RRT


class RRTStar(RRT):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def plan(self, animation=True, verbose=False):
        self._node_list.append(self._start)
        
        for i in range(self._max_iter):
            # sample a random node
            rand_node = self._get_random_node()

            # find the nearest node in the tree
            nearest_ind, _ = self._get_nearest_node(rand_node)
            nearest_node = self._node_list[nearest_ind]
            
            # get new node candidate
            new_node = self._steer(nearest_node, rand_node, self._step_size)
            new_node_cost = nearest_node.cost + self._compute_node_distance(new_node, nearest_node)
            new_node.set_cost(new_node_cost)
            print(f"Iter: {i} || No. Nodes: {len(self._node_list)}")
        
        
if __name__ == '__main__':
    
    world_map = WorldMap([0., 2., 0., 2.])
    start = jnp.array([0., 0.])
    goal = jnp.array([0.8, 0.6])
    
    world_map.update_start(start)
    world_map.update_goal(goal)
    
    obs1 = (0.5, 0.4, 0.25)
    obs2 = (0.4, 0.8, 0.2)
    obs3 = (0.8, 0.8, 0.15)
    obs4 = (1.0, 0.4, 0.15)
    world_map.add_obstacle(obs1)
    world_map.add_obstacle(obs2)
    world_map.add_obstacle(obs3)
    world_map.add_obstacle(obs4)
    
    planner = RRTStar(start, goal, map=world_map)
    
    path_solution = planner.plan()
    
    print(planner._goal.state)