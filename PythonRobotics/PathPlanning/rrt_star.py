"""RRT* Implementation"""
import jax.numpy as jnp
import jax.random as random
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
import copy, math
from rrt import WorldMap, Node, RRT


class RRTStar(RRT):
    def __init__(self, connect_range, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._connect_range = connect_range
        
    def _find_near_nodes(self, new_node: Node):
        """find all the nodes in some range

        Args:
            new_node (Node): node
        """
        num_node = len(self._node_list) + 1
        node_dists = [RRT._compute_node_distance(new_node, node) for node in self._node_list]
        near_inds = [node_dists.index(i) for i in node_dists if i <= self._connect_range**2]
        
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
            
            if not self._check_node_collision(new_node):
                near_inds = self._find_near_nodes(new_node)
                # collision-free
                pass
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
    
    planner = RRTStar(2.0, start, goal, map=world_map)
    
    print(planner._connect_range, planner._goal.state)
    
    # path_solution = planner.plan()
    
    