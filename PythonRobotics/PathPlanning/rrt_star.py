"""RRT* Implementation"""
import jax.numpy as jnp
import jax.random as random
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
import copy
from rrt import WorldMap, Node, RRT


class NodeX(Node):
    def __init__(self, state) -> None:
        super().__init__(state)
        self._cost = 0.0
        
    @property
    def cost(self):
        return self._cost


class RRTStar(RRT):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        
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