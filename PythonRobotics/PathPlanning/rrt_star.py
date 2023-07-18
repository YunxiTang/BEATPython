"""RRT* Implementation"""
import jax.numpy as jnp
import jax.random as random
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
import copy
from rrt import WorldMap, Node


class RRTStar:
    def __init__(self, 
                 start_config: jnp.ndarray,
                 goal_config: jnp.ndarray,
                 map: WorldMap,
                 step_size: float = 0.1,
                 max_iter: int = 500
                 ) -> None:
        self._start = Node(start_config)
        self._goal = Node(goal_config)
        
        self._resolution = 0.01
        
        self._map = map
        self._node_list = []
        
        self._step_size = step_size
        self._max_iter = max_iter
        self._goal_sample_rate = 5
        
        self._rng_key = random.PRNGKey(seed=4)