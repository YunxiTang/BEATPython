from abc import abstractmethod, ABCMeta
import jax.numpy as jnp
from rrt import Node


class ABCCostMapLayer(metaclass=ABCMeta):
    @abstractmethod
    def get_node_cost(self):
        return NotImplemented
    
    @abstractmethod
    def get_edge_cost(self):
        return NotImplemented
    

class EuclideanCostMapLayer(ABCCostMapLayer):
    def __init__(self, world_map):
        self._world_map = world_map

    def get_node_cost(self, node):
        cost = 0.0
        return cost
    
    def get_edge_cost(self, node1: Node, node2: Node):
        cost = jnp.linalg.norm(node1._state - node2._state)
        return cost
