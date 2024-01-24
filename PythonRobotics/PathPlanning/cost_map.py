from abc import abstractmethod, ABCMeta
import jax.numpy as jnp
from rrt import Node


class ABCCostMapLayer(metaclass=ABCMeta):
    
    @abstractmethod
    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        return NotImplemented
    

class EuclideanCostMapLayer(ABCCostMapLayer):

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        cost = jnp.linalg.norm(parent_node._state - child_node._state)
        return cost
    

class CostMap:
    def __init__(self):
        self._cost_layers = []

    def add_cost_layer(self, cost_layer):
        self._cost_layers.append(cost_layer)
    
    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        cost = 0.0
        for cost_layer in self._cost_layers:
            cost += cost_layer.compute_edge_cost(parent_node, child_node)
        return cost