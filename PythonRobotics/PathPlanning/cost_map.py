from abc import abstractmethod, ABCMeta
import jax.numpy as jnp
from rrt import Node
from world_map import CityMap


class ABCCostMapLayer(metaclass=ABCMeta):
    
    @abstractmethod
    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        return NotImplemented
    

class EuclideanCostMapLayer(ABCCostMapLayer):

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        cost = jnp.linalg.norm(parent_node._state - child_node._state)
        return cost
    

class CityCostMapLayer(ABCCostMapLayer):
    def __init__(self, city_map: CityMap):
        self._city_map = city_map
        zs = [0.] + [block._size_z for block in city_map._obstacle] + [200.]
        self._sorted_zs = sorted(zs)
        self._sorted_z_idx = sorted(range(len(zs)), key=lambda k: zs[k])
        print(zs, '\n', self._sorted_zs, '\n', self._sorted_z_idx)

    
    # def construct_height_map(self, height):
    #     for i in self._sorted_zs:


    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        return 0


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