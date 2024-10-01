from abc import abstractmethod, ABCMeta
from enum import Enum
import numpy as np
import copy
from typing import List, NamedTuple, Tuple


class Terrian(Enum):
    EvenTerrain = 1
    UnEvenTerrain = 2
    Obstacle = 3
    
    
class Modality(Enum):
    Rolling = 1
    Hopping = 2
    Flipping = 3
    

class State(NamedTuple):
    x: float
    mode: Modality

    
class Node:
    """
        Node: state is consisted of (x: float, mode: Modality)
    """
    def __init__(self, state: State):
        self._state = state
        self._parent : State = None
        
        # For usage in RRR*
        self._cost = -100. * 200.
        
    def set_parent(self, node):
        self._parent = copy.deepcopy(node)
        
    def set_cost(self, cost):
        self._cost = cost

    def reset_parent(self, node):
        self._parent = copy.deepcopy(node)

    def __eq__(self, other: 'Node'):
        eq = False
        dist = np.linalg.norm(self.state.x - other.state.x)
        if dist < 1e-3 and self.state.mode == other.state.mode:
            eq = True
        return eq
        
    @property
    def state(self):
        return self._state
    
    @property
    def parent(self):
        return self._parent
    
    @property
    def cost(self):
        return self._cost
    
    
class ABCCostMapLayer(metaclass=ABCMeta):
    '''
        abstract cost map layer
    '''
    @abstractmethod
    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        return NotImplemented
    
    
class EuclideanCostMapLayer(ABCCostMapLayer):
    '''
        Eulidean cost map layer
    '''
    def __init__(self, name: str = 'euclidean'):
        super().__init__()
        self.name = name

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        cost = 0
        if parent_node.state.mode == None:
            pass
        return (cost, None)
    

class ModalityCostMap(ABCCostMapLayer):
    '''
        modality-related cost map
    '''
    def __init__(self, name: str = 'modality'):
        super().__init__()
        self.name = name

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        cost = np.linalg.norm(parent_node._state - child_node._state)
        return (cost, None)


class CostMap:
    '''
        Cost map interface
    '''
    def __init__(self):
        self._cost_layers = []

    def add_cost_layer(self, cost_layer):
        self._cost_layers.append(cost_layer)
    
    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        cost = 0.0
        aux_infos = []
        for cost_layer in self._cost_layers:
            sub_cost, aux_info = cost_layer.compute_edge_cost(parent_node, child_node)
            cost += sub_cost
            aux_infos.append(aux_info)
        return cost, aux_infos

from abc import abstractmethod, ABCMeta


class ABCMap(metaclass=ABCMeta):
    '''
        abstract map
    '''
    @abstractmethod
    def update_start(self):
        return NotImplemented
    
    @abstractmethod
    def update_goal(self):
        return NotImplemented
    
    @abstractmethod
    def update_start(self):
        return NotImplemented
    
    @abstractmethod
    def sample_free_pos(self):
        return NotImplemented
    
    @abstractmethod
    def check_pos_collision(self):
        return NotImplemented
    
    @abstractmethod
    def check_line_collision(self):
        return NotImplemented
    
    @abstractmethod
    def finalize(self):
        return NotImplemented


class WorldMap(ABCMap):
    def __init__(self):
        self.desc = {
            Terrian.EvenTerrain: [0, 10.],
            Terrian.UnEvenTerrain: [10., 20.],
            Terrian.Obstacle: [20, 25.]
        }  
        self._resolution = 0.2
        self._dim = 1
        
    def get_terrain(self, x):
        if 0. <= x and x < 10.:
            return Terrian.EvenTerrain
        elif 10 <= x and x < 20.:
            return Terrian.UnEvenTerrain
        elif 20. <= x and x < 25.:
            return Terrian.Obstacle
        else:
            return Terrian.EvenTerrain

    
if __name__ == '__main__':
    from pprint import pprint
    mode = Modality.Rolling
    print(mode.name, mode.value)
    state = State(5.0, Modality.Rolling)
    state2 = State(5.0, Modality.Flipping)
    print(state)
    
    node = Node(state)
    node2 = Node(state2)
    pprint(node == node2)
    
    map = WorldMap()
    print(map.get_terrain(12))