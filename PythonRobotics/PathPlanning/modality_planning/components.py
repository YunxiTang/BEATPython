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