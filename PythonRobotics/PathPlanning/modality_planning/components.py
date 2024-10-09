from abc import abstractmethod, ABCMeta
from enum import Enum
import jax.numpy as jnp
import numpy as np
import copy
from typing import List, NamedTuple, Tuple
import random
import jax

class Terrian(Enum):
    EvenTerrain = 1
    UnEvenTerrain = 2
    Obstacle = 3
    
    
class Modality(Enum):
    Rolling = 1
    Hopping = 2
    Flipping = 3
    

class State(NamedTuple):
    x: jnp.ndarray
    mode: Modality


class ABCCostMapLayer(metaclass=ABCMeta):
    '''
        abstract cost map layer
    '''
    @abstractmethod
    def compute_node_cost(self, node):
        raise NotImplementedError
    
    @abstractmethod
    def compute_edge_cost(self, parent_node, child_node):
        raise NotImplementedError
    
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

class WorldMap:
    def __init__(self, x_min, x_max, resolution: 0.5):  
        self._resolution = resolution
        self._dim = 1
        self.x_min = x_min
        self.x_max = x_max
        num = int((x_max - x_min) / resolution) + 1
        self.points = np.linspace(x_min, x_max, num=num, endpoint=True).tolist()
        
    def get_terrain(self, x):
        if 0. <= x and x < 10.:
            return Terrian.EvenTerrain
        elif 10 <= x and x < 20.:
            return Terrian.UnEvenTerrain
        # elif 20 <= x and x < 30:
        #     return Terrian.EvenTerrain
        # elif 30. <= x and x < 35.:
        #     return Terrian.Obstacle
        else:
            return Terrian.EvenTerrain
    
    def get_modality_cost(self, x: float, mode: Modality):
        '''
            get the modality cost at a given pos `x`  in current map
        '''
        terrain = self.get_terrain(x)
        if terrain == Terrian.EvenTerrain:
            if mode == Modality.Rolling:
                cost = 10.
            elif mode == Modality.Hopping:
                cost = 30.
            elif mode == Modality.Flipping:
                cost = 800.
            else:
                raise NotImplementedError
        
        elif terrain == Terrian.UnEvenTerrain:
            if mode == Modality.Rolling:
                cost =  10000.
            elif mode == Modality.Hopping:
                cost = 100.
            elif mode == Modality.Flipping:
                cost = 1000.
            else:
                raise NotImplementedError
            
        elif terrain == Terrian.Obstacle:
            if mode == Modality.Rolling:
                cost = 10000.
            elif mode == Modality.Hopping:
                cost = 10000.
            elif mode == Modality.Flipping:
                cost = 1000.
            else:
                raise NotImplementedError
            
        else:
            raise NotImplementedError
        return cost
    
    def sample_pos(self, rng_key, toward_goal=False):
        if not toward_goal:
            x = jax.random.uniform(rng_key, shape=(1,), 
                                   minval=self.x_min,
                                   maxval=self.x_max) 
            # x = jnp.array( [random.choice(self.points)] )
        else:
            x = jnp.array([self.x_max])
        terrrian = self.get_terrain(x)
        return x, terrrian
        
    
class Node:
    """
        Node: state is consisted of (x: float, mode: Modality)
    """
    def __init__(self, state: State):
        self._state = state
        self._parent : Node = None
        
        self._node_cost = None # cost caused by node itself, depends on the motion modality
        self._path_cost = None # cost from root node to current node (include the node_cost)
        
    def set_parent(self, node: 'Node'):
        self._parent = copy.deepcopy(node)
        
    def set_node_cost(self, node_cost):
        self._node_cost = node_cost
        
    def set_path_cost(self, path_cost):
        self._path_cost = path_cost

    def reset_parent(self, node: 'Node'):
        self._parent = copy.deepcopy(node)

    def __eq__(self, other: 'Node'):
        eq = False
        dist = jnp.linalg.norm(self.state.x - other.state.x)
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
    def path_cost(self):
        return self._path_cost
    
    @property
    def node_cost(self):
        return self._node_cost
    
    
class EuclideanCostMapLayer(ABCCostMapLayer):
    '''
        Eulidean cost map layer
    '''
    def __init__(self, map: WorldMap, name: str = 'euclidean'):
        super().__init__()
        self.name = name
        self.map = map
        
    def compute_node_cost(self, node: Node = None):
        return 0.0, None

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        cost = jnp.linalg.norm(parent_node._state.x - child_node._state.x) * 10
        return cost, None
    

class ModalityCostMapLayer(ABCCostMapLayer):
    '''
        modality-related cost map layer
    '''
    def __init__(self, map: WorldMap, name: str = 'modality'):
        super().__init__()
        self.name = name
        self.map = map
        
    def compute_node_cost(self, node: Node):
        '''
            modality cost for node
        '''
        node_cost = self.map.get_modality_cost(node._state.x, node._state.mode)
        return node_cost, None

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        '''
            modality switching cost
        '''
        if parent_node._state.mode == Modality.Rolling:
            if child_node._state.mode == Modality.Rolling:
                cost = 0.0
            elif child_node._state.mode == Modality.Hopping:
                cost = 500.0
            elif child_node._state.mode == Modality.Flipping:
                cost = 10000.0
            else:
                raise NotImplementedError
            
        elif parent_node._state.mode == Modality.Hopping:
            if child_node._state.mode == Modality.Rolling:
                cost = 50.0
            elif child_node._state.mode == Modality.Hopping:
                cost = 0.0
            elif child_node._state.mode == Modality.Flipping:
                cost = 10.0
            else:
                raise NotImplementedError
            
        elif parent_node._state.mode == Modality.Flipping:
            if child_node._state.mode == Modality.Rolling:
                cost = 10000.0
            elif child_node._state.mode == Modality.Hopping:
                cost = 10.0
            elif child_node._state.mode == Modality.Flipping:
                cost = 0.0
            else:
                raise NotImplementedError
            
        return cost, None


class CostMap:
    '''
        Cost map interface
    '''
    def __init__(self):
        self._cost_layers = []

    def add_cost_layer(self, cost_layer):
        self._cost_layers.append(cost_layer)
        
    def compute_node_cost(self, node: Node):
        cost = 0.0
        aux_infos = []
        for cost_layer in self._cost_layers:
            sub_cost, aux_info = cost_layer.compute_node_cost(node)
            cost += sub_cost
            aux_infos.append(aux_info)
        return cost, aux_infos
    
    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        cost = 0.0
        aux_infos = []
        for cost_layer in self._cost_layers:
            sub_cost, aux_info = cost_layer.compute_edge_cost(parent_node, child_node)
            cost += sub_cost
            aux_infos.append(aux_info)
        return cost, aux_infos

    
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
    
    map = WorldMap(0.5)
    print(map.get_terrain(12))
    
    print(list(Modality))