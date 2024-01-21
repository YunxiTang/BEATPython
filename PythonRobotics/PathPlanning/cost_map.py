from abc import abstractmethod, ABCMeta
import jax.numpy as jnp


class ABCCostMap(metaclass=ABCMeta):
    @abstractmethod
    def get_node_cost(self):
        return NotImplemented
    
    @abstractmethod
    def get_edge_cost(self):
        return NotImplemented
