from __future__ import print_function
import numpy as np


class Vertex:
    def __init__(self, node):
        self._id = node
        self._adjacent = {}  # dict: key = neighbor, value = weight

    def get_id(self):
        return self._id
    
    def set_neighbor_weight(self, neighbor, weight=0):
        self._adjacent[neighbor] = weight

    def get_weight(self, neighbor):
        return self._adjacent[neighbor]
    
    def get_neighbors(self):
        return self._adjacent.keys()
    

if __name__ == '__main__':
    x = np.array([1.,2.])
    vrtx = Vertex(x)