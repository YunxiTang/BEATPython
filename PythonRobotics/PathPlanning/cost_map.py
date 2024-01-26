from abc import abstractmethod, ABCMeta
from itertools import combinations
import jax.numpy as jnp
from typing import NamedTuple

from rrt import Node
from world_map import CityMap
import fcl
import matplotlib.pyplot as plt
from utils import plot_rectangle


class Passage(NamedTuple):
    vrtx1: list
    vrtx2: list
    min_dist: float


class PassageMapObtacle(object):
    def __init__(self, size_x, size_y, pos_x, pos_y, color='b'):
        self._size_x = size_x
        self._size_y = size_y
        self._pos_x = pos_x
        self._pos_y = pos_y

        self._geom = fcl.Box(size_x, size_y, 0.001)
        self._tf = fcl.Transform(jnp.array([pos_x, pos_y, 0.0]))

        # collision property
        self._collision_obj = fcl.CollisionObject(self._geom, self._tf)

        self._clr = color


class PassageMap(object):
    def __init__(self, region, name):
        self._region = region
        self._name = name

        self._obstacle = []

        self._passages = []

        self._finalized = False

    def add_obstacle(self, size_x, size_y, pos_x, pos_y, color='b'):
        obs = PassageMapObtacle(size_x, size_y, pos_x, pos_y, color)
        self._obstacle.append(obs)

    
    def finalize(self):
        self.construct_passage()
        self.filter_passage()
        self._finalized = True
        return None


    def construct_passage(self):
        for obs_pair in combinations(self._obstacle, 2):
            # 1
            request1 = fcl.DistanceRequest()
            result1 = fcl.DistanceResult()
            fcl.distance(obs_pair[0]._collision_obj, obs_pair[1]._collision_obj, request1, result1)

            # 2
            request2 = fcl.DistanceRequest()
            result2 = fcl.DistanceResult()
            fcl.distance(obs_pair[1]._collision_obj, obs_pair[0]._collision_obj, request2, result2)

            if result1.min_distance <= result2.min_distance:
                self._passages.append(Passage(result1.nearest_points[0], result1.nearest_points[1], result1.min_distance))
            else:
                self._passages.append(Passage(result2.nearest_points[0], result2.nearest_points[1], result2.min_distance))

    def filter_passage(self):
        """
            filter out the useless passage
        """
        self._filtered_passages = []
        for passage in self._passages:
            passage_center = [(passage.vrtx1[0] + passage.vrtx2[0]) / 2.,
                              (passage.vrtx1[1] + passage.vrtx2[1]) / 2.]
            
            # passage_half_length = jnp.linalg.norm(
            #     jnp.array([passage.vrtx1[0] - passage.vrtx2[0],
            #                passage.vrtx1[1] - passage.vrtx2[1]])
            #     ) / 2. - 0.001
            passage_half_length = passage.min_dist / 2. - 0.0001
            cylinder_g = fcl.Cylinder(passage_half_length, 0.001)
            cylinder_t = fcl.Transform(jnp.array([passage_center[0], passage_center[1], 0.0]))
            cylinder = fcl.CollisionObject(cylinder_g, cylinder_t)
            collision = False
            for obs in self._obstacle:
                request = fcl.CollisionRequest()
                result = fcl.CollisionResult()
                fcl.collide(cylinder, obs._collision_obj, request, result)
                collision = (collision or result.is_collision)
                
            if not collision:
                self._filtered_passages.append(passage)

        return self._filtered_passages


    def visualize(self, ax):
        for obs in self._obstacle:
            plot_rectangle(obs._size_x, obs._size_y, obs._pos_x, obs._pos_y, ax, color=obs._clr)
        
        for passage in self._passages:
            ax.plot([passage.vrtx1[0], passage.vrtx2[0]], [passage.vrtx1[1], passage.vrtx2[1]], 'k-.')

        for passage in self._filtered_passages:
            ax.plot([passage.vrtx1[0], passage.vrtx2[0]], [passage.vrtx1[1], passage.vrtx2[1]], 'r-')

    def __repr__(self) -> str:
        return 'PassageMap' + self._name + f' with #{len(self._obstacle)} obstacles' \
                            + f' between ({self._region[0]}, {self._region[1]})'


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
        self._num_regions = len(self._sorted_zs) - 1
        self.construct_height_regions()


    def construct_height_regions(self):
        self._regions = []
        self._passage_maps = []
        for i in range(self._num_regions):
            self._regions.append([self._sorted_zs[i], self._sorted_zs[i+1]])

        for idx, region in list(zip(range(self._num_regions), self._regions)):
            p_map = PassageMap(region, f'Layer{idx}')
            for block in self._city_map._obstacle:
                if block._size_z >= region[1]:
                    p_map.add_obstacle(block._size_x, block._size_y, block._pos_x, block._pos_y, block._color)
            p_map.finalize()
            self._passage_maps.append(p_map)


    def visualize(self):
        fig, ax = plt.subplots(1, self._num_regions)
        i = 0
        
        for p_map in self._passage_maps:
            p_map.visualize(ax[i])
            ax[i].set_aspect('equal', adjustable='box')
            ax[i].set_xlim([0, 200])
            ax[i].set_xlabel('x')
            ax[i].set_ylim([0, 200])
            ax[i].set_ylabel('y')
            ax[i].set_title(p_map._name + f'({p_map._region})')
            i += 1
        plt.show()

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