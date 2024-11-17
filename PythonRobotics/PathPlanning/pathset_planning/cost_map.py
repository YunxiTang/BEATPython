from abc import abstractmethod, ABCMeta
from itertools import combinations
import jax.numpy as jnp
from typing import NamedTuple

from rrt import Node
from world_map import CityMap
import fcl
import matplotlib.pyplot as plt
from utils import Point, check_intersection, plot_rectangle, get_intersection_point, check_on_segment


class ABCCostMapLayer(metaclass=ABCMeta):
    '''
        abstract cost map layer
    '''
    @abstractmethod
    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        return NotImplemented


class Passage(NamedTuple):
    '''
        Passage in Passage Map
    '''
    vrtx1: list
    vrtx2: list
    min_dist: float


class PassageMapObtacle(object):
    '''
        Obstacle in Passage Map
    '''
    def __init__(self, size_x, size_y, pos_x, pos_y, color='b', is_wall: bool = False):
        self._size_x = size_x
        self._size_y = size_y
        self._pos_x = pos_x
        self._pos_y = pos_y

        self._geom = fcl.Box(size_x, size_y, 0.001)
        self._tf = fcl.Transform(jnp.array([pos_x, pos_y, 0.0]))

        # collision property
        self._collision_obj = fcl.CollisionObject(self._geom, self._tf)
        
        self._is_wall = is_wall

        self._clr = color


class PassageMap(object):
    '''
        Passage Map
    '''
    def __init__(self, region, name):
        self._region = region
        self._name = name

        self._obstacle = []

        self._passages = []

        self._finalized = False

    def add_obstacle(self, size_x, size_y, pos_x, pos_y, color='b', is_wall: bool = False):
        '''
            add obstacle into the passage map
        '''
        obs = PassageMapObtacle(size_x, size_y, pos_x, pos_y, color, is_wall)
        self._obstacle.append(obs)

    
    def finalize(self):
        '''
            finalize the passage map
        '''
        self._finalized = True
        self.construct_passage()
        self.filter_passage()
        return None


    def construct_passage(self):
        assert self._finalized, 'city_map is not finalized!'
        for obs_pair in combinations(self._obstacle, 2):
            if not (obs_pair[0]._is_wall and obs_pair[1]._is_wall):
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
        assert self._finalized, 'city_map is not finalized!'
        self._filtered_passages = []
        for passage in self._passages:
            passage_center = [(passage.vrtx1[0] + passage.vrtx2[0]) / 2.,
                              (passage.vrtx1[1] + passage.vrtx2[1]) / 2.]
            
            passage_half_length = passage.min_dist / 2. - 0.0001
            cylinder_g = fcl.Cylinder(passage_half_length, 0.001)
            cylinder_t = fcl.Transform(jnp.array([passage_center[0], passage_center[1], 0.0]))
            cylinder = fcl.CollisionObject(cylinder_g, cylinder_t)
            collision = False

            for obs in self._obstacle:
                request = fcl.CollisionRequest()
                result = fcl.CollisionResult()
                fcl.collide(cylinder, obs._collision_obj, request, result)
                #
                collision = (collision or result.is_collision)
                
            if not collision:
                self._filtered_passages.append(passage)

        return self._filtered_passages
    

    def check_passage_intersection(self, parent_node: Node, child_node: Node):
        '''
            check whether the node pair intersects with some passage
        '''
        for passage in self._filtered_passages:
            intersection = check_intersection(
                Point(passage.vrtx1[0], passage.vrtx1[1]),
                Point(passage.vrtx2[0], passage.vrtx2[1]),
                Point(parent_node.state[0], parent_node.state[1]),
                Point(child_node.state[0], child_node.state[1])
            )
            if intersection:
                return passage
        return None
    

    def get_path_intersection(self, path):
        '''
            get the intersectio between the path and passage
        '''
        intersects = []
        path_len = path.shape[0]
        for i in range(path_len-1):
            for passage in self._filtered_passages:
                path_1 = Point(path[i, 0], path[i, 1])
                path_2 = Point(path[i+1, 0], path[i+1, 1])

                direction_p = jnp.array([passage.vrtx2[0] - passage.vrtx1[0], passage.vrtx2[1] - passage.vrtx1[1]])
                direction_p = direction_p / jnp.linalg.norm(direction_p)
                direction_n = jnp.array([passage.vrtx1[0] - passage.vrtx2[0], passage.vrtx1[1] - passage.vrtx2[1]])
                direction_n = direction_n / jnp.linalg.norm(direction_n)

                extended_vrtx1 = jnp.array([passage.vrtx1[0], passage.vrtx1[1]]) + 250. * direction_n
                extended_vrtx2 = jnp.array([passage.vrtx2[0], passage.vrtx2[1]]) + 250. * direction_p

                passage_1 = Point(passage.vrtx1[0], passage.vrtx1[1])
                passage_2 = Point(passage.vrtx2[0], passage.vrtx2[1])

                extended_passage_1 = Point(extended_vrtx1[0], extended_vrtx1[1])
                extended_passage_2 = Point(extended_vrtx2[0], extended_vrtx2[1])

                point = get_intersection_point(path_1,
                                               path_2,
                                               extended_passage_1,
                                               extended_passage_2)
                
                if point:
                    # if check_on_segment(path_1, point, path_2) and check_on_segment(passage_1, point, passage_2):
                    intersects.append({'passage': passage, 'point': point})
        return intersects

    def visualize(self, ax):
        assert self._finalized, 'city_map is not finalized!'
        for obs in self._obstacle:
            plot_rectangle(obs._size_x, obs._size_y, obs._pos_x, obs._pos_y, ax, color=obs._clr)
        
        # for passage in self._passages:
        #     ax.plot([passage.vrtx1[0], passage.vrtx2[0]], [passage.vrtx1[1], passage.vrtx2[1]], 'k--')

        for passage in self._filtered_passages:
            ax.plot([passage.vrtx1[0], passage.vrtx2[0]], [passage.vrtx1[1], passage.vrtx2[1]], 'r-', linewidth=1.0)


    def __repr__(self) -> str:
        return 'PassageMap' + self._name + f' with #{len(self._obstacle)} obstacles' \
                            + f' between ({self._region[0]}, {self._region[1]})'
    

class EuclideanCostMapLayer(ABCCostMapLayer):
    '''
        Eulidean cost map layer
    '''
    def __init__(self, name: str = 'euclidean'):
        super().__init__()
        self.name = name

    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        cost = jnp.linalg.norm(parent_node._state - child_node._state)
        return (cost, None)
    

class CityCostMapLayer(ABCCostMapLayer):
    '''
        passage-based city cost map
    '''
    def __init__(self, city_map: CityMap, k: float = -100.):
        
        self.name = 'city'

        self._city_map = city_map
        self._k = k

        zs = [0.] + [block._size_z for block in city_map._obstacle] + [200.]
        unique_zs = list(set(zs))
        self._sorted_zs = sorted(unique_zs)
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
                    p_map.add_obstacle(block._size_x, block._size_y, block._pos_x, block._pos_y, block._color, is_wall=block._wall)
            p_map.finalize()
            self._passage_maps.append(p_map)


    def compute_edge_cost(self, parent_node: Node, child_node: Node):
        height = parent_node.state[2]
        for passage_map in self._passage_maps:
            if height >= passage_map._region[0] and height < passage_map._region[1]:
                passage = passage_map.check_passage_intersection(parent_node, child_node)
                if passage is None:
                    # no intersection
                    edge_cost = 0. #self._k * (parent_node.min_dist - parent_node.min_dist)
                    passage_len = parent_node.min_dist
                else:
                    # intersection
                    if passage.min_dist < parent_node.min_dist:
                        edge_cost = self._k * (passage.min_dist - parent_node.min_dist)
                        passage_len = passage.min_dist
                    else:
                        edge_cost = 0. # self._k * parent_node.min_dist
                        passage_len = parent_node.min_dist
                return (edge_cost, passage_len)
            
    
    def get_path_intersection(self, path):
        intersects = []
        for passage_map in self._passage_maps:
            sub_intersects = passage_map.get_path_intersection(path)
            intersects.append(sub_intersects)
        return intersects


    def visualize(self, instant_show: bool = True):
        _, ax = plt.subplots(1, self._num_regions)
        i = 0
        
        for p_map in self._passage_maps:
            p_map.visualize(ax[i])
            ax[i].set_aspect('equal', adjustable='box')
            ax[i].set_xlim([-5, 205])
            ax[i].set_xlabel('x')
            ax[i].set_ylim([-5, 205])
            ax[i].set_ylabel('y')
            ax[i].set_title(p_map._name + f'({p_map._region})')
            i += 1
        if instant_show:
            plt.show()
        return ax


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