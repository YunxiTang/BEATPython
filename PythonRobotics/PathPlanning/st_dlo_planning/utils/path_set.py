import jax
import jax.numpy as jnp
from typing import List
# import numpy 
from .path_interpolation import query_point_from_path
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from .world_map import WorldMap, get_intersection_point, Point
from pprint import pprint
from itertools import chain
import copy


class PathSet:
    def __init__(self, all_path, T:int, seg_len:float):
        self.all_path = jnp.array(all_path)
        self.num_path = self.all_path.shape[0]
        self.T = T
        self.seg_len = seg_len

        self.query_dlo_shape_fn = jax.jit( jax.vmap(query_point_from_path, in_axes=[0, 0]) )
        
    def T(self):
        return self.T
    
    def all_path(self):
        return self.all_path
    
    def query_dlo_shape(self, sigma):
        '''
            given a sigma [sigma_1, ..., sigma_n], return a dlo shape
        '''
        dlo_shape = self.query_dlo_shape_fn(sigma, self.all_path)
        return dlo_shape

    def vis_all_path(self, ax):
        vec_smooth_traj_fn = jax.vmap(query_point_from_path, in_axes=[0, None, None])
        sigmas = jnp.linspace(0, 1, 100)
        for waypoints in self.all_path:   
            trajectory = vec_smooth_traj_fn(sigmas, waypoints, 30)
            ax.plot(trajectory[:, 0], trajectory[:, 1], '-', label="Smooth Path")
            # plt.plot(waypoints[:, 0], waypoints[:, 1], 'k-.', label="Raw Path")
            # plt.scatter(waypoints[:, 0], waypoints[:, 1], color='k', label="Waypoints")
        plt.axis('equal')


# ============================================================================= #
def transfer_path_between_start_and_goal(pivot_path, delta_start, delta_goal):
    '''
        transfer the pivot path between start and goal
    '''
    pivot_path_num = pivot_path.shape[0]

    # clculate distances between consecutive waypoints
    distances = jnp.linalg.norm( jnp.diff( pivot_path, axis=0 ), axis=1, ord=2)

    # compute chord length along the pivolt path
    chord_distances = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(distances)])

    pivot_arc_len = chord_distances[-1]
    
    new_path = []
    for i in range(pivot_path_num):
        tmp_state = pivot_path[i] + chord_distances[i] / pivot_arc_len * (delta_goal - delta_start) + delta_start
        new_path.append(tmp_state[None])
    return np.concatenate(new_path, axis=0)


def redistribute_points_with_ratios(points, 
                                    original_start, original_end,
                                    new_start, new_end, max_dist=None):
    # compute distance ratios
    total_distance = np.linalg.norm( original_end - original_start )
    
    # ratios = np.linalg.norm( points - original_start, axis=1 ) / total_distance
    ratios = np.linspace(0.0, 1.0, len(points), endpoint=True)

    # map points to the new segment
    if max_dist:
        scale_factor = min(np.linalg.norm(new_end - new_start), max_dist) / np.linalg.norm(new_end - new_start)
    else:
        scale_factor = 1.0
    new_segment_vector = (new_end - new_start) * scale_factor
    redistributed_points = [new_start + r * new_segment_vector for r in ratios]
    return redistributed_points


def deform_pathset(pivot_path, pathset_list: List[np.ndarray], world_map: WorldMap, max_pw=None):
    '''
        deform the pathset to obtain feasibility
    '''
    pathset_list = copy.deepcopy(pathset_list)
    num_waypoints = pivot_path.shape[0]
    num_path = len(pathset_list)
    z_const = pathset_list[0][0, 2]

    # ============= get the related passages ============================ #
    central_intersections = world_map.get_path_intersection(pivot_path)
    passages = []
    for intersects in central_intersections:
        passages.append( intersects['passage'] )

    num_passage = len(passages)

    # get the intersects between the pathset and the passages
    pathset_intersects = dict()
    
    for passage_num in range(num_passage):
        passage = passages[passage_num]
        direction_p = np.array([passage.vrtx2[0] - passage.vrtx1[0], passage.vrtx2[1] - passage.vrtx1[1]])
        direction_p = direction_p / np.linalg.norm(direction_p)
        direction_n = np.array([passage.vrtx1[0] - passage.vrtx2[0], passage.vrtx1[1] - passage.vrtx2[1]])
        direction_n = direction_n / np.linalg.norm(direction_n)

        extended_vrtx1 = np.array([passage.vrtx1[0], passage.vrtx1[1]]) + 250. * direction_n
        extended_vrtx2 = np.array([passage.vrtx2[0], passage.vrtx2[1]]) + 250. * direction_p

        extended_passage_1 = Point(extended_vrtx1[0], extended_vrtx1[1])
        extended_passage_2 = Point(extended_vrtx2[0], extended_vrtx2[1])

        intersects_on_this_passage = []
        
        for path_num in range(num_path):
            single_path = pathset_list[path_num]
            for i in range(num_waypoints-1):
                path_1 = Point(single_path[i, 0], single_path[i, 1])
                path_2 = Point(single_path[i+1, 0], single_path[i+1, 1])
                point = get_intersection_point(path_1, path_2, extended_passage_1, extended_passage_2)
                if point:
                    intersects_on_this_passage.append({'passage': passage, 
                                                       'intersect_point': point, 
                                                       'passage_num': passage_num,
                                                       'path_num': path_num, 
                                                       'path_waypoint_idx': i})
                    break
        pathset_intersects[f'passage_{passage_num}'] = intersects_on_this_passage

    pprint(pathset_intersects)

    # ========== get the re-distributed points =================
    pathset_and_passage_intersects = dict()
    
    for passage_id, each_passage_intersects in pathset_intersects.items():
        raw_points = []
        for each_intersect in each_passage_intersects:
            passage_vrtx1 = np.array(each_intersect['passage'].vrtx1)[0:2]
            passage_vrtx2 = np.array(each_intersect['passage'].vrtx2)[0:2]
            point = each_intersect['intersect_point']
            raw_points.append(point)

        raw_points_np = np.array([[point.x, point.y] for point in raw_points])
        
        # set the vertex which is closer to the intersects as passage start
        if np.sum(np.linalg.norm(raw_points_np - passage_vrtx1, axis=1)) < np.sum(np.linalg.norm(raw_points_np - passage_vrtx2, axis=1)):
            passage_start = passage_vrtx1
            passage_end = passage_vrtx2
        else:
            passage_start = passage_vrtx2
            passage_end = passage_vrtx1

        distances = np.linalg.norm(raw_points_np[:, None] - raw_points_np, axis=2)
        endpoint_indices = np.unravel_index(np.argmax(distances), distances.shape)

        endpoint1 = raw_points_np[endpoint_indices[0]]
        endpoint2 = raw_points_np[endpoint_indices[1]]

        sign_check = (endpoint1 - endpoint2).T @ (passage_start - passage_end)
        print(sign_check)
        if sign_check < 0:
            tmp = endpoint1
            endpoint1 = endpoint2
            endpoint2 = tmp
        
        redistributed_points = redistribute_points_with_ratios(raw_points_np, 
                                                               endpoint1, endpoint2, 
                                                               passage_start, passage_end, 
                                                               max_dist=max_pw)
        
        pathset_and_passage_intersects[passage_id] = (raw_points, redistributed_points, endpoint_indices)

    # pprint( pathset_and_passage_intersects )
    # insert the new redistributed_points back into the pathset as new nodes
    for passage_id, each_passage_intersects in pathset_intersects.items():
        
        for each_intersect in each_passage_intersects:
            passage_num = each_intersect['passage_num']
            path_num = each_intersect['path_num']
            path_waypoint_idx = each_intersect['path_waypoint_idx']

            new_waypoint_np = pathset_and_passage_intersects[passage_id][1][path_num]
            new_waypoint_np = np.array([new_waypoint_np[0], new_waypoint_np[1], z_const])
            pathset_list[path_num] = np.insert(pathset_list[path_num], path_waypoint_idx+1, new_waypoint_np, axis=0)

    # for node in path_insert_idx:
    #     passage_num = node[0]
    #     path_num = node[1]
    #     path_indice = node[2]
    #     print(passage_num, path_num, path_indice)
        
    #     new_waypoint_np = pathset_and_passage_intersects[f'passage_{passage_num}'][path_num][1]
    #     new_waypoint_np = np.array([new_waypoint_np[0], new_waypoint_np[1], z_const])
        
    #     print(new_waypoint_np)
    #     pathset_list[path_num] = np.insert(pathset_list[path_num], 
    #                                        path_indice, 
    #                                        new_waypoint_np, axis=0)
    
    return pathset_intersects, pathset_and_passage_intersects, pathset_list
        
