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
            trajectory = vec_smooth_traj_fn(sigmas, waypoints, 120)
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-.', label="Smooth Path", linewidth=1.0)
            # plt.plot(waypoints[:, 0], waypoints[:, 1], 'k-.', label="Raw Path")
            # plt.scatter(waypoints[:, 0], waypoints[:, 1], color='k', label="Waypoints")
        # plt.axis('equal')


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
    '''
        redistribute the intersects on the passage 
        (the order of new points must follow the pathset order!!!)
    '''
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

    # set the correct order
    dists_from_original_start = [np.linalg.norm(p - original_start) for p in points]
    
    idx_by_dist = np.argsort(dists_from_original_start)
   
    result = np.zeros_like(points)
    for m in range(len(points)):
        result[m] = redistributed_points[idx_by_dist[m]]
    return result

def redistribute_points_by_scale(points, 
                                 original_start, original_end,
                                 new_start, new_end, max_dist):
    '''
        redistribute the inpoints
    '''
    num_path = len(points)
    original_distance = np.linalg.norm( original_end - original_start )
    new_distance = min(np.linalg.norm( new_start - new_end )-0.02, max_dist)
    ratio = new_distance / original_distance

    points = points - (original_start + original_end) / 2 + (new_start + new_end) / 2
    middle_point = (new_start + new_end) / 2

    result = np.zeros_like(points)
    for i in range( num_path ):
        directional_vec = (points[i] - middle_point) / np.linalg.norm(points[i] - middle_point)
        result[i] = middle_point + ratio * directional_vec * np.linalg.norm(points[i] - middle_point)
    return result


def deform_pathset_step1(pivot_path, pathset_list: List[np.ndarray], world_map: WorldMap, max_pw=None):
    '''
        deform the pathset to obtain feasibility
    '''
    backup_pathset_list = copy.deepcopy(pathset_list)
    pathset_list = copy.deepcopy(pathset_list)
    num_waypoints = pivot_path.shape[0]
    num_path = len(pathset_list)
    z_const = pathset_list[0][0, 2]

    # ============= step1: get the passages passed by the pivot path ============ #
    central_intersections = world_map.get_path_intersection(pivot_path)
    passages = []
    for intersects in central_intersections:
        passages.append( intersects['passage'] )
    num_passage = len(passages)

    # ============= step2: group the intersects by the passages ================= #
    pathset_intersects = dict()
    
    for passage_num in range(num_passage):
        passage = passages[passage_num]
        direction_p = np.array([passage.vrtx2[0] - passage.vrtx1[0], passage.vrtx2[1] - passage.vrtx1[1]])
        direction_p = direction_p / np.linalg.norm(direction_p)
        direction_n = np.array([passage.vrtx1[0] - passage.vrtx2[0], passage.vrtx1[1] - passage.vrtx2[1]])
        direction_n = direction_n / np.linalg.norm(direction_n)

        extended_vrtx1 = np.array([passage.vrtx1[0], passage.vrtx1[1]]) + 0.0 * direction_n
        extended_vrtx2 = np.array([passage.vrtx2[0], passage.vrtx2[1]]) + 0.0 * direction_p

        extended_passage_1 = Point(extended_vrtx1[0], extended_vrtx1[1])
        extended_passage_2 = Point(extended_vrtx2[0], extended_vrtx2[1])

        intersects_on_this_passage = []
        for path_num in range(num_path):
            single_path = pathset_list[path_num]
            distances = np.linalg.norm( jnp.diff( single_path, axis=0 ), axis=1, ord=2)
            cumulative_distances = np.concatenate([jnp.array([0.0]), jnp.cumsum(distances)])
            total_length = cumulative_distances[-1]
            for i in range(num_waypoints-1):
                path_1 = Point(single_path[i, 0], single_path[i, 1])
                path_2 = Point(single_path[i+1, 0], single_path[i+1, 1])
                point = get_intersection_point(path_1, path_2, extended_passage_1, extended_passage_2)
                if point:
                    # compute the path factor at this intersect
                    sigma = (np.linalg.norm( point.to_numpy() - path_1.to_numpy() ) + cumulative_distances[i]) / total_length
                    intersects_on_this_passage.append({'passage': passage, 
                                                       'intersect_point': point, 
                                                       'passage_num': passage_num,
                                                       'path_num': path_num, 
                                                       'path_waypoint_idx': i,
                                                       'path_factor': sigma})
                    # print(passage_num, path_num, i, sigma)
                    break
        pathset_intersects[f'passage_{passage_num}'] = intersects_on_this_passage

    # ========== step3: redistribute intersects along the passage =================
    pathset_and_passage_intersects = dict()
    
    for passage_id, each_passage_intersects in pathset_intersects.items():
        raw_points = []
        Sigma = []
        for each_intersect in each_passage_intersects:
            passage_vrtx1 = np.array(each_intersect['passage'].vrtx1)[0:2]
            passage_vrtx2 = np.array(each_intersect['passage'].vrtx2)[0:2]
            point = each_intersect['intersect_point']
            raw_points.append(point)
            Sigma.append(each_intersect['path_factor'])
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

        passage_direction = (passage_end - passage_start) / np.linalg.norm(passage_end - passage_start)
        endpoints_vec = endpoint2 - endpoint1
        sign_check = np.dot(endpoints_vec, passage_direction)
        
        if sign_check < 0:
            original_start = endpoint2
            original_end = endpoint1
        else:
            original_start = endpoint1
            original_end = endpoint2

        redistributed_points = redistribute_points_by_scale(raw_points, 
                                                            original_start, original_end, 
                                                            passage_start, passage_end, max_dist=max_pw)
        pathset_and_passage_intersects[passage_id] = (raw_points, 
                                                      redistributed_points, 
                                                      endpoint_indices,
                                                      Sigma)

    # ========== step4: insert the new redistributed_points back into the pathset as new waypoints ========
    insertion_counter = {}
    for passage_id, each_passage_intersects in pathset_intersects.items():
        for each_intersect in each_passage_intersects:
            passage_num = each_intersect['passage_num']
            path_num = each_intersect['path_num']
            path_waypoint_idx = each_intersect['path_waypoint_idx']
            
            # default to 0 if this is the first time we're inserting for this path
            if path_num not in insertion_counter:
                insertion_counter[path_num] = 0

            new_waypoint_np = pathset_and_passage_intersects[passage_id][1][path_num]
            new_waypoint_np = np.array([new_waypoint_np[0], new_waypoint_np[1], z_const])
            pathset_list[path_num] = np.insert(pathset_list[path_num], 
                                               path_waypoint_idx+1+insertion_counter[path_num], 
                                               new_waypoint_np, axis=0)

            # increment counter for this path
            insertion_counter[path_num] += 1

    insertion_counter = {}
    for passage_id, each_passage_intersects in pathset_intersects.items():
        for each_intersect in each_passage_intersects:
            passage_num = each_intersect['passage_num']
            path_num = each_intersect['path_num']
            path_waypoint_idx = each_intersect['path_waypoint_idx']
            
            # default to 0 if this is the first time we're inserting for this path
            if path_num not in insertion_counter:
                insertion_counter[path_num] = 0

            new_waypoint_np = pathset_and_passage_intersects[passage_id][0][path_num]
            new_waypoint_np = np.array([new_waypoint_np[0], new_waypoint_np[1], z_const])
            backup_pathset_list[path_num] = np.insert(backup_pathset_list[path_num], 
                                                      path_waypoint_idx+1+insertion_counter[path_num], 
                                                      new_waypoint_np, axis=0)

            # increment counter for this path
            insertion_counter[path_num] += 1
    return pathset_intersects, pathset_and_passage_intersects, pathset_list, backup_pathset_list


def deform_pathset_step2(backup_pathset, new_pathset):
    num_path, backup_num_waypoint, _ = backup_pathset.shape
    num_path, new_num_waypoint, _ = new_pathset.shape

    polished_pathset = np.copy(new_pathset)
    SegIdx = []
    for k in range(num_path):
        segment_idx = [(0, 0, np.zeros(3))]
        single_path = backup_pathset[k]
        distances = np.linalg.norm( np.diff( single_path, axis=0 ), axis=1, ord=2)
        cumulative_distances = np.concatenate([np.array([0.0]), np.cumsum(distances)])
        total_length = cumulative_distances[-1]

        for i in range(backup_num_waypoint):
            if np.linalg.norm(new_pathset[k, i] - backup_pathset[k, i]) != 0:
                sigma_ki = cumulative_distances[i] / total_length
                delta_ki = new_pathset[k, i] - backup_pathset[k, i]
                segment_idx.append((i, sigma_ki, delta_ki))

                for j in range(segment_idx[-2][0], segment_idx[-1][0]):
                    ratio = (cumulative_distances[j] / total_length - segment_idx[-2][1]) / (segment_idx[-1][1] - segment_idx[-2][1])
                    polished_pathset[k, j] = backup_pathset[k, j] + ratio * (segment_idx[-1][2] - segment_idx[-2][2]) + segment_idx[-2][2]
        
        segment_idx.append((i, 1.0, np.zeros(3)))
        for j in range(segment_idx[-2][0], segment_idx[-1][0]+1):
            ratio = (cumulative_distances[j] / total_length - segment_idx[-2][1]) / (segment_idx[-1][1] - segment_idx[-2][1])
            polished_pathset[k, j] = backup_pathset[k, j] + ratio * (segment_idx[-1][2] - segment_idx[-2][2]) + segment_idx[-2][2]
        SegIdx.append(segment_idx)
    return polished_pathset, SegIdx

def deform_path_step3():
    pass






    
