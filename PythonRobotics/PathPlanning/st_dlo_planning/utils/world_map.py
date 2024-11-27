import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import fcl
from itertools import combinations
from typing import NamedTuple
from dataclasses import dataclass
from matplotlib.patches import Rectangle 
from scipy.spatial.transform import Rotation as R


class Point(NamedTuple):
    '''
        2D point
    '''
    x: float
    y: float

    def to_list(self):
        return [self.x, self.y]
    

class Passage(NamedTuple):
    '''
        Passage in Passage Map
    '''
    vrtx1: list
    vrtx2: list
    min_dist: float
    

def line(p1: Point, p2: Point):
    A = (p1.y - p2.y)
    B = (p2.x - p1.x)
    C = (p1.x * p2.y - p2.x * p1.y)
    return A, B, -C


def get_intersection_point(p1: Point, q1: Point, p2: Point, q2: Point):
    x1,y1 = p1.x, p1.y
    x2,y2 = q1.x, q1.y
    x3,y3 = p2.x, p2.y
    x4,y4 = q2.x, q2.y
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)

    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return Point(x, y)
    

def get_orientation(p: Point, q: Point, r: Point): 
    '''
        get the orientation of an ordered triplet (p, q, r) 
        Return: 
            0 : Collinear points 
            1 : Clockwise points 
            2 : Counterclockwise points
        See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ for details.  
    '''
    val = ((r.x - q.x) * (q.y - p.y)) - ((r.y - q.y) * (q.x - p.x)) 
    if (val > 0.): 
        # Clockwise orientation 
        return 1
    elif (val < 0.): 
        # Counterclockwise orientation 
        return 2
    else: 
        # Collinear orientation 
        return 0
    
 
def check_on_segment(p: Point, q: Point, r: Point): 
    '''
        Given three collinear points (p, q, r), 
        check if q lies on line segment <p, r> 
    '''
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and \
         (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y)) ): 
        return True
    return False

 
def check_intersection(p1: Point, q1: Point, p2: Point, q2: Point): 
    '''
        returns true if the line segment <p1, q1> and <p2, q2> intersect.
        Find the four orientations required for the general and special cases 
    '''
    o1 = get_orientation(p1, q1, p2) 
    o2 = get_orientation(p1, q1, q2) 
    o3 = get_orientation(p2, q2, p1) 
    o4 = get_orientation(p2, q2, q1) 
  
    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return True
  
    # Special Cases 
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1 
    if ((o1 == 0) and check_on_segment(p1, p2, q1)): 
        return True
  
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1 
    if ((o2 == 0) and check_on_segment(p1, q2, q1)): 
        return True
  
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2 
    if ((o3 == 0) and check_on_segment(p2, p1, q2)): 
        return True
  
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2 
    if ((o4 == 0) and check_on_segment(p2, q1, q2)): 
        return True
  
    # if none of the cases 
    return False


def plot_rectangle(size_x, size_y, pos_x, pos_y, ax, angle:float=0., center=True, color='b'):
    if center:
        angle = angle * 180 / np.pi
        ax.add_patch(Rectangle([pos_x, pos_y], size_x / 2., size_y / 2., angle=angle, rotation_point='xy', color=color))
        ax.add_patch(Rectangle((pos_x, pos_y), -size_x / 2., size_y / 2., angle=angle, rotation_point='xy', color=color))
        ax.add_patch(Rectangle((pos_x, pos_y), -size_x / 2., -size_y / 2., angle=angle, rotation_point='xy', color=color))
        ax.add_patch(Rectangle((pos_x, pos_y), size_x / 2., -size_y / 2., angle=angle, rotation_point='xy', color=color))
    else:
        pass


def plot_circle(x, y, radius, ax, color="-b"):
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + radius * jnp.cos(jnp.deg2rad(d)) for d in deg]
    yl = [y + radius * jnp.sin(jnp.deg2rad(d)) for d in deg]
    ax.plot(xl, yl, color)


class Block:
    '''
        city building block
    '''
    def __init__(self, size_x, size_y, size_z, 
                 pos_x, pos_y, pos_z = None, angle: float = 0.,
                 clr='gray', is_wall: bool = False):
        
        self._size_x = size_x
        self._size_y = size_y
        self._size_z = size_z

        self._pos_x = pos_x
        self._pos_y = pos_y
        self._pos_z = pos_z if pos_z is not None else size_z / 2.0

        self.angle = angle
        self.rotation = R.from_matrix([[np.cos(angle), -np.sin(angle), 0],
                                       [np.sin(angle), np.cos(angle), 0],
                                       [0, 0, 1.0]]).as_matrix()

        # collision property
        geom = fcl.Box(self._size_x, self._size_y, self._size_z)
        T = np.array([self._pos_x, self._pos_y, self._pos_z])
        tf = fcl.Transform(self.rotation, T)
        # tf = fcl.Transform(T)
        self._collision_obj = fcl.CollisionObject(geom, tf)

        # visualization property
        self._color = clr

        self._wall = is_wall


@dataclass
class MapCfg:
    resolution: float = 0.1
    map_xmin: float = 0.
    map_xmax: float = 200.
    map_ymin: float = 0.
    map_ymax: float = 200.
    
    map_zmin: float = 99.9
    map_zmax: float = 100.1
    
    robot_size: float = 0.1
    
    dim: int = 3


class WorldMap:
    '''
        the world map of tasks
    '''
    def __init__(self, map_cfg: MapCfg):

        self._obstacle = []
        self.map_cfg = map_cfg
        self._resolution = map_cfg.resolution
        self._s = fcl.Sphere(map_cfg.robot_size)
        
        self._finalized = False

        # add walls
        wall_left = Block(map_cfg.robot_size, map_cfg.map_ymax, map_cfg.map_zmax, 
                          -map_cfg.robot_size/2, map_cfg.map_ymax/2, is_wall=True)
        
        wall_right = Block(map_cfg.robot_size, map_cfg.map_ymax, map_cfg.map_zmax, 
                           map_cfg.map_ymax+map_cfg.robot_size/2, map_cfg.map_ymax/2, is_wall=True)
        
        wall_down = Block(map_cfg.map_xmax, map_cfg.robot_size, map_cfg.map_zmax, 
                          map_cfg.map_xmax/2, -map_cfg.robot_size/2, is_wall=True)
        
        wall_up = Block(map_cfg.map_xmax, map_cfg.robot_size, map_cfg.map_zmax, 
                        map_cfg.map_xmax/2, map_cfg.map_ymax+map_cfg.robot_size/2, is_wall=True)
        
        self.add_obstacle(wall_left)
        self.add_obstacle(wall_right)
        self.add_obstacle(wall_down)
        self.add_obstacle(wall_up)

        # passages
        self._passages = []

        
    def add_obstacle(self, obstacle: Block):
        self._obstacle.append(obstacle)

    def finalize(self):
        self._collision_instances = [obstacle._collision_obj for obstacle in self._obstacle]

        self._collision_manager = fcl.DynamicAABBTreeCollisionManager()
        self._collision_manager.registerObjects(self._collision_instances)
        self._collision_manager.setup()

        self._finalized = True

        self.construct_passage()
        self.filter_passage()
        
        return None
    

    def construct_passage(self):
        assert self._finalized, 'world_map is not finalized!'
        for obs_pair in combinations(self._obstacle, 2):
            if not (obs_pair[0]._wall and obs_pair[1]._wall):
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
        assert self._finalized, 'world_map is not finalized!'
        self._filtered_passages = []
        for passage in self._passages:
            passage_center = [(passage.vrtx1[0] + passage.vrtx2[0]) / 2.,
                              (passage.vrtx1[1] + passage.vrtx2[1]) / 2.]
            
            passage_half_length = passage.min_dist / 2. - 0.0001
            cylinder_g = fcl.Cylinder(passage_half_length, 0.001)
            cylinder_t = fcl.Transform(np.array([passage_center[0], passage_center[1], 0.0]))
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
    

    def compute_clearance(self, state):
        '''
            compute the minimal clearance from obstacles
        '''
        assert self._finalized, 'world_map is not finalized!'
        T = np.array([state[0], state[1], state[2]])
        tf = fcl.Transform(T)
        robot = fcl.CollisionObject(self._s, tf)

        clearance = []
        for i in range(len(self._collision_instances)):
            request = fcl.DistanceRequest()
            result = fcl.DistanceResult()
            obs = self._collision_instances[i]
            ret = fcl.distance(robot, obs, request, result)
            clearance.append(ret)
        cel = np.min(clearance)
        return cel
    

    def sample_validate_position(self):
        '''
            sample a validate position
        '''
        while True:
            print('*****')
            position_candidate = np.random.uniform([self.map_cfg.map_xmin, self.map_cfg.map_ymin, self.map_cfg.map_zmin],
                                                [self.map_cfg.map_xmin, self.map_cfg.map_ymin, self.map_cfg.map_zmax],
                                                size=(3,))
            if self.check_pos_collision(position_candidate):
                break
        return position_candidate


    def check_passage_intersection(self, parent_state, child_state):
        '''
            check whether the <parent_state, child_state> pair intersects with some passage in the map
        '''
        for passage in self._filtered_passages:
            intersection = check_intersection(
                Point(passage.vrtx1[0], passage.vrtx1[1]),
                Point(passage.vrtx2[0], passage.vrtx2[1]),
                Point(parent_state[0], parent_state[1]),
                Point(child_state[0], child_state[1])
            )
            if intersection:
                return passage
        return None
    

    def get_path_intersection(self, sinle_path, use_extension: bool = False):
        '''
            get the intersection between the path and passage
        '''
        intersects = []
        path_len = sinle_path.shape[0]
        for i in range(path_len-1):
            for passage in self._filtered_passages:
                path_1 = Point(sinle_path[i, 0], sinle_path[i, 1])
                path_2 = Point(sinle_path[i+1, 0], sinle_path[i+1, 1])

                direction_p = np.array([passage.vrtx2[0] - passage.vrtx1[0], passage.vrtx2[1] - passage.vrtx1[1]])
                direction_p = direction_p / np.linalg.norm(direction_p)
                direction_n = np.array([passage.vrtx1[0] - passage.vrtx2[0], passage.vrtx1[1] - passage.vrtx2[1]])
                direction_n = direction_n / np.linalg.norm(direction_n)

                extended_vrtx1 = np.array([passage.vrtx1[0], passage.vrtx1[1]]) + 250. * direction_n
                extended_vrtx2 = np.array([passage.vrtx2[0], passage.vrtx2[1]]) + 250. * direction_p

                passage_1 = Point(passage.vrtx1[0], passage.vrtx1[1])
                passage_2 = Point(passage.vrtx2[0], passage.vrtx2[1])

                if use_extension:
                    extended_passage_1 = Point(extended_vrtx1[0], extended_vrtx1[1])
                    extended_passage_2 = Point(extended_vrtx2[0], extended_vrtx2[1])
                    point = get_intersection_point(path_1, path_2, extended_passage_1, extended_passage_2)
                else:
                    point = get_intersection_point(path_1, path_2, passage_1, passage_2)
                
                if point:
                    # if check_on_segment(path_1, point, path_2) and check_on_segment(passage_1, point, passage_2):
                    intersects.append({'passage': passage, 'passage_width': passage.min_dist, 'point': point})
        return intersects


    def check_pos_collision(self, state):
        assert self._finalized, 'world_map is not finalized!'
        T = np.array([state[0], state[1], state[2]])
        tf = fcl.Transform(T)
        robot = fcl.CollisionObject(self._s, tf)

        req = fcl.CollisionRequest()
        rdata = fcl.CollisionData(request = req)
        self._collision_manager.collide(robot, rdata, fcl.defaultCollisionCallback)
        validate = not rdata.result.is_collision
        return validate
        

    def check_line_collision(self, start_state: np.ndarray, end_state: np.ndarray) -> bool:
        assert self._finalized, 'world_map is not finalized!'
        state_distance = np.linalg.norm(start_state - end_state)
        N = int(state_distance / (self._resolution))
        ratios = np.linspace(0., 1.0, num=N)
        
        for ratio in ratios:
            state_sample = (1 - ratio) * start_state + ratio * end_state
            res = self.check_pos_collision(state_sample)
            if res:
                return True  
        return False
    
    def visualize_map(self, ax=None, show_wall: bool=False):
        from ..spatial_pathset_gen.utils import plot_box
        if ax is None:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(projection='3d')
        ax.set_xlim(self.map_cfg.map_xmin, self.map_cfg.map_xmax)
        ax.set_ylim(self.map_cfg.map_ymin, self.map_cfg.map_ymax)
        ax.set_zlim(0, self.map_cfg.map_zmax)
        ax.set_aspect('equal')
        ax.xaxis.set_ticks(np.arange(self.map_cfg.map_xmin, self.map_cfg.map_xmax, 50.0))
        ax.yaxis.set_ticks(np.arange(self.map_cfg.map_ymin, self.map_cfg.map_ymax, 50.0))
        ax.zaxis.set_ticks(np.arange(self.map_cfg.map_zmin, self.map_cfg.map_zmax, 50.0))
        ax.set(xlabel='$x (m)$', ylabel='$y (m)$', zlabel='$z (m)$')
        ax.grid(True)

        # obstacle visulization
        for obstacle in self._obstacle:
            if show_wall:
                plot_box(center = (obstacle._pos_x, obstacle._pos_y, obstacle._pos_z), 
                        size = (obstacle._size_x, obstacle._size_y, obstacle._size_z),
                        ax = ax,
                        clr = obstacle._color)
            else:
                if not obstacle._wall:
                    plot_box(center = (obstacle._pos_x, obstacle._pos_y, obstacle._pos_z), 
                            size = (obstacle._size_x, obstacle._size_y, obstacle._size_z),
                            ax = ax,
                            clr = obstacle._color)
        return ax
    

    def visualize_passage(self, ax=None, full_passage: bool = True):
        assert self._finalized, 'world_map is not finalized!'
        if ax is None:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot()

        for obs in self._obstacle:
            plot_rectangle(obs._size_x, obs._size_y, obs._pos_x, obs._pos_y, ax, angle=obs.angle, color=obs._color)
        
        if full_passage:
            for passage in self._passages:
                ax.plot([passage.vrtx1[0], passage.vrtx2[0]], [passage.vrtx1[1], passage.vrtx2[1]], 'k--')

        for passage in self._filtered_passages:
            ax.plot([passage.vrtx1[0], passage.vrtx2[0]], [passage.vrtx1[1], passage.vrtx2[1]], 'k-.', linewidth=1.0)
        return fig, ax
    

if __name__ == '__main__':
    from pprint import pprint
    cfg = MapCfg()
    print(cfg.resolution)