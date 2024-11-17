import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import fcl

from st_dlo_planning.spatial_pathset_gen.configuration_map import MapCfg


class Block:
    '''
        city building block
    '''
    def __init__(self, size_x, size_y, size_z, 
                 pos_x, pos_y, pos_z = None,
                 clr='gray', is_wall: bool = False):
        
        self._size_x = size_x
        self._size_y = size_y
        self._size_z = size_z

        self._pos_x = pos_x
        self._pos_y = pos_y
        self._pos_z = pos_z if pos_z is not None else size_z / 2.0

        # collision property
        geom = fcl.Box(self._size_x, self._size_y, self._size_z)
        T = np.array([self._pos_x, self._pos_y, self._pos_z])
        tf = fcl.Transform(T)
        self._collision_obj = fcl.CollisionObject(geom, tf)

        # visualization property
        self._color = clr

        self._wall = is_wall


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
        wall_left = Block(5., 200., 200., -2.5, 100., is_wall=True)
        wall_right = Block(5., 200., 200., 202.5, 100., is_wall=True)
        wall_down = Block(200., 5., 200., 100, -2.5, is_wall=True)
        wall_up = Block(200., 5., 200., 100, 202.5, is_wall=True)
        self.add_obstacle(wall_left)
        self.add_obstacle(wall_right)
        self.add_obstacle(wall_down)
        self.add_obstacle(wall_up)
        
    def add_obstacle(self, obstacle: Block):
        self._obstacle.append(obstacle)

    def finalize(self):
        self._collision_instances = [obstacle._collision_obj for obstacle in self._obstacle]

        self._collision_manager = fcl.DynamicAABBTreeCollisionManager()
        self._collision_manager.registerObjects(self._collision_instances)
        self._collision_manager.setup()

        self._finalized = True


    def check_pos_collision(self, state):
        assert self._finalized, 'world_map is not finalized!'
        T = np.array([state[0], state[1], state[2]])
        tf = fcl.Transform(T)
        robot = fcl.CollisionObject(self._s, tf)

        req = fcl.CollisionRequest()
        rdata = fcl.CollisionData(request = req)
        self._collision_manager.collide(robot, rdata, fcl.defaultCollisionCallback)
        return rdata.result.is_collision
        

    def check_line_collision(self, start_state: np.ndarray, end_state: np.ndarray) -> bool:
        assert self._finalized, 'city_map is not finalized!'
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
        from .utils import plot_box
        if ax is None:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(projection='3d')
        ax.set_xlim(self.map_cfg.map_xmin-5, self.map_cfg.map_xmax+5)
        ax.set_ylim(self.map_cfg.map_ymin-5, self.map_cfg.map_ymax+5)
        ax.set_zlim(0, self.map_cfg.map_zmax+5)
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