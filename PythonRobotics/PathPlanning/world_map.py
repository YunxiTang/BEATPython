import os
import jax.numpy as jnp
import numpy as np
import jax.random as random
import matplotlib.pyplot as plt

import fcl
from typing import Tuple
from abc import abstractmethod, ABCMeta


class ABCMap(metaclass=ABCMeta):
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


class Block:
    def __init__(self, size_x, size_y, size_z, 
                 pos_x, pos_y, pos_z = None,
                 clr='gray'):
        
        self._size_x = size_x
        self._size_y = size_y
        self._size_z = size_z

        self._pos_x = pos_x
        self._pos_y = pos_y
        self._pos_z = pos_z if pos_z is not None else size_z / 2.0

        # collision property
        geom = fcl.Box(self._size_x, self._size_y, self._size_z)
        T = jnp.array([self._pos_x, self._pos_y, self._pos_z])
        tf = fcl.Transform(T)
        self._collision_obj = fcl.CollisionObject(geom, tf)

        # visualization property
        self._color = clr


class CityMap(ABCMap):
    def __init__(self, start=None, goal=None, resolution: float = 0.1):

        self._obstacle = []
        
        self._start = start
        self._goal = goal
        
        self._resolution = resolution

        self._xmin = 0.
        self._xmax = 200.
        self._ymin = 0.
        self._ymax = 200.
        self._zmin = 0.
        self._zmax = 200.
        
        self._dim = 3

        self._rng_key = random.PRNGKey(seed=126)

        self._finalized = False

        # dummy quadrotor with size of radius 0.1 m
        self._s = fcl.Sphere(0.1)

        # TODO: visualization
        
        
    def update_start(self, start: jnp.ndarray):
        self._start = start
        
    def update_goal(self, goal: jnp.ndarray):
        self._goal = goal
        
    def add_obstacle(self, obstacle: Block):
        self._obstacle.append(obstacle)

    def finalize(self):
        self._collision_instances = [obstacle._collision_obj for obstacle in self._obstacle]

        self._collision_manager = fcl.DynamicAABBTreeCollisionManager()
        self._collision_manager.registerObjects(self._collision_instances)
        self._collision_manager.setup()

        self._finalized = True
        
    def sample_free_pos(self, toward_goal: bool = False) -> jnp.ndarray:
        assert self._finalized, 'city_map is not finalized!'
        if toward_goal and self._goal is not None:
            return self._goal
        else:
            while True:
                self._rng_key, rng_key_x, rng_key_y, rng_key_z = random.split(self._rng_key, 4)
                x = random.uniform(rng_key_x, shape=(1,), minval=self._xmin, maxval=self._xmax)
                y = random.uniform(rng_key_y, shape=(1,), minval=self._ymin, maxval=self._ymax)
                z = random.uniform(rng_key_z, shape=(1,), minval=self._zmin, maxval=self._zmax)
                sampled_pos = jnp.array([x[0], y[0], z[0]])
                collision = self.check_pos_collision(sampled_pos)
                if not collision:
                    return sampled_pos


    def check_pos_collision(self, state):
        assert self._finalized, 'city_map is not finalized!'
        T = jnp.array([state[0], state[1], state[2]])
        tf = fcl.Transform(T)
        robot = fcl.CollisionObject(self._s, tf)

        req = fcl.CollisionRequest()
        rdata = fcl.CollisionData(request = req)
        self._collision_manager.collide(robot, rdata, fcl.defaultCollisionCallback)
        return rdata.result.is_collision
        

    def check_line_collision(self, start_state: jnp.ndarray, end_state: jnp.ndarray) -> bool:
        assert self._finalized, 'city_map is not finalized!'
        state_distance = jnp.linalg.norm(start_state - end_state)
        N = int(state_distance / (self._resolution))
        ratios = jnp.linspace(0., 1.0, num=N)

        # robots = []
        # for ratio in ratios:
        #     state_sample = (1 - ratio) * start_state + ratio * end_state
        #     T = jnp.array([state_sample[0], state_sample[1], state_sample[2]])
        #     tf = fcl.Transform(T)
        #     robots.append( fcl.CollisionObject(self._s, tf) )
        # robots_manager = fcl.DynamicAABBTreeCollisionManager()
        # robots_manager.registerObjects(robots)
        # robots_manager.setup()

        # req = fcl.CollisionRequest()
        # rdata = fcl.CollisionData(request = req)
        # self._collision_manager.collide(robots_manager, rdata, fcl.defaultCollisionCallback)
        # return rdata.result.is_collision
        
        for ratio in ratios:
            state_sample = (1 - ratio) * start_state + ratio * end_state
            res = self.check_pos_collision(state_sample)
            if res: # collision
                return True  
        return False
    
    def visualize_map(self, ax):
        from utils import plot_box
        ax.set_xlim(self._xmin, self._xmax)
        ax.set_ylim(self._ymin, self._ymax)
        ax.set_zlim(self._zmin, self._zmax)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.grid(False)

        # start and goal visualzation
        if self._start is not None:
            ax.scatter(self._start[0], self._start[1], self._start[2], color='b', marker='o', s=125)
        if self._goal is not None:
            ax.scatter(self._goal[0], self._goal[1], self._goal[2], color='r',  marker='*', s=175)

        # obstacle visulization
        for obstacle in self._obstacle:
            plot_box(center = (obstacle._pos_x, obstacle._pos_y, obstacle._pos_z), 
                     size = (obstacle._size_x, obstacle._size_y, obstacle._size_z),
                     ax = ax,
                     clr = obstacle._color)
        return None


class TwoDimMap(ABCMap):
    def __init__(self, arena, resolution: float = 0.01) -> None:
        
        self.xmin = float(arena[0])
        self.xmax = float(arena[1])
        self.ymin = float(arena[2])
        self.ymax = float(arena[3])
        
        self._lb = jnp.array([self.xmin, self.ymin])
        self._ub = jnp.array([self.xmax, self.ymax])
        
        self._resolution = resolution
        
        self._dim = 2
        
        self._obstacle = []
        
        self._start = None
        self._goal = None
        
        self._rng_key = random.PRNGKey(seed=146)
        
    def update_start(self, start: jnp.ndarray):
        self._start = start
        
    def update_goal(self, goal: jnp.ndarray):
        self._goal = goal
        
    def add_obstacle(self, obstacle: Tuple):
        self._obstacle.append(obstacle)
        
    def sample_free_pos(self, toward_goal: bool = False) -> jnp.ndarray:
        if toward_goal and self._goal is not None:
            return self._goal
        
        else:
            while True:
                self._rng_key, rng_key = random.split(self._rng_key, 2)
                sampled_pos = random.uniform(rng_key, shape=(self._dim,), 
                                             minval=self._lb, 
                                             maxval=self._ub)
                collision = self.check_pos_collision(sampled_pos)
                if not collision:
                    return sampled_pos
        
    def check_line_collision(self, start_state: jnp.ndarray, end_state: jnp.ndarray) -> bool:
        state_distance = jnp.linalg.norm(start_state - end_state)
        N = int(state_distance / self._resolution)
        ratios = jnp.linspace(0., 1.0, num=N)
        for (ox, oy, size) in self._obstacle:
            center = jnp.array([ox, oy])
            for ratio in ratios[1:]:
                state_sample = (1 - ratio) * start_state + ratio * end_state
                dist = jnp.linalg.norm(state_sample - center) - size
                if dist <= 0.:
                    # collision
                    return True  
        return False
    
    def check_pos_collision(self, state):
        for (ox, oy, size) in self._obstacle:
            center = jnp.array([ox, oy])
            dist = jnp.linalg.norm(state - center) - size
            if dist <= 0.:
                # collision
                return True  
        return False


if __name__ == '__main__':

    import time

    city_map = CityMap(start=np.array([0., 0., 0.]),
                       goal=[200., 200., 200.])

    # add some obstacles
    obs1 = Block(60., 30., 120., 100., 140., clr=[0.4, 0.5, 0.4])
    obs2 = Block(60., 40., 180., 170., 50., clr=[0.5, 0.5, 0.6])
    obs3 = Block(40., 60., 90., 30., 70., clr=[0.3, 0.3, 0.4])
    city_map.add_obstacle(obs1)
    city_map.add_obstacle(obs2)
    city_map.add_obstacle(obs3)
    city_map.finalize()

    # visualize the world map and others
    fig, ax = city_map.visualize()
    # T = 0
    # N = 200
    # for i in range(N):
    #     sample = city_map.sample_pos()
    #     ts = time.time()
    #     collision = city_map.check_pos_collision(sample)
    #     delta_t = time.time() - ts
    #     print( i, delta_t ) 
    #     T += delta_t
    #     clr = 'k'
    #     if collision:
    #         clr = 'r'
    #         ax.scatter(sample[0], sample[1], sample[2], color=clr)
    # print(T / N)
    
    T = 0
    for i in range(200):
        sample1 = city_map.sample_pos()
        sample2 = city_map.sample_pos()
        ts = time.time()
        res = city_map.check_line_collision(sample1, sample2)
        delta_t = time.time() - ts
        print( delta_t ) 
        T += delta_t
        if res:
            ax.plot([sample1[0], sample2[0]], [sample1[1], sample2[1]], [sample1[2], sample2[2]])
    print(T / 200.)
    plt.show()
    
    