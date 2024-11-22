import jax
import jax.numpy as jnp
# import numpy 
from .path_interpolation import query_point_from_path
from functools import partial

import matplotlib.pyplot as plt


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
