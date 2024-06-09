import jax
import jax.numpy as jnp
from daxbench.core.envs import ShapeRopeEnv
from daxbench.core.envs.shape_rope_env import DefaultConf
import numpy as np
import matplotlib.pyplot as plt
from daxbench.core.utils.util import get_expert_start_end_mpm
from daxbench.core.engine.mpm_simulator import MPMState
import time

import zarr
import os
import pathlib
import glob
import numpy as np
from typing import List, Dict


class ZarrLogger:
    '''
        Zarr Logger: log large numeric data with numpy as backend \\
        Put the data of interest under ``data`` group and meta info under ``meta`` group.
    '''
    def __init__(self, 
                 path_to_save: str, 
                 data_ks: List[str], 
                 meta_ks: List[str],
                 chunk_size: int = 1000,
                 dtype: str = 'f4'):
        
        self._path = path_to_save
        self._data_ks = data_ks
        self._meta_ks = meta_ks
        
        self._chunk_size = chunk_size
        self._dtype = dtype

        self._data = dict()
        self._meta = dict()

        for key in self._data_ks:
            self._data[key] = list()

        for key in self._meta_ks:
            self._meta[key] = list()


    def log_data(self, k: str, v: np.ndarray):
        assert k in self._data_ks, 'key, {}, is not in the data key list [{}]'.format(k, self._data_ks)
        self._data[k].append(v)


    def log_meta(self, k: str, v: np.ndarray):
        assert k in self._meta_ks, 'key, {}, is not in the meta key list [{}]'.format(k, self._meta_ks)
        self._meta[k].append(v)
        

    def save_data(self):
        self._root = zarr.open(self._path, 'w')
        self._data_ptr = self._root.create_group('data', overwrite=True)
        self._meta_ptr = self._root.create_group('meta', overwrite=True)

        for key in self._data_ks:
            data = np.array( self._data[key] )
            data_shape = data.shape
            chunk_shape = (self._chunk_size, ) + (None,) * (len(data_shape) -  1)
            data_zarr = self._data_ptr.create_dataset(key,
                                                      shape=data_shape,
                                                      dtype=self._dtype,
                                                      chunks=chunk_shape, 
                                                      overwrite=True)
            data_zarr[:] = data

        
        for key in self._meta_ks:
            meta = np.array( self._meta[key] )
            meta_shape = meta.shape
            chunk_shape = (self._chunk_size, ) + (None,) * (len(meta_shape) -  1)
            meta_zarr = self._meta_ptr.create_dataset(key,
                                                      shape=meta_shape,
                                                      dtype=self._dtype,
                                                      chunks=chunk_shape, 
                                                      overwrite=True)
            meta_zarr[:] = meta


def visualize_shape(dlo: np.ndarray, ax, clr=None):
    '''
        visualize a rope shape
    '''
    if clr is None:
        clr = 0.5 + 0.5 * np.random.random(3)

    num_kp = dlo.shape[0]

    for i in range(num_kp):
        ax.scatter(dlo[i][0], dlo[i][1], dlo[i][2], color=clr,  marker='o', s=45)
    for i in range(num_kp-1):
        ax.plot3D([dlo[i][0], dlo[i+1][0]], 
                  [dlo[i][1], dlo[i+1][1]], 
                  [dlo[i][2], dlo[i+1][2]], color=clr, linewidth=3.0)
    ax.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')


def demo():
    print(jax.devices())
    env_cfg = DefaultConf()
    env_cfg.rope_hardness = 2.5
    env_cfg.rope_width = [0.25, 0.006, 0.006]
    env = ShapeRopeEnv(batch_size=1, seed=1, conf=env_cfg)
    # env.collect_goal()
    # env.collect_expert_demo(10)
    # actions = jnp.zeros((env.batch_size, 6))
    # obs, state = env.reset(env.simulator.key)
    print("time start")
    start_time = time.time()
    for it in range(100):
        obs, state = env.reset(env.simulator.key)
        for i in range(6):
            actions = get_expert_start_end_mpm(state.x, size=512)
            print(actions)
            obs, reward, done, info = env.step_with_render(actions, state)
            state = info["state"]
            print("it", it, "step", i, time.time() - start_time)
            
        print(time.time() - start_time)


def test():
    # Create the environments
    env_cfg = DefaultConf()
    env_cfg.rope_hardness = 0.5
    env_cfg.rope_width = [0.5, 0.006, 0.006]
    env_cfg.dt = 1e-4
    env = ShapeRopeEnv(batch_size=2, seed=10, conf=env_cfg)
    obs, state = env.reset(env.simulator.key)

    # Actions to be simulated in each environment
    actions = jnp.array(
        [
            [0.4, 0, 0.4, 0.6, 0, 0.6],
            [0.2, 0, 0.5, 0.4, 0, 0.4]
        ]
    )

    obs, reward, done, info = env.step_diff(actions, state)
    next_state = info["state"]
    image = env.render(next_state, visualize=True)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal')
    ax.scatter(0.0, 0.0, 0.0, color='r',  marker='*', s=100)
    next_state = np.array(next_state[0])
    print(next_state[0, :, :].shape)
    visualize_shape(next_state[0, 0:582:30, :], ax, clr='r')
    visualize_shape(next_state[1, 0:582:30, :], ax, clr='r')
    plt.show()


# @jax.jit
def random_policy(n_actions, state:MPMState, radius=0.2):
    pc = np.array(state.x[0])
    n_particles = pc.shape[0]
    p_ids = np.random.randint(n_particles // 5, (n_particles // 5) * 4, n_actions)
    end_list = pc[p_ids]
    angles = np.random.uniform(0, 2*np.pi, (n_actions,))

    for idx in range(n_actions):
        if abs(angles[idx] - 0.0) <= 0.4 or abs(angles[idx] - np.pi) <= 0.4 or abs(angles[idx] - 2 * np.pi) <= 0.4:
            print("reset action direction")
            if np.random.randint(0, 6) <= 2:
                angles[idx] = np.pi / 2
            else:
                angles[idx] = 3 * np.pi / 2
            print(angles)

    end_list[:, 0] += np.cos(angles) * radius
    end_list[:, 2] += np.sin(angles) * radius
    start_list = pc[p_ids]
    start_list[:, 0] -= np.cos(angles) * (radius / 2.)
    start_list[:, 2] -= np.sin(angles) * (radius / 2.)

    act_list = []
    for i in range(n_actions):
        start_pos = start_list[i]
        end_pos = end_list[i]
        act_list.append([*start_pos, *end_pos])

    act_list = np.array(act_list)
    return act_list


def test1():
    print(jax.devices())
    num_feat = 20
    # create the environments
    batch_size = 1
    dlo_len = 0.21
    env_cfg = DefaultConf()
    env_cfg.rope_hardness = 0.1
    env_cfg.rope_width = [dlo_len, 0.005, 0.005]
    env_cfg.dt = 1e-4
    env_cfg.rope_z_rotation_angle = 0 * np.pi / 4

    seed = 1000
    np.random.seed(seed)
    env = ShapeRopeEnv(batch_size=batch_size, seed=seed, conf=env_cfg)

    data_dir = '/home/yxtang/CodeBase/DOBERT/datasets/dax/test'
    tmp = str(dlo_len)[0] + str(dlo_len)[2] + str(dlo_len)[3]
    path_to_save = os.path.join(data_dir, f'{tmp}_dax_dlo_{seed}_test.zarr')
    
    logger = ZarrLogger(path_to_save, data_ks=['keypoints',], meta_ks=['dlo_len',])

    obs, state = env.reset(0)
    next_state = state[0]

    next_state = np.array(next_state)[0]
    num_particle = next_state.shape[0]

    stride = int( np.ceil(num_particle // num_feat / 10) * 10 )
    print(num_particle, num_feat, stride)
    N = 200
    dlo_shape_list = []

    for it in range(N):
        print(f'====== {it} =========')
        acts = random_policy(batch_size, state=state, radius=0.4)
        acts[0, 1] = 0
        acts[0, 4] = 0

        obs, reward, done, info = env.step_with_render(acts, state)
        # obs, reward, done, info = env.step_diff(acts, state)
        next_state = info["state"]
        # image = env.render(next_state, visualize=True)

        # [batch_size(1), num_paticle, 3]
        shape = np.array(next_state[0])[0, 0:num_particle:stride, :]
        print(shape.shape)
        shape_tmp = np.copy( shape )
        shape[:, 1] = shape_tmp[:, 2]
        shape[:, 2] = shape_tmp[:, 1]
        
        logger.log_meta('dlo_len', dlo_len)
        logger.log_data('keypoints', shape)
        dlo_shape_list.append(shape)

        # state = info["state"]
        
    logger.save_data()

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal')
    ax.scatter(0.0, 0.0, 0.0, color='r',  marker='*', s=100)
    for idx in range(5):
        visualize_shape(dlo_shape_list[idx], ax, clr='r')
    plt.show()


if __name__ == '__main__':
    # demo()
    test1()
