'''
    dataset for global deformation model learning
'''
import os
import pathlib
import torch
import random
import numpy as np
from torch import Tensor

import torch.utils
import torch.utils.data
from torch.utils.data import Dataset
from einops import reduce
import copy

import zarr


def normalize_data(stats, delta_shape=None, delta_ee_pos=None):
    '''
        normalize the input/output
    '''
    if delta_shape is not None:
        normalized_delta_shape = (delta_shape - stats['delta_shape']['min']) / (stats['delta_shape']['max'] - stats['delta_shape']['min'] + 1e-6)
        normalized_delta_shape = normalized_delta_shape * 2 - 1
    else:
        normalized_delta_shape = None

    if delta_ee_pos is not None:
        normalized_delta_ee_pos = (delta_ee_pos - stats['delta_ee_pos']['min']) / (stats['delta_ee_pos']['max'] - stats['delta_ee_pos']['min'] + 1e-6)
        normalized_delta_ee_pos = normalized_delta_ee_pos * 2 - 1
    else:
        normalized_delta_ee_pos = None

    return normalized_delta_shape, normalized_delta_ee_pos


def unnormalize_data(stats, normalized_delta_shape=None, normalized_delta_ee_pos=None):
    if normalized_delta_shape is not None:
        normalized_delta_shape = (normalized_delta_shape + 1) / 2
        unnormalized_delta_shape = normalized_delta_shape * (stats['delta_shape']['max'] - stats['delta_shape']['min']) + stats['delta_shape']['min']
    else:
        unnormalized_delta_shape = None

    if normalized_delta_ee_pos is not None:
        normalized_delta_ee_pos = (normalized_delta_ee_pos + 1) / 2
        unnormalized_delta_ee_pos = normalized_delta_ee_pos * (stats['delta_ee_pos']['max'] - stats['delta_ee_pos']['min']) + stats['delta_ee_pos']['min']
    else:
        unnormalized_delta_ee_pos = None
        
    return unnormalized_delta_shape, unnormalized_delta_ee_pos


class GDMDataset(Dataset):
    def __init__(self, data_path:str, stats=None, max_step=5):
        super(GDMDataset, self).__init__()
        self.data_path = data_path

        root = zarr.open(self.data_path, 'r')

        self.actions = root['data']['action']
        self.states = root['data']['state']
        self.next_states = root['data']['next_state']
        self.dlo_lens = root['meta']['dlo_len']

        self.num_grasps = self.actions.shape[1] // 3
        self.num_feats = self.states.shape[1] // 3 - self.num_grasps

        self.max_step = max_step
        self.capacity = self.dlo_lens.shape[0] - self.max_step
        self.steps = np.random.randint(-self.max_step, self.max_step, size=[self.capacity,])

        if stats is None:
            self.stats = self._normalize_dataset()
        else:
            self.stats = stats

    def _normalize_dataset(self):
        stats = {}
        delta_shapes = self.next_states[:, 0:self.num_feats*3] - self.states[:, 0:self.num_feats*3]
        delta_ee_poses = self.next_states[:, self.num_feats*3:] - self.states[:, self.num_feats*3:]

        stats['delta_shape'] = {'min': self.max_step * reduce(delta_shapes, 'b d -> d', reduction='min'),
                                'max': self.max_step * reduce(delta_shapes, 'b d -> d', reduction='max')}
        stats['delta_ee_pos'] = {'min': self.max_step * reduce(delta_ee_poses, 'b d -> d', reduction='min'),
                                 'max': self.max_step * reduce(delta_ee_poses, 'b d -> d', reduction='max')}
        return stats

    def get_stats(self):
        return self.stats
    
    def update_stats(self, new_stats):
        '''
            update the dataset status
        '''
        self.stats = copy.deepcopy(new_stats)
        return None
    
    def normalize_data(self, delta_shape, delta_ee_pos):
        normalized_delta_shape = (delta_shape - self.stats['delta_shape']['min']) / (self.stats['delta_shape']['max'] - self.stats['delta_shape']['min'] + 1e-6)
        normalized_delta_shape = normalized_delta_shape * 2 - 1
        
        normalized_delta_ee_pos = (delta_ee_pos - self.stats['delta_ee_pos']['min']) / (self.stats['delta_ee_pos']['max'] - self.stats['delta_ee_pos']['min'] + 1e-6)
        normalized_delta_ee_pos = normalized_delta_ee_pos * 2 - 1

        return normalized_delta_shape, normalized_delta_ee_pos
    
    def unnormalize_data(self, normalized_delta_shape, normalized_delta_ee_pos=None):
        normalized_delta_shape = (normalized_delta_shape + 1) / 2
        unnormalized_delta_shape = normalized_delta_shape * (self.stats['delta_shape']['max'] - self.stats['delta_shape']['min']) + self.stats['delta_shape']['min']
        
        if normalized_delta_ee_pos is not None:
            normalized_delta_ee_pos = (normalized_delta_ee_pos + 1) / 2
            unnormalized_delta_ee_pos = normalized_delta_ee_pos * (self.stats['delta_ee_pos']['max'] - self.stats['delta_ee_pos']['min']) + self.stats['delta_ee_pos']['min']
        else:
            unnormalized_delta_ee_pos = None
        return unnormalized_delta_shape, unnormalized_delta_ee_pos

    def __len__(self):
        return self.capacity

    def __getitem__(self, idx):
        state = self.states[idx]
        dlo_len = self.dlo_lens[idx]
        
        # target
        # self.delta_shapes = self.next_states[:, 0:self.num_feats*3] - self.states[:, 0:self.num_feats*3]

        # inputs
        # self.delta_ee_poses = self.next_states[:, self.num_feats*3:] - self.states[:, self.num_feats*3:]
        step = max(self.steps[idx], 0)
        delta_ee_pos = self.next_states[idx+step, self.num_feats*3:] - self.states[idx, self.num_feats*3:]
        delta_shape = self.next_states[idx+step, 0:self.num_feats*3] - self.states[idx, 0:self.num_feats*3]

        normalized_delta_shape, normalized_delta_ee_pos = self.normalize_data(delta_shape, delta_ee_pos)

        output = {
            'state': state,
            'delta_ee_pos': normalized_delta_ee_pos,
            'delta_shape': normalized_delta_shape,
            'dlo_len': dlo_len
        }
        return output 


class MultiStepGDMDataset(Dataset):
    def __init__(self, data_path:str, max_step:int=5):
        super(MultiStepGDMDataset, self).__init__()
        self.data_path = data_path

        root = zarr.open(self.data_path, 'r')

        self.actions = root['data']['action']
        self.states = root['data']['state']
        self.next_states = root['data']['next_state']
        self.dlo_lens = root['meta']['dlo_len']

        self.num_grasps = self.actions.shape[1] // 3
        self.num_feats = self.states.shape[1] // 3 - self.num_grasps
        
        self.max_step = max_step
        self.capacity = self.dlo_lens.shape[0] - max_step

        if max_step > 0:
            self.steps = np.random.randint(-self.max_step, self.max_step, size=[self.capacity,])
        else:
            self.steps = [0,] * self.capacity

    def __len__(self):
        return self.capacity

    def __getitem__(self, idx):
        step = max(self.steps[idx], 0)
        
        state = self.states[idx]
        dlo_len = self.dlo_lens[idx]
        
        delta_shape = self.next_states[idx+step, 0:self.num_feats*3] - self.states[idx, 0:self.num_feats*3]
        delta_ee_pos = self.next_states[idx+step, self.num_feats*3:] - self.states[idx, self.num_feats*3:]

        output = {
            'state': state,
            'delta_ee_pos': delta_ee_pos,
            'delta_shape': delta_shape,
            'dlo_len': dlo_len
        }
        return output 


if __name__ == '__main__':
    data_path = '/home/yxtang/CodeBase/DOBERT/datasets/gdm_mj/train/gdm_05_mujoco_rope_train.zarr'
    dataset = GDMDataset( data_path )
    for key, val in dataset[0].items():
        print(key)
        print(val.shape)

    print(dataset.num_feats, dataset.num_grasps)
    print(dataset.capacity)
    
    for key, val in dataset.get_stats().items():
        print(key)
        print(val)

    data_path = '/home/yxtang/CodeBase/DOBERT/datasets/gdm_mj/train/gdm_05_mujoco_rope_train.zarr'
    dataset = MultiStepGDMDataset( data_path )
    for key, val in dataset[0].items():
        print(key)
        print(val.shape)