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


def visualize_shape(dlo: np.ndarray, ax, ld=3.0, s=25, clr=None):
    '''
        visualize a rope shape
    '''
    if clr is None:
        clr = 0.5 + 0.5 * np.random.random(3)

    num_kp = dlo.shape[0]

    for i in range(num_kp):
        ax.scatter(dlo[i][0], dlo[i][1], dlo[i][2], color=clr,  marker='o', s=s)
    for i in range(num_kp-1):
        ax.plot3D([dlo[i][0], dlo[i+1][0]], 
                  [dlo[i][1], dlo[i+1][1]], 
                  [dlo[i][2], dlo[i+1][2]], color=clr, linewidth=ld)
    ax.axis('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')


def plotCoordinateFrame(axis, T_0f, size=1, linestyle='-', linewidth=3, name=None):
    """ draw a coordinate frame on a 3d axis. 
        In the resulting plot, ```x = red, y = green, z = blue```
    
        ```plotCoordinateFrame(axis, T_0f, size=1, linewidth=3)```

        Arguments:
        ```axis```: an axis of type matplotlib.axes.Axes3D
        ```T_0f```: The 4x4 transformation matrix that takes points from the frame of interest, to the plotting frame
        ```size```: the length of each line in the coordinate frame
        ```linewidth```: the width of each line in the coordinate frame
    """

    p_f = np.array([ [ 0,0,0,1], 
                        [size,0,0,1], 
                        [0,size,0,1], 
                        [0,0,size,1]]).T
    p_0 = np.dot(T_0f, p_f)

    X = np.append([p_0[:,0].T], [p_0[:,1].T], axis=0 )
    Y = np.append([p_0[:,0].T], [p_0[:,2].T], axis=0 )
    Z = np.append([p_0[:,0].T], [p_0[:,3].T], axis=0 )
    axis.plot3D(X[:,0],X[:,1],X[:,2], f'r{linestyle}', linewidth=linewidth)
    axis.plot3D(Y[:,0],Y[:,1],Y[:,2], f'g{linestyle}', linewidth=linewidth)
    axis.plot3D(Z[:,0],Z[:,1],Z[:,2], f'b{linestyle}', linewidth=linewidth)

    if name is not None:
        axis.text(X[0,0],X[0,1],X[0,2], name, zdir='x')


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


class MultiStepGDMDataset(Dataset):
    def __init__(self, data_path:str, max_step:int=5):
        super(MultiStepGDMDataset, self).__init__()
        self.data_path = data_path

        root = zarr.open(self.data_path, 'r')

        self.actions = root['data']['action']

        self.dlo_keypoints = root['data']['dlo_keypoints']
        self.eef_states = root['data']['eef_states']
        self.eef_transforms = root['data']['eef_transforms']

        self.next_dlo_keypoints = root['data']['next_dlo_keypoints']
        self.next_eef_states = root['data']['next_eef_states']
        self.next_eef_transforms = root['data']['next_eef_transforms']

        self.ep_num = root['data']['ep_num']

        self.dlo_lens = root['meta']['dlo_len']

        self.num_grasps = self.actions.shape[1] // 3
        self.num_feats = self.dlo_keypoints.shape[1] // 3
        
        self.max_step = max_step
        self.capacity = self.dlo_lens.shape[0] - max_step

        if max_step > 0:
            self.steps = np.random.randint(0, self.max_step, size=[self.capacity,], dtype=int)
        else:
            self.steps = [0,] * self.capacity

    def __len__(self):
        return self.capacity

    def __getitem__(self, idx):
        step = self.steps[idx]
        next_idx = idx + step
        
        while True:
            if self.ep_num[next_idx] != self.ep_num[idx]:
                next_idx = next_idx - 1
            else:
                break

        dlo_keypoints = self.dlo_keypoints[idx]
        eef_states = self.eef_states[idx]
        
        delta_shape = self.next_dlo_keypoints[next_idx] - self.dlo_keypoints[idx]
        delta_eef = self.next_eef_states[next_idx] - self.eef_states[idx]

        dlo_len = self.dlo_lens[idx]

        eef_transforms = self.eef_transforms[idx]
        next_eef_transforms = self.next_eef_transforms[next_idx]

        output = {
            'dlo_keypoints': dlo_keypoints.reshape(self.num_feats, 3),
            'eef_states': eef_states.reshape(self.num_grasps, -1),
            'delta_shape': delta_shape.reshape(self.num_feats, 3),
            'delta_eef': delta_eef.reshape(self.num_grasps, -1),
            'dlo_len': dlo_len,
            'eef_transforms': eef_transforms,
            'next_eef_transforms': next_eef_transforms,
        }
        return output 


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.transform import Rotation as sciR

    data_path = '/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/gdm_data/task_03_train.zarr'
    dataset = MultiStepGDMDataset( data_path, max_step=5 )
    for key, val in dataset[0].items():
        print(key, ': ', val.shape)

    
    for _ in range(100):
        i = random.randint(0, dataset.capacity)
        sample = dataset[i]

        dlo_keypoints = sample['dlo_keypoints']
        delta_shape = sample['delta_shape']
        delta_eef = sample['delta_eef']

        next_dlo_keypoints = dlo_keypoints + delta_shape

        print( np.linalg.norm(delta_shape), np.linalg.norm(delta_eef) )
        print( '=======================' )
        
        # fig = plt.figure(figsize=plt.figaspect(0.5))
        # ax = fig.add_subplot(projection='3d')

        # visualize_shape(dlo_keypoints.reshape(-1, 3), ax, clr='r', ld=1)
        # l_eef_pos = sample['eef_transforms'][0:3]
        # l_eef_quat = sample['eef_transforms'][3:7]

        # left_Rm = sciR.from_quat([l_eef_quat[1],l_eef_quat[2],l_eef_quat[3],l_eef_quat[0]]).as_matrix()
        # left_Rm_tmp = np.concatenate((left_Rm, np.zeros([1,3])), axis=0)
        # left_Pm = np.array(list(l_eef_pos) + [1.0]).reshape(4,1)

        # left_Tm= np.concatenate((left_Rm_tmp, left_Pm), axis=1)
        # plotCoordinateFrame(ax, left_Tm, linestyle='--', size=0.1)

        # r_eef_pos = sample['eef_transforms'][7:10]
        # r_eef_quat = sample['eef_transforms'][10:]

        # right_Rm = sciR.from_quat([r_eef_quat[1],r_eef_quat[2],r_eef_quat[3],r_eef_quat[0]]).as_matrix()
        # right_Rm_tmp = np.concatenate((right_Rm, np.zeros([1,3])), axis=0)
        # right_Pm = np.array(list(r_eef_pos) + [1.0]).reshape(4,1)

        # right_Tm= np.concatenate((right_Rm_tmp, right_Pm), axis=1)
        # plotCoordinateFrame(ax, right_Tm, size=0.1)

        # visualize_shape(next_dlo_keypoints.reshape(-1, 3), ax, clr='k', ld=1)
        # l_eef_pos = sample['next_eef_transforms'][0:3]
        # l_eef_quat = sample['next_eef_transforms'][3:7]

        # left_Rm = sciR.from_quat([l_eef_quat[1],l_eef_quat[2],l_eef_quat[3],l_eef_quat[0]]).as_matrix()
        # left_Rm_tmp = np.concatenate((left_Rm, np.zeros([1,3])), axis=0)
        # left_Pm = np.array(list(l_eef_pos) + [1.0]).reshape(4,1)

        # left_Tm= np.concatenate((left_Rm_tmp, left_Pm), axis=1)
        # plotCoordinateFrame(ax, left_Tm, linestyle='--', size=0.1)

        # r_eef_pos = sample['next_eef_transforms'][7:10]
        # r_eef_quat = sample['next_eef_transforms'][10:]

        # right_Rm = sciR.from_quat([r_eef_quat[1],r_eef_quat[2],r_eef_quat[3],r_eef_quat[0]]).as_matrix()
        # right_Rm_tmp = np.concatenate((right_Rm, np.zeros([1,3])), axis=0)
        # right_Pm = np.array(list(r_eef_pos) + [1.0]).reshape(4,1)

        # right_Tm= np.concatenate((right_Rm_tmp, right_Pm), axis=1)
        # plotCoordinateFrame(ax, right_Tm, size=0.1)
        # plt.show()