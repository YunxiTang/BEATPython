import os
from typing import List, Dict, Optional
import numpy as np
import yaml
import json
import zarr
import mediapy as media
import torch
import random


def setup_seed(seed):
    """set all the seed

    Args:
        seed (int): seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)


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
        assert k in self._data_ks, 'data key, {}, is not in the data key list [{}]'.format(k, self._data_ks)
        self._data[k].append(v)


    def log_meta(self, k: str, v: np.ndarray):
        assert k in self._meta_ks, 'meta key, {}, is not in the meta key list [{}]'.format(k, self._meta_ks)
        self._meta[k].append(v)
        

    def save_data(self):
        self._root = zarr.open(self._path, 'w')
        self._data_ptr = self._root.create_group('data', overwrite=True)
        self._meta_ptr = self._root.create_group('meta', overwrite=True)

        for key in self._data_ks:
            print(f'saving {key} data')
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
            print(f'saving {key} data')
            meta = np.array( self._meta[key] )
            meta_shape = meta.shape
            chunk_shape = (self._chunk_size, ) + (None,) * (len(meta_shape) -  1)
            meta_zarr = self._meta_ptr.create_dataset(key,
                                                      shape=meta_shape,
                                                      dtype=self._dtype,
                                                      chunks=chunk_shape, 
                                                      overwrite=True)
            meta_zarr[:] = meta
        return self._path


class VideoLoggerPro:
    def __init__(self, path_to_save: str, fps:int=60):
        self._video_path = path_to_save
        self._fps = fps
        self._frames = list()
        
    def log_frame(self, frame):
        self._frames.append(frame)
        
    def create_video(self, imgs=None, fps=None):
        video_fps = fps if fps is not None else self._fps
        if imgs is None:
            if len(self._frames) > 0 and self._frames[0] is not None:
                media.write_video(self._video_path, self._frames, fps=video_fps)
            else:
                print('No frames exist. Fail to create video.')
        else:
            media.write_video(self._video_path, imgs, fps=video_fps)
        return None
    
    @property
    def num_frame(self):
        return len(self._frames)
    
    @property
    def frame_dtype(self):
        return type(self._frames[0])