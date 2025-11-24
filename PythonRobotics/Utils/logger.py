"""
    Loggers
"""
import os
from typing import List, Dict, Optional
import numpy as np
import yaml
import json
import zarr
import wandb
import time
from skvideo.io import vwrite
import PIL
import mediapy as media
import itertools


class BaseLogger(object):
    '''
        base logger class
    '''
    def __init__(self, path_to_save, name: str = None):
       directory = os.path.dirname(path_to_save)
       if not os.path.exists(directory):
            print("Making new directory at {}".format(directory))
            os.makedirs(directory)
       self._path = path_to_save
       self._logger_name = name

    @property
    def logger_name(self):
        return self._logger_name
    

class JsonLogger:
    '''
        Json Logger: log simple data into json file
    '''
    def __init__(self, path_to_save: str):
        self.check_dir(path_to_save)
        self._path = path_to_save
        self._dict_data = dict()

    def check_dir(self, path_to_save):
        directory = os.path.dirname(path_to_save)
        if not os.path.exists(directory):
            print("Making new directory at {}".format(directory))
            os.makedirs(directory)

    def log_kv(self, k: str, v):
        self._dict_data[k] = v

    def log_dict_data(self, dict_data):
        for key, val in dict_data.items():
            self._dict_data[key] = val

    def save_data(self):
        with open(self._path, 'w') as f:
            json.dump(self._dict_data, f, indent=4)
        return
    

class YamlLogger:
    '''
        Yaml Logger: log data into yaml file
    '''
    def __init__(self, path_to_save: str):
        self._path = path_to_save
        self._dict_data = dict()

    def log_kv(self, k: str, v):
        self._dict_data[k] = v


    def log_dict_data(self, dict_data):
        for key, val in dict_data.items():
            self._dict_data[key] = val

    def save_data(self):
        with open(self._path, 'w') as f:
            yaml.dump(self._dict_data, f)
        return None


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
        return self._path


class VideoLogger(BaseLogger):
    '''
        Video Logger: save and create video
    '''
    def __init__(self, path_to_save: str, **kargs):
        super().__init__(path_to_save, **kargs)
        self._video_path = path_to_save
        self._frames = list()
        
    def log_frame(self, frame):
        self._frames.append(frame)
        
    def create_video(self):
        if len(self._frames) > 0 and self._frames[0] is not None:
            vwrite(self._video_path, self._frames)
        else:
            print('No frames exist. Fail to create video.')
            return
    
    @property
    def num_frame(self):
        return len(self._frames)
    
    @property
    def frame_dtype(self):
        return type(self._frames[0])
        

class VideoLoggerPro:
    def __init__(self, path_to_save: str, fps:int=60):
        self._video_path = path_to_save
        self._fps = fps
        self._frames = list()
        
    def log_frame(self, frame):
        self._frames.append(frame)
        
    def create_video(self, imgs=None):
        if imgs is None:
            if len(self._frames) > 0 and self._frames[0] is not None:
                media.write_video(self._video_path, self._frames, fps=self._fps)
            else:
                print('No frames exist. Fail to create video.')
                return
        else:
            media.write_video(self._video_path, imgs, fps=self._fps)
        return None
    
    @property
    def num_frame(self):
        return len(self._frames)
    
    @property
    def frame_dtype(self):
        return type(self._frames[0])


if __name__ == '__main__':
    video_logger = VideoLoggerPro(path_to_save='./tmp_slow.mp4', fps=20)
    
    from mediapy import moving_circle, show_video
    images = moving_circle((480, 640), 60)
    for image in images:
        video_logger.log_frame(image)
    video_logger.create_video()
    # show_video(images, fps=20, title='Move Circle')