import os
import math
import zarr
import numpy as np

from typing import Union, Dict


class ReplayBuffer:
    '''
        Zarr-based temporal datastructure. Used as a replay buffer in learning
    '''
    def __init__(self, root: Union[zarr.Group, Dict[str, dict]]):
        '''
            Dummy constructor. Use `copy_from*` and `create_from*` class methods instead.
        '''
        print('data struct: ')
        print(root.tree())

        # check required fields
        assert('data' in root)
        assert('meta' in root)
        assert('episode_ends' in root['meta'])

        # check shape
        for key, val in root['data'].items():
            assert( val.shape[0] == root['meta']['episode_ends'][-1] )
            print('{}: {}'.format(key, val.shape))

        self.root = root


    # ============= constructors ===============
    @classmethod
    def create_empty_zarr(cls, storage: zarr.MemoryStore =None, root: zarr.Group =None):
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore()
            root = zarr.group(storage)

        data = root.require_group('data', overwrite=False)
        meta = root.require_group('meta', overwrite=False)

        if 'episode_ends' not in meta:
            episode_ends = meta.zeros('episode_ends', shape=(0,), dtype=np.int64, compressor=None, overwrite=False)
        return cls(root=root)
    
    @classmethod
    def create_from_group(cls, group, **kwargs):
        if 'data' not in group:
            # create from stratch
            buffer = cls.create_empty_zarr(root=group, **kwargs)
        else:
            # already exist
            buffer = cls(root=group, **kwargs)
        return buffer
    
    @classmethod
    def create_from_path(cls, zarr_path, mode='r', **kwargs):
        """
        Open a on-disk zarr directly (for dataset larger than memory). (Slower)
        """
        group = zarr.open(os.path.expanduser(zarr_path), mode)
        return cls.create_from_group(group, **kwargs)

    # ============= copy constructors ==========


    # ============= save methods ===============


    # ============= properties =================


    # =========== dict-like API ================



    # =============== chunking =================