''' convert zarr data into hf dataset'''
import os
import zarr
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
import numpy as np
import glob


if __name__ == '__main__':
    features = {}
    features["state"] = Sequence(feature=Value(dtype="float32", id=None), length=(10+2)*3)
    features["action"] = Sequence(
        feature=Value(dtype="float32", id=None), length=2*3
    )
    features["next_state"] = Sequence(
        feature=Value(dtype="float32", id=None), length=(10+2)*3
    )
    features["dlo_len"] = Value(dtype='float32', id=None)

    # grab the data
    data_dict = {}

    train_data_paths = glob.glob(os.path.join('./train', '*_train.zarr'))

    for train_path in train_data_paths:
        print(train_path)
        root = zarr.open(train_path)
    data_dict['state'] = root['data']['state'][:]
    data_dict['next_state'] = root['data']['next_state'][:]
    data_dict['action'] = root['data']['action'][:]
    data_dict['dlo_len'] = root['meta']['dlo_len'][:]
    print(data_dict['next_state'].shape, data_dict['dlo_len'].shape)
    
    ds = Dataset.from_dict(data_dict, features=Features(features))
    ds.push_to_hub("YXTang/private_gdm_mj", private=True)
    exit()
    ds.save_to_disk('/home/yxtang/CodeBase/DOBERT/datasets/gdm_mj/hf_data/train')
    ds = ds.with_format('torch', device='cuda:0')
