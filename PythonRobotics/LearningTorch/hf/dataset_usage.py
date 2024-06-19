from datasets import Dataset, DatasetDict
from datasets import Sequence, Value, Image, Features
from datasets import load_dataset_builder, load_dataset, load_from_disk
from datasets import get_dataset_split_names
import torch
import numpy as np
from torch.utils.data import DataLoader


def test1():
    ds_builder = load_dataset_builder("lerobot/pusht_image")
    print( ds_builder.info.description )
    print( ds_builder.info.dataset_size )
    for key, val in ds_builder.info.features.items():
        print( f'{key}: {val}' )
    print( get_dataset_split_names("lerobot/pusht_image") )


def test2():
    data_x = np.linspace(0., 10., 1000).reshape(-1, 1)
    data_y = 2 * np.sin(data_x) + 0.5
    ds = Dataset.from_dict({"data": data_x,
                            "label": data_y}, split='train')
    
    ds = ds.with_format("torch")
    for key, val in ds.info.features.items():
        print( f'{key}: {val}' )

    dataloader = DataLoader(ds, batch_size=4)
    for batch in dataloader:
        print(batch['data']) 
        break

    ds.save_to_disk('../../dataset/hf_ds')
    ds.to_parquet('../../dataset/hf_ds/tmp.parquet')

if __name__ == '__main__':
    test2()