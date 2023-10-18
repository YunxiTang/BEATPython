import zarr
import numpy as np
import tqdm


tmp = zarr.open('dataset/example2.zarr', 'r')
print(type(tmp))
print(tmp.tree())