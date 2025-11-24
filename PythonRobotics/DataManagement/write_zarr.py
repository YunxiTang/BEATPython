import zarr
import numpy as np
import tqdm


# Creating an array
z = zarr.zeros(shape=(10000, 10000), 
               chunks=(1000, 1000), 
               dtype='f4')
print(z, z.chunks)

# Reading and writing data
z[:] = 42
z_np = np.array(z)
print(type(z_np))
print(z[0, 0:5])

# Persistent arrays
z1 = zarr.open('dataset/example.zarr', mode='w', 
               shape=(1000, 1000), chunks=(100, 100), 
               dtype='i4')

for i in tqdm.tqdm(range(1000)):
    z1[i,...] = np.random.rand(1000)
z1[0, :] = np.arange(1000)
z2 = zarr.open('dataset/example.zarr', mode='r')
print(np.all(z1[:] == z2[:]))

# Groups
root = zarr.group('dataset/example2.zarr', 'w')
foo = root.create_group('foo')
bar = root.create_group('bar')
# group can contain arrays
z3 = bar.zeros('baz', shape=(1000, 1000), chunks=(100, 100), dtype='i4')
z4 = foo.create_dataset('fooz', shape=(1000, 1000), chunks=(100, 100), dtype='i4')

for key, val in root.items():
    print(key, val)

tmp = ['sd', 'st']
print('=============', 'foo' in root, '=============')
print('=============', 'baz' in bar, '=============')
print('=============', 'sd' in tmp, '=============')

print(z3, z4)
print(root['foo']['fooz'], root['foo/fooz'])
print(root.tree())

tmp = zarr.open('dataset/example2.zarr', 'r')
print(type(tmp))
print(tmp.tree())