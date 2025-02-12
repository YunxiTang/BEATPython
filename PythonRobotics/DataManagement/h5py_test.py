# h5py for large data storage
import h5py
import numpy as np
import os

current_path = os.path.dirname(__file__)
file_name = os.path.join(current_path, 'test.h5')

# write data
f = h5py.File(file_name, mode='w')

test_data = np.array([1,2,3,4,5,6])
f.create_dataset('test_data', data=test_data)

g1 = f.create_group(name='g1')
f1 = g1.create_dataset(name='joint_pos', shape=(200, 7), data=np.zeros((200, 7)))

f.close()

# read data
new_f = h5py.File(file_name, 'r')

print(new_f['g1']['joint_pos'][:])

