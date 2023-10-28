from replay_buffer import ReplayBuffer
import zarr


dataset_path = 'dataset/pusht_cchi_v7_replay.zarr'
group = zarr.open(dataset_path, 'r')
replay_buffer = ReplayBuffer(group)