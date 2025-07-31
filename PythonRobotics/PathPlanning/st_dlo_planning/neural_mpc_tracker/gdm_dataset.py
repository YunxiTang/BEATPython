"""
dataset for global deformation model learning
"""

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
from scipy.interpolate import splprep, splev
import zarr


class ReplayBuffer(Dataset):
    def __init__(self, args=None, obs_dim=None, act_dim=None, buffer_capacity=int(1e6)):
        """
        Replay Buffer
        """
        super(ReplayBuffer, self).__init__()
        if args is not None:
            self.obs_dim = args.obs_dim
            self.act_dim = args.act_dim
            self.max_size = int(args.buffer_capacity)
        else:
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            self.max_size = int(buffer_capacity)

        self.pointer = 0
        self.size = 0
        self.s = np.zeros((self.max_size, self.obs_dim), np.float32)
        self.a = np.zeros((self.max_size, self.act_dim), np.float32)
        self.r = np.zeros((self.max_size, 1), np.float32)
        self.s_ = np.zeros((self.max_size, self.obs_dim), np.float32)
        self.dw = np.zeros((self.max_size, 1), bool)

    def dict_flatten(self, obs):
        dlo_kp = obs["dlo_keypoints"]
        eef_states = obs["lowdim_eef_transforms"]
        s = np.hstack((dlo_kp, eef_states))
        return s

    def store(self, s, a, r, s_, dw):
        """
        store a tuple
        """
        self.s[self.pointer] = self.dict_flatten(s)
        self.a[self.pointer] = a
        self.r[self.pointer] = r
        self.s_[self.pointer] = self.dict_flatten(s_)
        self.dw[self.pointer] = dw
        # if reaches max_size, reset pointer to 0
        self.pointer = (self.pointer + 1) % self.max_size
        # count number of transitions
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Random Sample a batch size
        return: <s, a, r, s_, done>
        """
        assert self.size > 0, "Buffer is empty"
        index = np.random.choice(self.size, size=batch_size)
        batch_s = self.s[index]
        batch_a = self.a[index]
        batch_r = self.r[index]
        batch_s_ = self.s_[index]
        batch_dw = self.dw[index]
        return batch_s, batch_a, batch_r, batch_s_, batch_dw

    def reset(self):
        """
        Reset the whole buffer.
        Only call this method when training on-policy agent!!!
        """
        self.s.fill(0.0)
        self.a.fill(0.0)
        self.r.fill(0.0)
        self.s_.fill(0.0)
        self.dw.fill(False)
        self.pointer = 0
        self.size = 0
        return None

    def sample_all(self):
        """
        sample all the data in replay buffer
        """
        assert self.size > 0, "Buffer is empty"

        index = range(self.size)
        batch_s = self.s[index]
        batch_a = self.a[index]
        batch_r = self.r[index]
        batch_s_ = self.s_[index]
        batch_dw = self.dw[index]
        return batch_s, batch_a, batch_r, batch_s_, batch_dw

    def sample_recent_n_transits(self, n):
        """
        sample recent n transitions
        """
        if self.size < n:
            print(f"Buffer size is less than n ({n}). Set n as buffer size {self.size}")
            n = self.size

        if (self.pointer + 1) >= n:
            index = range(self.pointer + 1 - n, self.pointer + 1)
        else:
            index_p1 = range(0, self.pointer + 1)
            index_p2 = range(self.size - (n - self.pointer - 1), self.size)
            index = list(index_p1) + list(index_p2)

        batch_s = self.s[index]
        batch_a = self.a[index]
        batch_r = self.r[index]
        batch_s_ = self.s_[index]
        batch_dw = self.dw[index]
        return batch_s, batch_a, batch_r, batch_s_, batch_dw

    def save_data(self, path_to_save: str):
        """save the data

        Args:
            path_to_save (str): path to save the collected data
        """
        np.savez(
            path_to_save,
            state=self.s,
            action=self.a,
            reward=self.r,
            next_state=self.s_,
            dones=self.dw,
            pointer=self.pointer,
            size=self.size,
        )

    def load_data(self, file_path_to_load):
        """
        load the stored data
        """
        data = np.load(file_path_to_load)

        self.pointer = data["pointer"]
        self.size = data["size"]
        self.s = data["state"]
        self.a = data["action"]
        self.r = data["reward"]
        self.s_ = data["next_state"]
        self.dw = data["dones"]
        self.max_size = self.s.shape[0]

    def get_size(self):
        return self.size

    def __getitem__(self, index):
        """
        For regression training
        """
        batch_s = self.s[index]
        batch_a = self.a[index]
        batch_r = self.r[index]
        batch_s_ = self.s_[index]
        batch_dw = self.dw[index]
        return (batch_s, batch_a, batch_r, batch_s_, batch_dw), (
            batch_s,
            batch_a,
            batch_r,
            batch_s_,
            batch_dw,
        )

    def __len__(self):
        return self.size


def visualize_shape(dlo: np.ndarray, ax, ld=3.0, s=25, clr=None):
    """
    visualize a rope shape
    """
    if clr is None:
        clr = 0.5 + 0.5 * np.random.random(3)

    num_kp = dlo.shape[0]

    for i in range(num_kp):
        ax.scatter(dlo[i][0], dlo[i][1], dlo[i][2], color=clr, marker="o", s=s)
    for i in range(num_kp - 1):
        ax.plot3D(
            [dlo[i][0], dlo[i + 1][0]],
            [dlo[i][1], dlo[i + 1][1]],
            [dlo[i][2], dlo[i + 1][2]],
            color=clr,
            linewidth=ld,
        )
    ax.axis("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def plotCoordinateFrame(axis, T_0f, size=1, linestyle="-", linewidth=3, name=None):
    """draw a coordinate frame on a 3d axis.
    In the resulting plot, ```x = red, y = green, z = blue```

    ```plotCoordinateFrame(axis, T_0f, size=1, linewidth=3)```

    Arguments:
    ```axis```: an axis of type matplotlib.axes.Axes3D
    ```T_0f```: The 4x4 transformation matrix that takes points from the frame of interest, to the plotting frame
    ```size```: the length of each line in the coordinate frame
    ```linewidth```: the width of each line in the coordinate frame
    """

    p_f = np.array([[0, 0, 0, 1], [size, 0, 0, 1], [0, size, 0, 1], [0, 0, size, 1]]).T
    p_0 = np.dot(T_0f, p_f)

    X = np.append([p_0[:, 0].T], [p_0[:, 1].T], axis=0)
    Y = np.append([p_0[:, 0].T], [p_0[:, 2].T], axis=0)
    Z = np.append([p_0[:, 0].T], [p_0[:, 3].T], axis=0)
    axis.plot3D(X[:, 0], X[:, 1], X[:, 2], f"r{linestyle}", linewidth=linewidth)
    axis.plot3D(Y[:, 0], Y[:, 1], Y[:, 2], f"g{linestyle}", linewidth=linewidth)
    axis.plot3D(Z[:, 0], Z[:, 1], Z[:, 2], f"b{linestyle}", linewidth=linewidth)

    if name is not None:
        axis.text(X[0, 0], X[0, 1], X[0, 2], name, zdir="x")


def normalize_data(stats, delta_shape=None, delta_ee_pos=None):
    """
    normalize the input/output
    """
    if delta_shape is not None:
        normalized_delta_shape = (delta_shape - stats["delta_shape"]["min"]) / (
            stats["delta_shape"]["max"] - stats["delta_shape"]["min"] + 1e-6
        )
        normalized_delta_shape = normalized_delta_shape * 2 - 1
    else:
        normalized_delta_shape = None

    if delta_ee_pos is not None:
        normalized_delta_ee_pos = (delta_ee_pos - stats["delta_ee_pos"]["min"]) / (
            stats["delta_ee_pos"]["max"] - stats["delta_ee_pos"]["min"] + 1e-6
        )
        normalized_delta_ee_pos = normalized_delta_ee_pos * 2 - 1
    else:
        normalized_delta_ee_pos = None

    return normalized_delta_shape, normalized_delta_ee_pos


def unnormalize_data(stats, normalized_delta_shape=None, normalized_delta_ee_pos=None):
    if normalized_delta_shape is not None:
        normalized_delta_shape = (normalized_delta_shape + 1) / 2
        unnormalized_delta_shape = (
            normalized_delta_shape
            * (stats["delta_shape"]["max"] - stats["delta_shape"]["min"])
            + stats["delta_shape"]["min"]
        )
    else:
        unnormalized_delta_shape = None

    if normalized_delta_ee_pos is not None:
        normalized_delta_ee_pos = (normalized_delta_ee_pos + 1) / 2
        unnormalized_delta_ee_pos = (
            normalized_delta_ee_pos
            * (stats["delta_ee_pos"]["max"] - stats["delta_ee_pos"]["min"])
            + stats["delta_ee_pos"]["min"]
        )
    else:
        unnormalized_delta_ee_pos = None

    return unnormalized_delta_shape, unnormalized_delta_ee_pos


def fit_bspline(keypoints, num_samples=13, degree=3) -> np.ndarray:
    keypoints = np.array(keypoints)  # Ensure it's a NumPy array
    # num_points, dim = keypoints.shape  # Get number of points and dimensionality

    # Fit a B-Spline through the keypoints
    tck, _ = splprep(keypoints.T, s=0, k=degree)  # Transpose keypoints for splprep

    # Sample points along the fitted spline
    u_fine = np.linspace(0, 1, num_samples)  # Parametric range
    spline_points = splev(
        u_fine, tck
    )  # Returns a list of arrays, one for each dimension

    # Stack the arrays into a single (num_samples, dim) array
    spline_points = np.vstack(spline_points).T

    return spline_points


class MultiStepGDMDataset(Dataset):
    def __init__(self, data_path: str, max_step: int = 10, min_step: int = 0):
        super(MultiStepGDMDataset, self).__init__()
        self.data_path = data_path

        root = zarr.open(self.data_path, "r")

        # self.actions = root['data']['action']

        self.dlo_keypoints = root["data"]["dlo_keypoints"]
        self.eef_states = root["data"]["eef_states"]
        self.eef_transforms = root["data"]["eef_transforms"]

        self.next_dlo_keypoints = root["data"]["next_dlo_keypoints"]
        self.next_eef_states = root["data"]["next_eef_states"]
        self.next_eef_transforms = root["data"]["next_eef_transforms"]

        self.ep_num = root["data"]["ep_num"]

        self.dlo_lens = root["meta"]["dlo_len"]

        if len(self.eef_states.shape) == 2:
            self.num_grasps = self.eef_states.shape[1] // 3
        else:  # == 3
            self.num_grasps = self.eef_states.shape[1]

        if len(self.dlo_keypoints.shape) == 2:  # (num_frame. dlo_kp)
            self.num_feats = self.dlo_keypoints.shape[1] // 3
        else:  # == 3
            self.num_feats = self.dlo_keypoints.shape[1]

        self.max_step = max_step
        self.capacity = self.dlo_lens.shape[0] - max_step

        if max_step > 0:
            self.steps = np.random.randint(
                min_step,
                self.max_step,
                size=[
                    self.capacity,
                ],
            )
        else:
            self.steps = [
                0,
            ] * self.capacity

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

        # system configuration
        dlo_keypoints = fit_bspline(
            self.dlo_keypoints[idx].reshape(self.num_feats, -1), 13
        ).astype(np.float32)
        dlo_keypoints = dlo_keypoints[:, 0:2]

        next_dlo_keypoints = fit_bspline(
            self.next_dlo_keypoints[next_idx].reshape(self.num_feats, -1), 13
        ).astype(np.float32)
        next_dlo_keypoints = next_dlo_keypoints[:, 0:2]

        eef_states = self.eef_states[idx].reshape(self.num_grasps, -1)

        # movement of eefs
        delta_eef = self.next_eef_states[next_idx] - self.eef_states[idx]
        delta_eef = delta_eef.reshape(self.num_grasps, -1)

        delta_shape = next_dlo_keypoints - dlo_keypoints

        eef_transforms = self.eef_transforms[idx]
        next_eef_transforms = self.next_eef_transforms[next_idx]

        output = {
            "dlo_keypoints": dlo_keypoints,
            "eef_states": eef_states,
            "delta_shape": delta_shape,
            "delta_eef": delta_eef,
            "next_dlo_keypoints": next_dlo_keypoints,
            "eef_transforms": eef_transforms,
            "next_eef_transforms": next_eef_transforms,
        }
        return output


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.transform import Rotation as sciR
    from torch.utils.data import DataLoader

    def local_transform(dlo_kp, eef_states):
        batch_size = dlo_kp.shape[0]

        dlo_kp_center = torch.mean(dlo_kp, dim=1, keepdim=True)
        local_dlo_kp = dlo_kp - dlo_kp_center

        tmp = torch.concatenate([dlo_kp_center, torch.zeros(batch_size, 1, 1)], dim=2)
        local_eef_states = eef_states - tmp

        return local_dlo_kp, local_eef_states, dlo_kp_center

    data_path = (
        "/media/yxtang/Extreme SSD/HDP/gdm_dataset/train/task_ethernet_cable.zarr"
    )
    dataset = MultiStepGDMDataset(data_path, max_step=5)
    for key, val in dataset[0].items():
        print(key, ": ", val.shape)

    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        dlo_keypoints = batch["dlo_keypoints"]
        next_dlo_keypoints_gt = batch["next_dlo_keypoints"]

        eef_states = batch["eef_states"]
        dlo_kp, eef_states, dlo_kp_center = local_transform(dlo_keypoints, eef_states)

        delta_shape = batch["delta_shape"]

        dlo_keypoints_np = dlo_kp.to("cpu").detach().numpy()
        delta_shape_np = delta_shape.to("cpu").detach().numpy()

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(projection="3d")

        next_dlo_keypoints_np = dlo_keypoints_np + delta_shape_np
        next_dlo_keypoints_gt = next_dlo_keypoints_gt - dlo_kp_center
        next_dlo_keypoints_gt_np = next_dlo_keypoints_gt.to("cpu").detach().numpy()

        for i in range(batch_size):
            # lifted_dlo_kp = np.concatenate((dlo_keypoints_np[i], np.zeros((dataset.num_feats, 1))), axis=1)
            # visualize_shape(lifted_dlo_kp, ax, clr='r', ld=1)

            lifted_next_dlo_kp = np.concatenate(
                (next_dlo_keypoints_np[i], np.zeros((dataset.num_feats, 1))), axis=1
            )
            visualize_shape(lifted_next_dlo_kp, ax, clr="g", ld=1)

            lifted_next_dlo_gt_kp = np.concatenate(
                (next_dlo_keypoints_gt_np[i], np.zeros((dataset.num_feats, 1))), axis=1
            )
            visualize_shape(lifted_next_dlo_gt_kp, ax, clr="b", ld=1)

            ax.scatter(
                eef_states[i, 0, 0],
                eef_states[i, 0, 1],
                0.0,
                color="k",
                marker="o",
                s=15,
            )
            ax.scatter(
                eef_states[i, 1, 0],
                eef_states[i, 1, 1],
                0.0,
                color="k",
                marker="o",
                s=15,
            )

        plt.show()
        # break

    exit()

    for _ in range(100):
        i = random.randint(0, dataset.capacity)
        sample = dataset[i]

        dlo_keypoints = sample["dlo_keypoints"]
        delta_shape = sample["delta_shape"]
        delta_eef = sample["delta_eef"]

        next_dlo_keypoints = dlo_keypoints + delta_shape

        print(np.linalg.norm(delta_shape), np.linalg.norm(delta_eef))
        print("=======================")

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(projection="3d")

        lifted_dlo_kp = np.concatenate(
            (dlo_keypoints, np.zeros((dlo_keypoints.shape[0], 1))), axis=1
        )
        lifted_next_dlo_kp = np.concatenate(
            (next_dlo_keypoints, np.zeros((dlo_keypoints.shape[0], 1))), axis=1
        )
        visualize_shape(lifted_dlo_kp, ax, clr="r", ld=1)
        visualize_shape(lifted_next_dlo_kp, ax, clr="k", ld=1)
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
        plt.show()
