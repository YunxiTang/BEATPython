"""
sample some feasible DLO shapes from simulator
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.transform import Rotation as sciR
    from tqdm import tqdm

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)

    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    import matplotlib.pyplot as plt
    from st_dlo_planning.envs.planar_cable_deform.planar_cable_deform import (
        DualGripperCableEnv,
    )
    from st_dlo_planning.utils.misc_utils import ZarrLogger
    import seaborn as sns


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


def traj_generation(env: DualGripperCableEnv, num_steps: int, data_logger):
    """
    generate target trajectory
    """
    actions = []
    states = []
    next_states = []
    state, _ = env.reset()
    for _ in tqdm(range(num_steps), desc="Target Generation"):
        action = np.array([-0.02, 0.03, 0.3, 0.04, 0.01, 0.4]) * 1.0
        action = np.clip(action, -0.5, 0.5)
        next_state, _, _, _, _ = env.step(action)

        data_logger.log_data("dlo_keypoints", state["dlo_keypoints"])
        data_logger.log_meta("dlo_len", env.dlo_len)

        env.render()
        actions.append(action)
        states.append(state)
        next_states.append(next_state)
        tmp = np.linalg.norm(next_state["dlo_keypoints"] - state["dlo_keypoints"], 2)

        if tmp < 1e-5:
            break
        state = next_state
    return actions, states, next_states


if __name__ == "__main__":
    task = "03"
    env = DualGripperCableEnv(task, feat_stride=3, render_mode="human")

    num_feats = env.num_feat
    num_grasps = env.num_grasp

    data_path = "/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/dlo_03_samples_2.zarr"

    data_logger = ZarrLogger(
        path_to_save=data_path,
        data_ks=["dlo_keypoints"],
        meta_ks=[
            "dlo_len",
        ],
    )

    actions, states, _ = traj_generation(env, 100, data_logger)

    for state in states:
        print(state["dlo_keypoints"].shape)

    data_logger.save_data()
