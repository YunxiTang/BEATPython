if __name__ == "__main__":
    import sys
    import os
    import pathlib
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.transform import Rotation as sciR

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)

    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from st_dlo_planning.envs.planar_cable_deform.planar_cable_deform import (
        DualGripperCableEnv,
    )
    from st_dlo_planning.neural_mpc_tracker.policy import BroydenAgent
    import seaborn as sns

    sns.set_theme("paper")

    mode = "human"  #'rgb_array' #'depth_array' #
    camera = "topview"
    env = DualGripperCableEnv(
        task="03", feat_stride=3, render_mode=mode, camera_name=camera
    )
    num_feats = env.num_feat
    print(env.num_feat, env.na, env.dt)
    obs, _ = env.reset()

    # save visualization and rewards
    imgs = [env.render(mode=mode, camera_name=camera)]

    dlo_keypoints_hist = []
    gripper_transform_hist = []
    left_eef_twists = []
    right_eef_twists = []
    desired_twists = []
    ts = []

    pre_left_euler_z = 0.0
    for i in range(80):
        ts.append(i)
        action = np.array([0.02, 0.03, 0.2, -0.01, 0.03, 0.2])
        obs, _, done, _, _ = env.step(action)

        gripper_transforms = obs["eef_transforms"]

        l_eef_quat = gripper_transforms[3:7]
        left_R = sciR.from_quat(
            [l_eef_quat[1], l_eef_quat[2], l_eef_quat[3], l_eef_quat[0]]
        )
        left_euler = left_R.as_euler("xyz", degrees=False)
        print(
            "left_euler_z_rate: ",
            left_euler[2],
            "left_euler_z_rate: ",
            (left_euler[2] - pre_left_euler_z) / env.dt,
        )

        left_eef_twist, right_eef_twist = env.get_eef_twist()

        dlo_keypoints_hist.append(obs["dlo_keypoints"])
        gripper_transform_hist.append(obs["eef_transforms"])
        left_eef_twists.append(left_eef_twist[None])
        right_eef_twists.append(right_eef_twist[None])
        desired_twists.append(action[None])

        imgs.append(env.render(mode=mode, camera_name=camera))

        pre_left_euler_z = left_euler[2]

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(projection="3d")
    for i in range(0, len(dlo_keypoints_hist), 5):
        dlo_keypoints = dlo_keypoints_hist[i]
        gripper_transforms = gripper_transform_hist[i]

        l_eef_pos = gripper_transforms[0:3]
        l_eef_quat = gripper_transforms[3:7]

        left_Rm = sciR.from_quat(
            [l_eef_quat[1], l_eef_quat[2], l_eef_quat[3], l_eef_quat[0]]
        ).as_matrix()
        left_Rm_tmp = np.concatenate((left_Rm, np.zeros([1, 3])), axis=0)
        left_Pm = np.array(list(l_eef_pos) + [1.0]).reshape(4, 1)

        left_Tm = np.concatenate((left_Rm_tmp, left_Pm), axis=1)
        plotCoordinateFrame(ax, left_Tm, size=0.2)

        r_eef_pos = gripper_transforms[7:10]
        r_eef_quat = gripper_transforms[10:]

        right_Rm = sciR.from_quat(
            [r_eef_quat[1], r_eef_quat[2], r_eef_quat[3], r_eef_quat[0]]
        ).as_matrix()
        right_Rm_tmp = np.concatenate((right_Rm, np.zeros([1, 3])), axis=0)
        right_Pm = np.array(list(r_eef_pos) + [1.0]).reshape(4, 1)

        right_Tm = np.concatenate((right_Rm_tmp, right_Pm), axis=1)
        plotCoordinateFrame(ax, right_Tm, size=0.2)

        visualize_shape(dlo_keypoints.reshape(-1, 3), ax, clr="r")
        ax.scatter(
            l_eef_pos[0], l_eef_pos[1], l_eef_pos[2], color="k", marker="d", s=45
        )
        ax.scatter(
            r_eef_pos[0], r_eef_pos[1], r_eef_pos[2], color="k", marker="d", s=45
        )

    plt.show()

    fig = plt.figure(figsize=plt.figaspect(0.5))
    left_eef_twists = np.concatenate(left_eef_twists, axis=0)

    desired_twists = np.concatenate(desired_twists, axis=0)

    plt.plot(ts, left_eef_twists[:, 3], "k-")
    plt.plot(ts, desired_twists[:, 0], "k--")

    plt.plot(ts, left_eef_twists[:, 4], "r-")
    plt.plot(ts, desired_twists[:, 1], "r--")

    plt.plot(ts, left_eef_twists[:, 2], "b-")
    plt.plot(ts, desired_twists[:, 5], "b--")

    plt.show()

    exit()
    # ============================================= #
    target_shape = dlo_keypoints_hist[-1]
    agent = BroydenAgent(input_dim=6, output_dim=num_feats * 3)
    agent.set_target_q(target_shape)

    # ================ move the cable to a proper shape (if neccessary) ============
    obs, _ = env.reset()
    Action = action * 0.2

    for i in range(20):
        env.step(Action)

    for _ in range(100):
        next_obs, reward, done, truncated, info = env.step(Action * 0)

    N = 5000
    obs = next_obs
    error_list = []
    action_list = np.zeros((N, 6))
    error_pre = 1e3

    i = 0
    pre_action = 0.0

    left_eef_pos = []

    while i < N:
        dlo_keypoints = obs["dlo_keypoints"]
        eef_transforms = obs["eef_transforms"]

        left_eef_pos.append(eef_transforms[0:3])

        raw_action, _ = agent.select_action(dlo_keypoints, alpha=0.2)
        action = (
            np.clip(
                action,
                [-0.04, -0.04, -0.1, -0.04, -0.04, -0.1],
                [0.04, 0.04, 0.1, 0.04, 0.04, 0.1],
            )
            * 2.0
        )

        action_list[i, :] = raw_action

        action = raw_action
        action = 0.5 * pre_action + 0.5 * action
        # ================== take an env step ==================#
        next_obs, reward, done, truncated, info = env.step(action)
        env.render(mode="human")

        next_dlo_keypoints = next_obs["dlo_keypoints"]
        next_eef_transforms = next_obs["eef_transforms"]

        error = np.linalg.norm(target_shape - next_dlo_keypoints, 2) / num_feats
        error_list.append(error)

        # ============== update Jacobian ========================
        delta_s = next_dlo_keypoints - dlo_keypoints
        delta_x = raw_action * env.dt
        delta_s = delta_s.reshape((3 * num_feats, 1))
        delta_x = delta_x.reshape((6, 1))
        if i % 3 == 0:
            agent.update(delta_s, delta_x)

        # ================== log printing ====================== #
        if i % 20 == 0:
            print(
                f"Timestep: [{i} / {N}] || Shape Error: {error} Shape Error Reduction: {error_pre - error}"
            )
            error_pre = error
        osb = next_obs
        pre_action = action
        i += 1

    # ======================= print and save result ======================== #
    print(f"residual shape error {error}")

    plt.plot(np.array(left_eef_pos)[:, 2], "b--")

    plt.show()
