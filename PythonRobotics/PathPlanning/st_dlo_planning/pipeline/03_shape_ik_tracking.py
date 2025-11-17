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

    import matplotlib.pyplot as plt
    from st_dlo_planning.envs.planar_cable_deform.planar_cable_deform import (
        DualGripperCableEnv,
    )
    from st_dlo_planning.neural_mpc_tracker.policy import BroydenAgent
    import seaborn as sns
    from st_dlo_planning.utils.world_map import Block, WorldMap, MapCfg, plot_circle
    from st_dlo_planning.utils.path_set import PathSet
    from omegaconf import OmegaConf


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
    log = bool(0)

    task = "03"
    env = DualGripperCableEnv(task, feat_stride=3, render_mode="human")

    num_feats = env.num_feat
    print(num_feats)
    num_grasps = env.num_grasp

    agent = BroydenAgent(input_dim=3 * num_grasps, output_dim=3 * num_feats)

    map_case = "map_case1"

    result_path = pathlib.Path(__file__).parent.parent.joinpath(
        "results", f"{map_case}_optimal_shape_seq.npy"
    )
    planned_shape_seq = np.load(result_path, mmap_mode="r")
    planned_shape_seq = planned_shape_seq.reshape(-1, 13, 3)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(projection="3d")
    for p in range(planned_shape_seq.shape[0]):
        visualize_shape(planned_shape_seq[p], ax)
    plt.axis("equal")
    plt.show()

    pathset_list = []
    for k in range(num_feats):
        spath = []
        for q in range(planned_shape_seq.shape[0]):
            spath.append(planned_shape_seq[q, k])
        pathset_list.append(spath)

    pathset = PathSet(pathset_list, T=60, seg_len=0.3 / (num_feats - 1))

    final_target_shape = planned_shape_seq[-1].flatten()

    # ================ move the cable to a proper shape (if neccessary) ============
    state, _ = env.reset()
    for i in range(20):
        action = np.array([0.02, 0.00, -0.4, -0.04, 0.00, 0.4]) * 1.0
        action = np.clip(action, -0.5, 0.5)
        next_state, _, _, _, _ = env.step(action)
        env.render()

    N = 15000
    error_list = []
    ultra_error_list = []
    action_list = []
    error_pre = 1e3

    i = 0
    pre_action = 0.0

    dlos = []
    target_dlos = []

    jj = 0
    patience = 0
    while i < N:
        target_shape = planned_shape_seq[jj]
        target_shape = target_shape.flatten()
        agent.set_target_q(target_shape)

        action, _ = agent.select_action(state["dlo_keypoints"], alpha=0.01)
        action = (
            np.clip(
                action,
                [-0.04, -0.04, -0.4, -0.04, -0.04, -0.4],
                [0.04, 0.04, 0.4, 0.04, 0.04, 0.4],
            )
            * 1.0
        )

        action = 0.5 * pre_action + 0.5 * action
        # action[2] = 0.0
        # action[5] = 0.0
        # ================== take an env step ==================#
        action_list.append(action)
        next_state, reward, done, truncated, info = env.step(action)

        env.render(mode="human")

        error = np.linalg.norm(target_shape - state["dlo_keypoints"], 2) / num_feats
        ultra_error = (
            np.linalg.norm(final_target_shape - state["dlo_keypoints"], 2) / num_feats
        )
        error_list.append(error)
        ultra_error_list.append(ultra_error)

        # ============== update Jacobian ========================
        delta_s = next_state["dlo_keypoints"] - state["dlo_keypoints"]
        delta_x = action * env.dt
        delta_s = delta_s.reshape((3 * num_feats, 1))
        delta_x = delta_x.reshape((6, 1))
        if i % 1 == 0:
            #     print('update jacobian', np.linalg.norm(delta_s))
            agent.update(delta_s, delta_x)

        # ================== log printing ====================== #
        if i % 20 == 0:
            print(
                f"Timestep: [{i} / {N}] || Inter_Err: {error} Ultra_Err: {ultra_error} Idx: {jj}"
            )

        if abs(error) < 3e-3 or (patience > 250 and jj > 0):
            jj += 1
            jj = min(jj, planned_shape_seq.shape[0] - 1)
            dlos.append(state["dlo_keypoints"])
            target_dlos.append(target_shape.reshape(-1, 3))
            patience = 0

        elif abs(error) < 1e-2 and jj == 0:
            jj += 1
            jj = min(jj, planned_shape_seq.shape[0] - 1)
            dlos.append(state["dlo_keypoints"])
            target_dlos.append(target_shape.reshape(-1, 3))
            patience = 0

        if ultra_error <= 5e-3:
            dlos.append(state["dlo_keypoints"])
            target_dlos.append(final_target_shape.reshape(-1, 3))
            break

        error_pre = error
        state = next_state
        pre_action = action
        i += 1
        patience += 1
    action_list = np.array(action_list)
    # ======================= print and save result ======================== #
    print(f"residual shape error {error}")

    # ======================= plot ======================== #
    import matplotlib.pyplot as plt
    import seaborn as sns
    from st_dlo_planning.utils.path_interpolation import visualize_shape

    sns.set_theme("notebook")
    sns.lineplot(error_list)
    sns.lineplot(ultra_error_list)

    plt.figure(2)
    for j in range(6):
        sns.lineplot(action_list[:, j])
    plt.show()

    cfg_path = f"/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/envs/map_cfg/{map_case}.yaml"
    map_cfg_file = OmegaConf.load(cfg_path)

    map_cfg = MapCfg(
        resolution=map_cfg_file.workspace.resolution,
        map_xmin=map_cfg_file.workspace.map_xmin,
        map_xmax=map_cfg_file.workspace.map_xmax,
        map_ymin=map_cfg_file.workspace.map_ymin,
        map_ymax=map_cfg_file.workspace.map_ymax,
        map_zmin=map_cfg_file.workspace.map_zmin,
        map_zmax=map_cfg_file.workspace.map_zmax,
        robot_size=map_cfg_file.workspace.robot_size,
        dim=3,
    )

    world_map = WorldMap(map_cfg)
    # ============== add some obstacles =========================
    size_z = map_cfg_file.workspace.map_zmax
    obstacles = map_cfg_file.obstacle_info.obstacles
    i = 0
    for obstacle in obstacles:
        world_map.add_obstacle(
            Block(
                obstacle[0],
                obstacle[1],
                size_z,
                obstacle[2],
                obstacle[3],
                angle=obstacle[4] * np.pi,
                clr=[0.3 + 0.01 * i, 0.5, 0.4],
            )
        )
        i += 1
    world_map.finalize()
    _, ax = world_map.visualize_passage(full_passage=False)

    for k in range(0, len(dlos), 1):
        dlo_shape = dlos[k]
        dlo_shape = dlo_shape.reshape(-1, 3)
        visualize_shape(dlo_shape, ax, clr="k", ld=2.0)
        visualize_shape(target_dlos[k], ax, clr="r", ld=1.0)
    plt.axis("equal")
    plt.savefig(
        f"/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/tracking_res_{map_case}.png",
        dpi=2000,
    )
    plt.show()
