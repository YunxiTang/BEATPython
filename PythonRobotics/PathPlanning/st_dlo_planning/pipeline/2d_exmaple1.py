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

    from tqdm import tqdm


def traj_generation(env: DualGripperCableEnv, num_steps: int, mode: str = "3d"):
    """
    generate target trajectory
    """
    actions = []
    states = []
    next_states = []
    state, _ = env.reset()
    for idx in tqdm(range(num_steps), desc="Target Generation"):
        action = np.array([0.03, 0.03, 0.2, 0.04, 0.01, 0.4]) * 1.0
        action = np.clip(action, -0.5, 0.5)
        next_state, _, _, _, _ = env.step(action)

        env.render()
        actions.append(action)
        states.append(state)
        next_states.append(next_state)
        tmp = np.linalg.norm(next_state["dlo_keypoints"] - state["dlo_keypoints"], 2)

        if tmp < 1e-5:
            break
        state = next_state

    for _ in range(100):
        env.step(action * 0)
        env.render()

    return actions, states, next_states, idx


if __name__ == "__main__":
    log = bool(0)

    task = "03"
    env = DualGripperCableEnv(task, feat_stride=3, render_mode="human")

    num_feats = env.num_feat
    num_grasps = env.num_grasp

    actions, target_shape, _, _ = traj_generation(env, 100)

    target_shape = target_shape[-1]["dlo_keypoints"]
    agent = BroydenAgent(input_dim=3 * num_grasps, output_dim=3 * num_feats)

    agent.set_target_q(target_shape)

    # ================ move the cable to a proper shape (if neccessary) ============
    state, _ = env.reset()
    Action = actions[0] * 1.0

    for i in range(5):
        next_state, reward, done, truncated, info = env.step(Action)

    N = 5000
    state = next_state
    error_list = []
    action_list = np.zeros((N, 6))
    error_pre = 1e3

    i = 0
    pre_action = 0.0
    while i < N:
        action, _ = agent.select_action(state["dlo_keypoints"], alpha=0.2)
        action = np.clip(
            action,
            [-0.02, -0.03, -0.2, -0.04, -0.03, -0.2],
            [0.02, 0.03, 0.2, 0.04, 0.03, 0.2],
        )
        action = 0.5 * pre_action + 0.5 * action
        # ================== take an env step ==================#
        action_list[i, :] = action
        next_state, reward, done, truncated, info = env.step(action)
        left_twist, right_twist = env.get_eef_twist()

        env.render(mode="human")

        error = np.linalg.norm(target_shape - state["dlo_keypoints"], 2) / num_feats
        error_list.append(error)

        # ============== update Jacobian ========================
        delta_s = next_state["dlo_keypoints"] - state["dlo_keypoints"]
        delta_x = action * env.dt
        delta_s = delta_s.reshape((3 * num_feats, 1))
        delta_x = delta_x.reshape((6, 1))
        if i % 2 == 0:
            agent.update(delta_s, delta_x)

        # ================== log printing ====================== #
        if i % 20 == 0:
            print(
                f"Timestep: [{i} / {N}] || Shape Error: {error} Shape Error Reduction: {error_pre - error}"
            )
            if error_pre - error < 1e-4 and i > 2000:
                done = True
                break
            error_pre = error
        state = next_state
        pre_action = action
        i += 1
        if done and i > 2000:
            break

    # ======================= print and save result ======================== #
    print(f"residual shape error {error}")

    # ======================= plot ======================== #
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme("poster")
    sns.lineplot(error_list)

    plt.figure(2)
    for j in range(6):
        sns.lineplot(action_list[:, j])
    plt.show()
