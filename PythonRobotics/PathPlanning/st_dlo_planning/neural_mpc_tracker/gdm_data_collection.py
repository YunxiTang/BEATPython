"""Data Collection Script"""

import os
import st_dlo_planning.utils.misc_utils as misu
import st_dlo_planning.utils.pytorch_utils as ptu
import numpy as np
from st_dlo_planning.envs.planar_cable_deform.planar_cable_deform import (
    DualGripperCableEnv,
    wrap_angle,
)
from st_dlo_planning.utils.misc_utils import VideoLoggerPro, ZarrLogger
from scipy.spatial.transform import Rotation as sciR


def CollectOfflineData(
    env: DualGripperCableEnv,
    task: str,
    data_logger: ZarrLogger,
    seed: int,
    num_frame: int = 50000,
    episode_len: int = 200,
    mode: str = "random",
    render_mode: str = "human",
    render: bool = True,
):
    DataCollectMode = ["random", "teleop", "expert_policy"]
    if mode not in DataCollectMode:
        raise ValueError(f"mode {mode} is not supported.")

    dlo_len = env.dlo_len

    if render_mode != "human":
        video_logger = VideoLoggerPro(
            f"/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/{task}_dlo_collection.mp4",
            fps=1.0 / env.dt,
        )

    if mode == "random":
        center = np.array([dlo_len / 2.0, 0.0, 0.5])
        total_frame = 0

        ep_num = 0

        state, _ = env.reset()

        # straight DLO state
        L_pose_straight, R_pose_straight = env.get_lowdim_eef_state(state)

        while True:
            L_pose_init, R_pose_init = env.get_lowdim_eef_state(state)

            # target generation
            L_target = np.random.uniform(low=-1.0, high=1.0, size=(3,))
            R_target = np.random.uniform(low=-1.0, high=1.0, size=(3,))
            L_norm = (L_target[0] ** 2 + L_target[1] ** 2 + L_target[2] ** 2) ** (0.5)
            R_norm = (R_target[0] ** 2 + R_target[1] ** 2 + R_target[2] ** 2) ** (0.5)
            L_target[0] = -L_target[0] if L_target[0] > 0 else L_target[0]
            R_target[0] = -R_target[0] if R_target[0] < 0 else R_target[0]

            r = np.random.uniform(low=0.05, high=dlo_len / 2, size=(2,))

            theta_ = np.random.uniform(-np.pi / 2, np.pi / 2)
            Rot = sciR.from_euler(seq="zyx", angles=[theta_, 0, 0]).as_matrix()

            L_target = Rot @ (L_target / L_norm * r[0]) + center
            R_target = Rot @ (R_target / R_norm * r[1]) + center

            thetas = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(2,))

            left_target = np.array(
                [L_target[0], L_target[1], wrap_angle(thetas[0] + L_pose_init[2])]
            )
            right_target = np.array(
                [R_target[0], R_target[1], wrap_angle(thetas[1] + R_pose_init[2])]
            )

            if ep_num % 2 == 1:
                left_target = np.array(
                    [
                        L_pose_straight[0] - 0.0025,
                        L_pose_straight[1],
                        wrap_angle(L_pose_straight[2]),
                    ]
                )
                right_target = np.array(
                    [
                        R_pose_straight[0] + 0.0025,
                        R_pose_straight[1],
                        wrap_angle(R_pose_straight[2]),
                    ]
                )

            print("Targets: ", left_target, right_target)

            left_delta = left_target - L_pose_init
            right_delta = right_target - R_pose_init

            states, actions, next_states, imgs, traj_len = env.step_relative_eef(
                left_delta,
                right_delta,
                num_steps=200,
                return_traj=True,
                render_mode="human",
            )

            for k in range(traj_len):
                data_logger.log_data("dlo_keypoints", states[k]["dlo_keypoints"])
                data_logger.log_data("eef_states", states[k]["lowdim_eef_transforms"])
                data_logger.log_data("eef_transforms", states[k]["eef_transforms"])

                data_logger.log_data(
                    "next_dlo_keypoints", next_states[k]["dlo_keypoints"]
                )
                data_logger.log_data(
                    "next_eef_states", next_states[k]["lowdim_eef_transforms"]
                )
                data_logger.log_data(
                    "next_eef_transforms", next_states[k]["eef_transforms"]
                )
                data_logger.log_data("action", actions[k])
                data_logger.log_data("ep_num", ep_num)

                data_logger.log_meta("dlo_len", env.dlo_len)

            total_frame += traj_len
            ep_num += 1

            state = next_states[-1]
            L_pose, R_pose = env.get_lowdim_eef_state(obs=state)

            print("Final:   ", L_pose, R_pose)
            print(
                f"============== frame: {total_frame} with {ep_num} =================="
            )

            if total_frame >= num_frame:
                break

    else:
        raise NotImplementedError("Not implmented now.")

    if render_mode != "human":
        video_logger.create_video()
