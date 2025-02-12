from typing_extensions import NamedTuple, Dict, Optional
import numpy as np


class RobotObs(NamedTuple):
    """
        For each robot observation:
        - observations
            - rgb_image
                - cam_name_0        (H, W, 3)     'uint8'
                - ...               (H, W, 3)     'uint8'
                - cam_name_n        (H, W, 3)     'uint8'
            - depth_image
                - cam_name_0        (H, W, 3)     'uint8'
                - ...               (H, W, 3)     'uint8'
                - cam_name_n        (H, W, 3)     'uint8'
            - qpos                  (nq,)         'float64'
            - qvel                  (nq,)         'float64'
            - eef_pos               (7,)          'float64'
            - eef_vel               (6,)          'float64'
        - action                      (na,)         'float64'
        - timestamp                   (1,)          'float64'
    """
    rgb_image: Dict[str, np.ndarray]
    depth_image: Dict[str, np.ndarray]
    qpos: np.ndarray
    qvel: np.ndarray
    eef_pos: np.ndarray
    eef_vel: np.ndarray
    action: np.ndarray
    timestamp: float


if __name__ == '__main__':
    obs = RobotObs(
        {'camera_wrist': np.zeros((2, 2, 3), dtype=np.uint8),},
        {'camera_wrist': np.zeros((2, 2), dtype=np.uint8),},
        np.zeros((5,)),
        np.zeros((5,)),
        np.zeros((7,)),
        np.zeros((6,)),
        np.zeros((3,)),
        1.25
    )
    obs_dict = obs._asdict()
    for key, val in obs_dict.items():
        print(key)
        print(val)