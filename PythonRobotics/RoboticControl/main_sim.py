from ur5_env import UR5Env
from ik_controller import IKController
import numpy as np
import mujoco
from pydrake.all import RigidTransform, RollPitchYaw


def task_setting(frames, ts):
    key_frames = {
        'init_frame': frames[0],
        'pregrasp_frame': frames[1],
        'grasp0_frame': frames[2],
        'grasp1_frame': frames[2],
        'target_frame': frames[3]
    }

    grip_frames = {
        'init_frame': 0,
        'pregrasp_frame': 0,
        'grasp0_frame': 0,
        'grasp1_frame': 0.034,
        'target_frame': 0.034
    }

    key_times = {
        'init_frame': ts[0],
        'pregrasp_frame': ts[1],
        'grasp0_frame': ts[2],
        'grasp1_frame': ts[2]+1.0,
        'target_frame': ts[3]
    }
    return key_frames, grip_frames, key_times


class Logger:
    def __init__(self):
        self.q_hist = []
        self.qdot_hist = []
        self.ctrl_hist = []
        self.x_error_hist = []

    def add(self, q, qdot, ctrl, x_err):
        self.q_hist.append(q)
        self.qdot_hist.append(qdot)
        self.ctrl_hist.append(ctrl)
        self.x_error_hist.append(x_err)



if __name__ == '__main__':
    ur5_xml = 'ur5/ur5.urdf'
    ur5_scene_xml = 'ur5/scene.xml'
    rbt_model = mujoco.MjModel.from_xml_path(ur5_scene_xml)
    robot = UR5Env(rbt_model)
    
    obs = robot.home()

    q = obs[0:6]
    qdot = obs[6:12]
    t = robot.sim_time

    logger = Logger()
    ik_controller = IKController(ur5_xml, rbt_name='UR5', ee_name='fake_pinch_link', dt=0.001)

    init_frame = ik_controller.get_ee_pos(q, qdot, t)
    pregrasp_frame = RigidTransform(RollPitchYaw(np.pi, 0.0, np.pi/3), p=np.array([0.4, 0.3, 0.55]))
    grasp_frame = RigidTransform(RollPitchYaw(np.pi, 0.0, np.pi/3), p=np.array([0.4, 0.3, 0.46]))
    target_frame = RigidTransform(RollPitchYaw(np.pi, 0.0, 0.0), p=np.array([0.5, -0.1, 0.6]))

    key_frames, grip_frames, key_times = task_setting([init_frame, pregrasp_frame, grasp_frame, target_frame],
                                                      [t, t+3, t+6, t+9])
    task_traj = ik_controller.set_task(key_frames, key_times, 'linear')
    
    N = int(20 / robot.dt)
    
    for i in range(N):
        q = obs[0:6]
        qdot = obs[6:12]
        t = robot.sim_time

        ctrl = ik_controller.GetQPControl(q, qdot, t)
        grip = grip_frames['grasp0_frame'] if t <= key_times['grasp0_frame']+0.5 else grip_frames['grasp1_frame']
        if t > key_times['target_frame']+2:
            grip = 0
        obs = robot.step(ctrl, grip)

        if i % 25 == 0:
            robot.render()

        x_error = robot.mj_data.site('pinch').xpos.ravel() - task_traj.GetPose(t).translation()
        logger.add(q, qdot, ctrl, x_error)


    import matplotlib.pyplot as plt

    plt.figure(1)
    for i in range(6):
        plt.plot(np.linspace(0, N, N)*robot.dt, np.array(logger.q_hist).T[i,:])

    plt.figure(2)
    for i in range(6):
        plt.plot(np.linspace(0, N, N)*robot.dt, np.array(logger.qdot_hist).T[i,:])

    plt.figure(3)
    for i in range(3):
        plt.plot(np.linspace(0, N, N)*robot.dt, np.array(logger.x_error_hist).T[i,:])
    plt.show()