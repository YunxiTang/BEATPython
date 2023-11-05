from ur5_env import UR5Env
from ik_controller import IKController
import numpy as np
import mujoco

from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix


if __name__ == '__main__':
    ur5_xml = 'ur5/ur5.urdf'
    ur5_scene_xml = 'ur5/scene.xml'
    rbt_model = mujoco.MjModel.from_xml_path(ur5_scene_xml)
    robot = UR5Env(rbt_model)

    ik_controller = IKController(ur5_xml, rbt_name='UR5', ee_name='fake_pinch_link', dt=0.002)

    q_ref = np.array([0, -1.57, -1.57, -1.0, -1.57, 0])

    goal_ee_pos = np.array([ 0.6, 0.2, 1.1, np.pi, 0, np.pi/6]) # translation + roll-pitch-yaw
    
    obs = robot.reset(0)
    mujoco.mj_forward(robot.mj_model, robot.mj_data)
    q_hist = []
    ctrl_hist = []

    N = 50000
    for i in range(N):
        q = obs[0:6]
        qdot = obs[6:12]
        t = robot.sim_time

        goal_ee_pos[0] = 0.5 + 0.1 * np.cos(2 * t)
        goal_ee_pos[1] = 0.3 + 0.1 * np.sin(2 * t)
        goal_ee_pos[2] = 0.8 + 0.1 * np.cos(2 * t)
        goal_ee_pos[5] = np.pi/6 + np.pi / 6 * np.cos(2 * t)

        ctrl = ik_controller.GetQPControl(goal_ee_pos, q, qdot, t)
        mujoco.mj_forward(robot.mj_model, robot.mj_data)

        print('mujoco: ', robot.mj_data.site('pinch').xpos.ravel() - goal_ee_pos[0:3])
        print('mujoco: {}'.format(robot.mj_data.site('pinch').xmat.reshape(3,3)))
        
        q_hist.append(q)
        ctrl_hist.append(ctrl)
        grip = 255 / 2. + 255 / 2. * np.sin(4 * t)
        obs = robot.step(ctrl, grip)
        robot.render()
        print('=================================')

    import matplotlib.pyplot as plt

    plt.figure(1)
    for i in range(6):
        plt.plot(np.linspace(0, N, N)*robot.dt, np.array(q_hist).T[i,:])
    plt.show()
    # plt.figure(2)
    # for i in range(6):
    #     plt.plot(np.linspace(0, N, N)*robot.dt, np.array(ctrl_hist).T[i,:])
    # plt.show()