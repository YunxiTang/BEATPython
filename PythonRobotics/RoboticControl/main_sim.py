from ur5_env import UR5Env
from ik_controller import IKController
import numpy as np
import mujoco


if __name__ == '__main__':
    ur5_xml = 'ur5/ur5.urdf'
    ur5_scene_xml = 'ur5/scene.xml'
    rbt_model = mujoco.MjModel.from_xml_path(ur5_scene_xml)
    robot = UR5Env(rbt_model)

    ik_controller = IKController(ur5_xml, rbt_name='UR5', ee_name='fake_pinch_link', dt=0.001)

    q_ref = np.array([0, -1.57, -1.57, -1.0, -1.57, 0])

    goal_ee_pos = np.array([0.4, 0.4, 0.75, np.pi, 0 , np.pi/3]) # translation + roll-pitch-yaw
    
    obs = robot.reset(0)
    q = obs[0:6]
    qdot = obs[6:12]
    t = robot.sim_time
    mujoco.mj_forward(robot.mj_model, robot.mj_data)

    q_hist = []
    qdot_hist = []
    ctrl_hist = []
    x_error_hist = []

    N = int(12 / robot.dt)
    ik_controller.set_goal(goal_ee_pos, 10, q, qdot, t)

    for i in range(N):
        q = obs[0:6]
        qdot = obs[6:12]
        t = robot.sim_time

        ctrl = ik_controller.GetQPControl(q, qdot, t)
        mujoco.mj_forward(robot.mj_model, robot.mj_data)
        x_error = robot.mj_data.site('pinch').xpos.ravel() - goal_ee_pos[0:3]
        print('position error: {}'.format(x_error))
        
        q_hist.append(q)
        qdot_hist.append(qdot)
        ctrl_hist.append(ctrl)
        x_error_hist.append(x_error)

        grip = 255 / 2. + 255 / 2. * np.sin(4 * t)
        obs = robot.step(ctrl, grip)
        robot.render()

    import matplotlib.pyplot as plt

    plt.figure(1)
    for i in range(6):
        plt.plot(np.linspace(0, N, N)*robot.dt, np.array(q_hist).T[i,:])

    plt.figure(2)
    for i in range(6):
        plt.plot(np.linspace(0, N, N)*robot.dt, np.array(qdot_hist).T[i,:])

    plt.figure(3)
    for i in range(3):
        plt.plot(np.linspace(0, N, N)*robot.dt, np.array(x_error_hist).T[i,:])
    plt.show()