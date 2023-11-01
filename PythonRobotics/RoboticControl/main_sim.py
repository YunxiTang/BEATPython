from ur5_env import UR5Env
from ik_controller import IKController
import numpy as np
import mujoco

if __name__ == '__main__':
    ur5_xml = 'ur5/ur5_control.xml'
    ur5_scene_xml = 'ur5/scene.xml'
    rbt_model = mujoco.MjModel.from_xml_path(ur5_scene_xml)
    robot = UR5Env(rbt_model)

    ik_controller = IKController(ur5_xml, rbt_name='UR5')

    Kp = np.diag([250., 250., 250., 80., 50., 50]) * 1.5
    Kd = np.diag([100., 100., 100., 40., 30., 20]) * 1
    q_ref = np.array([0, -1.57, -1.57, -1.57, -1.57, 0])

    ik_controller.set_pd(Kp, Kd)
    ik_controller.set_ref_pos(q_ref)

    robot.reset(0)

    q_hist = []
    ctrl_hist = []
    N = 25000
    for i in range(N):
        q = robot.state[0:6]
        qdot = robot.state[6:12]
        t = robot.sim_time

        ctrl = ik_controller.GetControlTorque(q, qdot, t)

        q_hist.append(q)
        ctrl_hist.append(ctrl)

        robot.step(ctrl)
        robot.render()
    
    import matplotlib.pyplot as plt

    plt.figure(1)
    # for i in range(6):
    plt.plot(np.linspace(0, N, N)*robot.dt, np.array(q_hist).T[3,:])

    plt.figure(2)
    # for i in range(6):
    plt.plot(np.linspace(0, N, N)*robot.dt, np.array(ctrl_hist).T[3,:])
    plt.show()