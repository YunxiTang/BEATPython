from ur5_env import UR5Env
from ik_controller import IKController
import numpy as np
import mujoco
from pydrake.all import RigidTransform, RollPitchYaw


if __name__ == '__main__':
    ur5_xml = 'ur5/ur5.urdf'
    ur5_scene_xml = 'ur5/scene.xml'
    rbt_model = mujoco.MjModel.from_xml_path(ur5_scene_xml)
    robot = UR5Env(rbt_model)

    ik_controller = IKController(ur5_xml, rbt_name='UR5', ee_name='fake_pinch_link', dt=0.001)

    q_ref = np.array([0, -1.57, -1.57, -1.0, -1.57, 0])

    goal_ee_pos = np.array([0.5, 0.3, 0.45, np.pi, 0.0, 0.0]) # translation + roll-pitch-yaw
    
    robot.reset()
    obs = robot.home()

    q = obs[0:6]
    qdot = obs[6:12]
    t = robot.sim_time
    mujoco.mj_forward(robot.mj_model, robot.mj_data)

    q_hist = []
    qdot_hist = []
    ctrl_hist = []
    x_error_hist = []

    N = int(25 / robot.dt)
    ik_controller.set_goal(goal_ee_pos, 5, q, qdot, t)
    flag = 0
    p = 0.48
    for i in range(N):
        q = obs[0:6]
        qdot = obs[6:12]
        t = robot.sim_time
        mujoco.mj_forward(robot.mj_model, robot.mj_data)

        x_error = robot.mj_data.site('pinch').xpos.ravel() - goal_ee_pos[0:3]
        print('position error: {}'.format(x_error))
        
        dist = robot.mj_data.site('pinch').xpos.ravel()-robot.mj_data.body('cube1').xpos.ravel()
        print(dist[2])
        if dist[2] >= 0.05 and flag == 0:
            ctrl = ik_controller.GetQPControl(q, qdot, t)
            grip = 0
        else:
            flag = 1
            p += 0.00005
            target_ee_pos = RigidTransform(RollPitchYaw(goal_ee_pos[3:6]), np.array([0.5, 0.3, p]))
            ctrl = ik_controller.GetQPControl(q, qdot, t, target_ee_pos)
            grip = 120
        print(p)
        obs = robot.step(ctrl, grip)
        robot.render()

        q_hist.append(q)
        qdot_hist.append(qdot)
        ctrl_hist.append(ctrl)
        x_error_hist.append(x_error)

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