from ur5_env import UR5Env
from ik_controller import IKController
import numpy as np
import mujoco
from pydrake.all import RigidTransform, RollPitchYaw, Quaternion
import time


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
    env = UR5Env(rbt_model)
    
    obs = env.home()

    # get box pos and orientation
    box_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    box_xpos = env.mj_data.geom_xpos[box_id]
    box_xmat = env.mj_data.geom_xmat[box_id]
    box_quat = np.zeros(4,)
    mujoco.mju_mat2Quat(box_quat, box_xmat)

    ee_quat = RollPitchYaw(np.pi, 0.0, np.pi / 3)
    ee_pos = box_xpos + np.array([0., 0., 0.07])
    target_ee_pos = RigidTransform(ee_quat, p=ee_pos)

    q = obs[0:6]
    qdot = obs[6:12]
    t = env.sim_time

    logger = Logger()
    ik_controller = IKController(ur5_xml, rbt_name='UR5', ee_name='fake_pinch_link', dt=0.001)
    
    N = int(3 / env.dt)
    
    count = 0
    for i in range(N):
        q = obs[0:6]
        qdot = obs[6:12]
        t = env.sim_time
        
        ctrl = ik_controller.GetQPControl(q, qdot, t, target_ee_pos)
        x_error = np.linalg.norm(env.mj_data.site('pinch').xpos.ravel() - box_xpos)
       
        if x_error <= 0.05:
            grasp = 1.0
            
        else:
            grasp = 0
            
        obs = env.step(ctrl, grasp)

        env.render()
        time.sleep(1 / 60.)
        logger.add(q, qdot, ctrl, x_error)
        count += 1

    import matplotlib.pyplot as plt

    plt.figure(1)
    for i in range(6):
        plt.plot(np.linspace(0, count, count)*env.dt, np.array(logger.q_hist).T[i,:])

    plt.figure(2)
    for i in range(6):
        plt.plot(np.linspace(0, count, count)*env.dt, np.array(logger.qdot_hist).T[i,:])

    plt.figure(3)
    for i in range(3):
        plt.plot(np.linspace(0, count, count)*env.dt, np.array(logger.x_error_hist).T[:])
    plt.show()