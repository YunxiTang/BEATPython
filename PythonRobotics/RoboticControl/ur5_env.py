import mujoco
import numpy as np
import os
from pathlib import Path, PureWindowsPath
from base_env import BaseEnv
from mj_utils import InteractiveRender, Renderer


class UR5Env(BaseEnv):

    def __init__(self, mj_model, use_render = True):
        """
            mj_model: mujoco.MjModel.from_xml_path (xml_path)
        """
        super().__init__(mj_model)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.dt = self.mj_model.opt.timestep
        self.dof = self.mj_model.nv

        self.sim_time = self.mj_data.time
        self.renderer = InteractiveRender(self.mj_model)
        self.counts = 0

        self.reset(0)
        # make a foward kinematic propogation
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.state = self.get_obs()
        self.vis_freq = 1. / 60.

    def get_obs(self):
        qpos = self.mj_data.qpos
        qvel = self.mj_data.qvel
        return np.concatenate([qpos[:], qvel[:]]).ravel()

    def set_state(self, qpos, qvel):
        """
            Set the joints position qpos and velocity qvel of the model.
        """
        assert qpos.shape == (self.mj_model.nq,) and qvel.shape == (self.mj_model.nv,)
        self.mj_data.qpos[:] = np.copy(qpos)
        self.mj_data.qvel[:] = np.copy(qvel)
        if self.mj_model.na == 0:
            self.mj_data.act[:] = None
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def step(self, ctrl):
        self.mj_data.ctrl = ctrl
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.state = self.get_obs()
        self.counts = self.counts + 1
        return np.copy(self.state)

    def reset(self, key_frame_num = None):
        if key_frame_num is None:
            mujoco.mj_resetData(self.mj_model, self.mj_data)
        else:
            # reset the state to the key frame
            if key_frame_num >= self.mj_model.nkey:
                print('key_frame_num out of the key frames range (max. {}). Reset to 0 key frame'.format(self.mj_model.nkey-1))
                key_frame_num = 0
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, key_frame_num)

        self.sim_time = self.mj_data.time
        self.counts = 0
        return self.get_obs()

    def render(self):
        if (self.mj_data.time - self.sim_time) > self.vis_freq:
            self.renderer.render(self.mj_data)
            self.sim_time = self.mj_data.time
        else:
            pass
        return None

    def close(self):
        self.renderer.close()
        return None

if __name__ == '__main__':

    import os
    # passive simulation testing

    current_path = os.path.abspath(__file__)
    filename = PureWindowsPath(r'ur5\\scene.xml')
    correct_path = Path(filename)
    ur5_xml = os.path.join( os.path.dirname(current_path), correct_path )

    rbt_model = mujoco.MjModel.from_xml_path(ur5_xml)

    robot = UR5Env(rbt_model)
    robot.reset(0)
    
    for _ in range(5 * 1000):
        if robot.counts > 1000:
            robot.reset(0)
        action = np.zeros((6,))
        robot.step(action)
        robot.render()