""" Rope Reaching Task Env with Dual Spatial Hand"""
import mujoco
import numpy as np
import os
from gym import utils
from gym.spaces import Box
import seaborn as sns
from st_dlo_planning.envs.mujoco_base_env import MujocoEnv
from pathlib import Path, PureWindowsPath


TaskSceneNames = {
    '03': {'file': Path(PureWindowsPath(r'assets\\dual_hand_thin_03.xml')), 'dlo_len':0.3, 'cable_geom_num': 39},
    '04': {'file': Path(PureWindowsPath(r'assets\\dual_hand_thin_04.xml')), 'dlo_len':0.4, 'cable_geom_num': 49},
    '05': {'file': Path(PureWindowsPath(r'assets\\dual_hand_thin_05.xml')), 'dlo_len':0.5, 'cable_geom_num': 39},
    '06': {'file': Path(PureWindowsPath(r'assets\\dual_hand_thin_06.xml')), 'dlo_len':0.6, 'cable_geom_num': 39}
}

TaskName = [ele[0] for ele in TaskSceneNames.items()]

current_path = os.path.abspath(__file__)

GRIPPER_NAMES = ['left_hand', 'right_hand']


class DualGripperCableEnv(MujocoEnv, utils.EzPickle):
    """
    ### Description 
    This quasistaic environment aims to manipulate a cable/rope into the goal region/shape.
    The DualGripperRopeEnv consists of two grippers and one rope.
    
    ### Action Space
    ### Action Space
    An action represents the x-y-rz velocities of grippers.
    | Num | Action        | Control_Min | Control_Max |
    |-----|---------------|-------------|-------------|
    | 0   | left-vx       | -0.3        | 0.3         |
    | 1   | left-vy       | -0.3        | 0.3         |
    | 2   | left-vz       | -0.0        | 0.0         |
    | 3   | left-rx       | -0.0        | 0.0         |
    | 4   | left-ry       | -0.0        | 0.0         |
    | 5   | left-rz       | -0.3        | 0.3         |
    | 6   | right-vx      | -0.3        | 0.3         |
    | 7   | right-vy      | -0.3        | 0.3         |
    | 8   | right-vz      | -0.0        | 0.0         |
    | 9   | right-rx      | -0.0        | 0.0         |
    | 10  | right-ry      | -0.0        | 0.0         |
    | 11  | right-rz      | -0.3        | 0.3         |
    
    ### Observation Space
    Observations consist of feature points' positions onf the
    rope and gripper's position.
    
    ### Rewards Design
    None. (Not for RL)
    
    ### Starting State
    The starting state is the default configuration as descriped in the `xml` file.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, task:str, feat_stride:int, **kargs):
        if task in TaskSceneNames:
            XML_FILE = os.path.join( os.path.dirname(current_path), TaskSceneNames[task]['file'] )
        else:
            raise NotImplementedError(f"Wrong Task Name {task}. Not In {TaskName }")
        
        self.frame_skip = 50
        if task == '03':
            self.frame_skip = 250

        self.model_path = XML_FILE

        self.dlo_len = TaskSceneNames[task]['dlo_len']
        self.cable_num_geom = TaskSceneNames[task]['cable_geom_num']

        # avoid collosion and over-stretching
        self.min_distance = 0.08 
        self.max_distance = self.dlo_len + 0.03
        
        self.cable_geom_names = [f'G{idx}' for idx in range(self.cable_num_geom)]

        # define observation and action space
        self.num_grasp = len(GRIPPER_NAMES)
        self.feat_idx = list( range(1, self.cable_num_geom, feat_stride) )
        self.num_feat = len(self.feat_idx)

        self.observation_space = Box(low=-np.inf, 
                                     high=np.inf, 
                                     shape=((self.num_feat + self.num_grasp) * 3,), 
                                     dtype=np.float64)
        self.action_low = -0.5
        self.action_high = 0.5
        self.action_space = Box(low=self.action_low, 
                                high=self.action_high, 
                                shape=(6*self.num_grasp,), 
                                dtype=np.float64)
        
        super(DualGripperCableEnv, self).__init__(
            self.model_path, 
            self.frame_skip, 
            self.observation_space,
            self.action_space,
            **kargs
        )
        utils.EzPickle.__init__(self)

        # reset the feature point color
        material_id = self.model.mat('cable_material').id
        
        for geom_name in self.cable_geom_names:
            geom_id = self.model.geom(geom_name).id
            self.model.geom_matid[geom_id] = material_id
            self.model.geom(geom_name).rgba = None

        # get the actuated joint idxs
        self.left_hand_jnt_idxs = [self._get_joint_index(joint_name) for joint_name in ['left_px', 'left_py', 'left_rz']]
        self.right_hand_jnt_idxs = [self._get_joint_index(joint_name) for joint_name in ['right_px', 'right_py', 'right_rz']]

        self.actuation_matrix = np.array([[1, 0, 0],
                                          [0, 1, 0],
                                          [0, 0, 1],], dtype=np.float32)

    def _get_obs(self):
        """
            get the env state/observation
        """
        # position of feature sites and  transforms of the two grippers
        # position of cable geoms
        geom_pos = [self.data.geom(geom_name).xpos.copy().ravel() for geom_name in self.cable_geom_names]
        self.geom_positions = np.concatenate(geom_pos)

        feat_pos = [geom_pos[idx] for idx in self.feat_idx]
        feat_positions = np.concatenate(feat_pos)

        gripper_transform = []
        for gripper_name in GRIPPER_NAMES:
            g_pos = self.data.body(gripper_name).xpos.copy().ravel()
            g_quat = self.data.body(gripper_name).xquat.copy().ravel()
            g_transform = np.hstack((g_pos, g_quat))
            gripper_transform.append(g_transform)
        
        gripper_transforms = np.concatenate(gripper_transform)
        
        obs = {'dlo_keypoints': feat_positions,
               'eef_transforms': gripper_transforms}
        return obs
    

    def compute_reward(self, obs, action):
        return 0


    def terminted(self, obs):
        """termination condition design"""
        done = False
        # gripper distance
        lpos = self.data.body('left_hand').xpos
        rpos = self.data.body('right_hand').xpos
        gripper_distance = np.sqrt( np.sum( (lpos - rpos) ** 2) )

        # minimum distance or super-extension of rope
        if gripper_distance <= self.min_distance or gripper_distance >= self.max_distance:
            done = True
            # print('overstretching')
        return done
    
    def _get_joint_index(self, joint_name):
        # get the joint id
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        joint_idx = self.model.jnt_dofadr[joint_id]
        return joint_idx

    def _compute_ee_jac(self, body_name:str):
        '''
            compute ee jacobian for body with name
        '''
        jacp_full = np.zeros((3, self.model.nv))
        jacr_full = np.zeros((3, self.model.nv))
        ee_body = self.model.body(body_name).id
        mujoco.mj_jacBody(self.model, self.data, jacp_full, jacr_full, ee_body)
        return jacp_full, jacr_full
    
    def get_eef_twist(self):
        left_eef_bodyId = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_hand')
        right_eef_bodyId = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_hand')
        
        left_eef_twsit = np.zeros(6,)
        right_eef_twist = np.zeros(6,)

        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, left_eef_bodyId, left_eef_twsit, 0)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, right_eef_bodyId, right_eef_twist, 0)

        return left_eef_twsit, right_eef_twist
    
    
    def step_relative_ee(self, delta_left_ee_pos, delta_right_ee_pos, render=False):
        '''
            move ee incrementally (the relative ee pos shoul be given in the world frame, not the ee frame)
        '''
        obs = self._get_obs()
        gripper_pos = obs[3*self.num_feat:3*(self.num_feat+self.num_grasp)]
        L_pos = gripper_pos[0:3]
        R_pos = gripper_pos[3:6]

        L_pos_0 = L_pos 
        R_pos_0 = R_pos 

        L_target = L_pos + delta_left_ee_pos
        R_target = R_pos + delta_right_ee_pos

        for i in range(100):
            left_jacp_full, left_jacr_full = self._compute_ee_jac('left_hand')
            right_jacp_full, right_jacr_full = self._compute_ee_jac('right_hand')

            left_jacp = left_jacp_full[:, self.left_hand_jnt_idxs]
            right_jacp = right_jacp_full[:, self.right_hand_jnt_idxs]

            delta_left_ee_pos = L_target - L_pos
            delta_right_ee_pos = R_target - R_pos

            delta_left_act = np.linalg.inv(left_jacp) @ delta_left_ee_pos
            delta_right_act = np.linalg.inv(right_jacp) @ delta_right_ee_pos
            
            full_act = np.concatenate((delta_left_act, np.zeros(3,), delta_right_act, np.zeros(3,))) * 1.0
            full_act = np.clip(full_act, -0.15, 0.15)
            
            obs, reward, done, info, truncated = self.step(full_act)
            if render:
                self.render('human')

            gripper_pos = obs[3*self.num_feat:3*(self.num_feat+self.num_grasp)]
            L_pos = gripper_pos[0:3]
            R_pos = gripper_pos[3:6]
            error = (np.linalg.norm(L_target-L_pos) + np.linalg.norm(R_target-R_pos))
            if error < 1e-2 and i > 1:
                break
        print(L_target-L_pos_0, R_target-R_pos_0)
        return obs, reward, done, info, truncated


    def step(self, action):
        """
            Action: in shape of (na, ). The desired twists (in the world frame) for the EEFs,
            with Inverse Kinematics (IK) Controller inside

                    left: [vx, vy, vz, wx, wy, wz]
                    right: [vx, vy, vz, wx, wy, wz]
        """
        # compute the jacobian firstly
        left_jacp_full, left_jacr_full = self._compute_ee_jac('left_hand')
        right_jacp_full, right_jacr_full = self._compute_ee_jac('right_hand')

        left_jacp = left_jacp_full[:, self.left_hand_jnt_idxs]
        left_jacr = left_jacr_full[:, self.left_hand_jnt_idxs]
        left_jac = np.concatenate([left_jacp, left_jacr], axis=0)
        
        right_jacp = right_jacp_full[:, self.right_hand_jnt_idxs]
        right_jacr = right_jacr_full[:, self.right_hand_jnt_idxs]
        right_jac = np.concatenate([right_jacp, right_jacr], axis=0)

        left_act_command = action[0:3]
        right_act_command = action[3:]
        # left_act_command = np.linalg.pinv(left_jac) @ action[0:6]
        # right_act_command = np.linalg.pinv(right_jac) @ action[6:]

        act_command = np.concatenate([left_act_command, right_act_command], axis=0)

        self.do_simulation(act_command, self.frame_skip)

        obs = self._get_obs()
        done = self.terminted(obs)
        reward = 1. if done else 0.
        info = {}
        truncated = False
        return obs, reward, done, info, truncated

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel). 
        """
        qpos_init = self.init_qpos
        qvel_init = self.init_qvel
        self.set_state(qpos_init, qvel_init)
        obs = self._get_obs()
        return obs


    def render(self, mode: str = 'human', camera_name = None):
        # self.viewer.vopt.frame = mujoco.mjtFrame.mjFRAME_BODY
        rendered_img = self._render(mode=mode, camera_name=camera_name)
        return rendered_img
    
    @staticmethod
    def plot_rope(rope_vec, ax, clr, lns=None, ldw=2.5):
        n, _ = rope_vec.shape
        for i in range(n-1):
            sns.scatterplot(x=[rope_vec[i,0], rope_vec[i+1,0]], y=[rope_vec[i,1], rope_vec[i+1,1]], ax=ax, color=clr, s=20)
            ax.plot([rope_vec[i,0], rope_vec[i+1,0]], [rope_vec[i,1], rope_vec[i+1,1]], color=clr, linestyle=lns, linewidth=ldw)
        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(-0.25, 0.25)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        return None
