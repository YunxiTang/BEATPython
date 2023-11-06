from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
import numpy as np
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.forwarddiff import jacobian
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve
import mujoco as mj
from pydrake.trajectories import PiecewiseQuaternionSlerp, PiecewisePose
from typing import Dict



def DiffIKQP(plant, 
             context, 
             ee_frame, 
             world_frame, 
             desired_pose, 
             q_init, 
             position_tolerance: float = 1e-8,
             verbose: bool = False):
    plant_context = plant.GetMyContextFromRoot(context)
    rot = desired_pose.rotation()
    trans = desired_pose.translation()
    ik = InverseKinematics(plant, plant_context, with_joint_limits=True)

    ik.AddPointToPointDistanceConstraint(ee_frame, np.array([0,0,0]), world_frame, trans, 0, position_tolerance)
    # orientation constraint
    ik.AddOrientationConstraint(ee_frame, RotationMatrix(), world_frame, rot, 0)


    prog = ik.get_mutable_prog()
    q = ik.q()
    prog.SetInitialGuess(q, q_init)
    result = Solve(prog)
    q_result = result.GetSolution() 
    success_flag = result.is_success()

    if verbose:
      print("\nStates Solution: \n", np.array(q_result))
      print ("\nSuccess Flag: ", success_flag, "\n")

    return np.array(q_result)


def make_gripper_trajectory(X_G: Dict, times: Dict):
    """
        Constructs a gripper position trajectory.
    """
    sample_times = []
    poses = []
    for name in ["initial", "target"]:
        sample_times.append(times[name])
        poses.append(X_G[name])

    return PiecewisePose.MakeCubicLinearWithEndLinearVelocity(sample_times, poses, start_vel=[0.0,0.0,0.0], end_vel=[0.0,0.0,0.0])


class IKController:
    def __init__(self, 
                 rbt_model_file: str, 
                 rbt_name: str, 
                 ee_name: str,
                 dt: float = 0.001, 
                 fixed_base: bool = True, 
                 visualize: bool = False):
        '''
            Inverse Kinematics Controller
        '''
        self.rbt_model_file = rbt_model_file
        self.rbt_name = rbt_name
        self.fixed_base = fixed_base
        self.visualization = visualize
        self.dt = dt

        self._plant = MultibodyPlant(time_step=dt)
        self._parser = Parser(self._plant)
        self.rbt_mdl = self._parser.AddModels(file_name=rbt_model_file) 
        self._plant.Finalize()
        

        self._nq = self._plant.num_positions()
        self._nv = self._plant.num_velocities()
        self._na = self._plant.num_actuators()

        self._vref = np.zeros((self.nv,))

        self._context = self._plant.CreateDefaultContext()  
        self._world_frame = self._plant.world_frame()
        self._base_frame = self._plant.GetFrameByName(name='base_link')
        self._ee_frame = self._plant.GetFrameByName(name=ee_name)

        self._plant_ad = self._plant.ToAutoDiffXd()
        self._context_ad = self._plant_ad.CreateDefaultContext()
        self._world_frame_autodiff = self._plant_ad.world_frame()
    

    def set_goal(self, goal_ee_pos, duration, q, v, t):
        self.goal_XG = goal_ee_pos
        self.duration = duration
        self.UpdateStoredContext(q, v, t)
        self.X_G_init = self._ee_frame.CalcPoseInWorld(self._context)
        self.X_G_target = RigidTransform(RollPitchYaw(self.goal_XG[3:6]), self.goal_XG[0:3])

        times = {"initial": t, "target": t + duration}
        X_Gs = {"initial": self.X_G_init, "target": self.X_G_target}
        self.traj = make_gripper_trajectory(X_Gs, times)


    def GetQPControl(self, q, v, t, target_ee_pos=None):
        '''
            Get control with robot's current state `(q,v,t)`
        '''
        self.UpdateStoredContext(q, v, t)

        if target_ee_pos is None:
            target_ee_pos = self.traj.GetPose(t)
        
        q_target = DiffIKQP(plant=self._plant,
                            context=self._context,
                            ee_frame=self._ee_frame,
                            world_frame=self._world_frame,
                            desired_pose=target_ee_pos,
                            q_init=q,
                            verbose=False) 

        ref_pos = q_target
        ref_vel = np.clip((q_target - q) / 10, -0.5, 0.5)
        res = np.hstack((ref_pos, ref_vel))
        return res

    
    def UpdateStoredContext(self, q, v, t):
        """
            Use the data in the given inputs to update `self._context`.
            called at the beginning of each control loop.
        """
        self._context.SetTime(t)
        self._plant.SetPositions(self._context, q)
        self._plant.SetVelocities(self._context, v)


    def CalcDynamics(self):
        """
        Compute dynamics quantities, M, Cv, tau_g, and S:
            Mv̇ + C(q, v)v = tau_g(q) + S'tau_app
        Assumes that self.context has been set properly. 
        """
        M = self._plant.CalcMassMatrix(self._context)
        Cv = self._plant.CalcBiasTerm(self._context)
        tau_g = self._plant.CalcGravityGeneralizedForces(self._context)
        S = self._plant.MakeActuationMatrix().T

        return M, Cv, tau_g, S
    
    def CalcCoriolisMatrix(self):
        """
        Compute the coriolis matrix C(q,qd) using autodiff.
        
        Assumes that self.context has been set properly.
        """
        q = self._plant.GetPositions(self._context)
        v = self._plant.GetVelocities(self._context)

        def Cv_fcn(v):
            self._plant_ad.SetPositions(self._context_ad, q)
            self._plant_ad.SetVelocities(self._context_ad, v)
            return self._plant_ad.CalcBiasTerm(self._context_ad)

        C = 0.5*jacobian(Cv_fcn, v)
        return C
    
    def CalcFramePositionQuantities(self, frame):
        """
        Compute the position (p), jacobian (J) and 
        Jdot-times-v (Jdv) for the given frame
        
        Assumes that self.context has been set properly. 
        """
        p = self._plant.CalcPointsPositions(self._context,
                                            frame,
                                            np.array([0,0,0]),
                                            self._world_frame)
        J_full = self._plant.CalcJacobianSpatialVelocity(self._context,
                                                          JacobianWrtVariable.kV,
                                                          frame,
                                                          np.array([0,0,0]),
                                                          self._world_frame,
                                                          self._world_frame)
        J_trans = self._plant.CalcJacobianTranslationalVelocity(self._context,
                                                          JacobianWrtVariable.kV,
                                                          frame,
                                                          np.array([0,0,0]),
                                                          self._world_frame,
                                                          self._world_frame)
        Jdv = self._plant.CalcBiasTranslationalAcceleration(self._context,
                                                            JacobianWrtVariable.kV,
                                                            frame,
                                                            np.array([0,0,0]),
                                                            self._world_frame,
                                                            self._world_frame)
        return p, J_full, J_trans, Jdv
    
    @property
    def nq(self):
        return self._nq
    
    @property
    def nv(self):
        return self._nv
    
    @property
    def na(self):
        return self._na
