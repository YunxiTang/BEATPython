from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
import numpy as np
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.forwarddiff import jacobian
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import MathematicalProgram, SnoptSolver
import mujoco as mj


def DiffIKPseudoInverse(J_G, V_G_desired, q_now=None, v_now=None, X_now=None):
    m, n = J_G.shape
    v = np.linalg.pinv(J_G + 1e-4 * np.eye(m, n)).dot(V_G_desired)
    return v


def DiffIKQP(J_G, V_G_desired, q_now, v_now, X_now):
    prog = MathematicalProgram()
    v = prog.NewContinuousVariables(7, "v")
    v_max = 3.0  # do not modify

    # Add cost and constraints to prog here.

    solver = SnoptSolver()
    result = solver.Solve(prog)

    if not (result.is_success()):
        raise ValueError("Could not find the optimal solution.")

    v_solution = result.GetSolution(v)

    return v_solution


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
        self.rbt_mdl = self._parser.AddModelFromFile(file_name=rbt_model_file) 
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

    def GetControl(self, goal_ee_pos, q, v, t):
        '''
            Get control with robot's current state `(q,v,t)`
        '''
        
        self.UpdateStoredContext(q, v, t)
        p_now, J_full, J_trans, _ = self.CalcFramePositionQuantities(self._ee_frame)
        
        R_now = self._ee_frame.CalcRotationMatrixInWorld(self._context)
        quat_now = R_now.ToQuaternion().wxyz()
        quat_diff = np.zeros((3,))
        mj.mju_subQuat(quat_diff, goal_ee_pos[0:4], quat_now)

        V_G_desired = np.concatenate([quat_diff.ravel(), (goal_ee_pos[4:] - p_now).ravel()]).reshape(-1, 1)

        ref_vel = DiffIKPseudoInverse(J_full, V_G_desired)
        scale = np.diag([3, 3, 3, 1, 1, 1]) / 10
        delta = np.clip(scale @ ref_vel.ravel(), -.1, .1 )
        ref_pos = q + delta

        res = np.hstack((ref_pos, self._vref))
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
