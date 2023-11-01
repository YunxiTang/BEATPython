from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
import numpy as np
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.forwarddiff import jacobian


class IKController:
    def __init__(self, rbt_model_file: str, rbt_name: str, dt: float = 0.001, fixed_base: bool = True, visualize: bool = False):
        '''
            Inverse kinematics controller
        '''
        self.rbt_model_file = rbt_model_file
        self.rbt_name = rbt_name
        self.fixed_base = fixed_base
        self.visualization = visualize
        self.dt = dt

        self._plant = MultibodyPlant(time_step=dt)
        self._parser = Parser(self._plant)
        self.rbt_mdl, = self._parser.AddModels(file_name=rbt_model_file) 
        self._plant.Finalize()

        self._nq = self._plant.num_positions()
        self._nv = self._plant.num_velocities()
        self._na = self._plant.num_actuators()

        
        self._context = self._plant.CreateDefaultContext()  
        self._world_frame = self._plant.world_frame()

        self._plant_ad = self._plant.ToAutoDiffXd()
        self._context_ad = self._plant_ad.CreateDefaultContext()
        self._world_frame_autodiff = self._plant_ad.world_frame()


    def set_pd(self, Kp, Kd):
        self._kp = Kp
        self._kd = Kd


    def set_ref_pos(self, q_ref):
        self._ref_pos = q_ref


    def GetControlTorque(self, q, v, t):
        '''
            Get control torque with robot's current state `(q,v,t)`
        '''
        self.UpdateStoredContext(q, v, t)
        q = self._plant.GetPositions(self._context)
        v = self._plant.GetVelocities(self._context)
        # Compute control torques
        res = self.ControlLaw(q, v, t)
        return res


    def _ik_solver(self):
        pass


    def ControlLaw(self, q, v, t):
        """
            main control law for the robot. 
        """
        ref_pos = self._ref_pos.copy()
        ref_pos[3] = self._ref_pos[3] + 0.5 * np.sin(2. * t)
        ref_vel = np.zeros((6,))
        ref_vel[3] = 1.0 * np.cos(2. * t)
        ref_acc = np.zeros((6,))
        ref_acc[3] = -2.0 * np.sin(2. * t)

        M, Cv, tau_grav, _ = self.CalcDynamics()
        u_ff = -tau_grav + Cv
        u_fb = M @ ( ref_acc + self._kp @ (ref_pos - q) + self._kd @ (ref_vel - v) )
        u = u_ff + u_fb
        u = np.clip(u, -100, 100)
        return u
    
    def UpdateStoredContext(self, q, v, t):
        """
            Use the data in the given inputs to update `self._context`.
            This should be called at the beginning of each control loop.
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
        J = self._plant.CalcJacobianTranslationalVelocity(self._context,
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
        return p, J, Jdv
    
    @property
    def nq(self):
        return self._nq
    
    @property
    def nv(self):
        return self._nv
    
    @property
    def na(self):
        return self._na
