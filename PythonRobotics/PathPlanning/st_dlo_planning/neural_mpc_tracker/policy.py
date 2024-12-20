import torch
import torch.nn as nn
import torch.optim as optim
from st_dlo_planning.neural_mpc_tracker.mpc_solver import GradientMPCSolver
from typing import Callable
import numpy as np
import dill
import st_dlo_planning.utils.pytorch_utils as ptu

import jax.numpy as jnp
from jax import jit, vmap, grad
from qpsolvers import solve_qp

import st_dlo_planning.utils.jax_utils as jau
from st_dlo_planning.utils.world_map import WorldMap
from typing import List


class GradientLMPCAgent:
    def __init__(self, 
                 nx:int, 
                 nu:int,
                 dlo_length:float, 
                 dt:float,
                 path_cost_func:Callable,
                 final_cost_func:Callable,
                 pretrained_model,
                 device,
                 umax, umin,
                 mdl_ft:bool=False,
                 log_prefix:str=None,
                 discount_factor:float=0.8,
                 traj_horizon:int=10,
                 replan_freq:int=1) -> None:
        """
        Learnined-based MPC Agent

        Args:
            num_feats (int): number of feature points
            num_grasps (int): number of grasp points
            dt (float): timestep
            
            path_cost_func (Callable): path cost function
            final_cost_func (Callable): final cost function
            device (torch.device): device (cpu or gpu)
            traj_horizon (int, optional): trajectory horizon of MPC. Defaults to ``10``.
            replan_freq (int, optional): re-plan frequency for MPC. Defaults to ``1``.
        """
        self.nx = nx
        self.nu = nu
        
        self.dim_q = self.nx + self.nu

        self.device = device
        self.dt_np = dt
        self.dt = ptu.from_numpy(np.array([dt]), self.device)
        
        self.dlo_length = dlo_length

        self.mdl_ft = mdl_ft

        # self.logger = DataLogger(
        #     log_dir='/home/yxtang/CodeBase/DOBERT/outputs/tb/sc/'+ f'{log_prefix}_' + time.strftime("%d-%m-%Y_%H-%M-%S"),
        #     log_tb=True,
        #     log_wandb=False
        # )
        
        # For model adaptation
        self.lr = 1e-3
        self.loss_func = nn.MSELoss()
        
        self.gdm_model = pretrained_model.to(self.device)
        # freeze the network
        for param in self.gdm_model.parameters():
            param.requires_grad = mdl_ft

        # unfreeze the decoder
        for param in self.gdm_model._decoder.parameters():
            param.requires_grad = True
            
        if hasattr( self.gdm_model, '_cond_encoder'):
            for param in self.gdm_model._cond_encoder.parameters():
                param.requires_grad = True
        
        
        self.optimizer = optim.AdamW( filter(lambda p: p.requires_grad, self.gdm_model.parameters()),
                                      lr=self.lr,
                                      weight_decay=1e-5 )
        
        # MPC settings
        self.traj_horizon = traj_horizon
        self.replan_freq = replan_freq
        self.discount_factor = discount_factor
        
        self.mpc_planner = GradientMPCSolver(
            system_model=self.gdm_model,
            path_cost_func=path_cost_func,
            final_cost_func=final_cost_func,
            dlo_len=dlo_length,
            nx=self.nx,
            nu=self.nu,
            dt=dt,
            u_max=umax,
            u_min=umin,
            device=device,
            discount_factor=discount_factor,
            horizon=traj_horizon,
            tol=1e-4,
            lr=0.05,
            max_iter=20
        )
        
        self.mpc_pointer = 0
        self.plan_count = 0
        
        self.inner_n = 0
        
        self.Uref = np.zeros(self.nu*(traj_horizon-1),)

    
    def save_model(self, model_path):
        torch.save(self.gdm_model, model_path, pickle_module=dill)

    
    def update(self, transition_dict):
        """
            Update gdm via gradient descent
        """
        self.gdm_model.train()
        # to allow model update
        for param in self.gdm_model.parameters():
            param.requires_grad = self.mdl_ft
        for param in self.gdm_model._decoder.parameters():
            param.requires_grad = True
        if hasattr( self.gdm_model, '_cond_encoder'):
            for param in self.gdm_model._cond_encoder.parameters():
                param.requires_grad = True

        states_tensors = ptu.from_numpy(transition_dict['states'], self.device)
        next_states_tensors = ptu.from_numpy(transition_dict['next_states'], self.device)
        actions_tensors = ptu.from_numpy(transition_dict['actions'], self.device)

        feat_pos = states_tensors[:, 0:self.num_feats*3]
        next_feat_pos = next_states_tensors[:, 0:self.num_feats*3]

        delta_feat_pos = next_feat_pos - feat_pos
        delta_eef_pos = next_states_tensors[:, self.num_feats*3:] - states_tensors[:, self.num_feats*3:]

        dlo_len = torch.tensor(self.dlo_length, device=self.device)[None]

        predicts = self.gdm_model(states_tensors, delta_eef_pos, dlo_len)
        loss = self.loss_func(predicts, delta_feat_pos)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_np = ptu.to_numpy(loss)

        self.inner_n += 1
        
        return loss_np
        
    
    def select_action(self, state:np.ndarray, xref: np.ndarray):
        """
            MPC Planner
        """
        self.gdm_model.eval() 
        for p in self.gdm_model.parameters():
            p.requires_grad = False

        self.Uref, obj_loss = self.mpc_planner.plan(state, xref)
        u_opt = self.Uref
        return u_opt, obj_loss
    
    def model_adapt(self, transition_dict):
        """
            model adaptation
        """
        loss_np = self.update(transition_dict)
        # self.logger.log('model_adapt_loss', loss_np, self.inner_n)
        return loss_np
        
    @torch.no_grad()
    def evaluate(self, state)->np.ndarray:
        action, _ = self.select_action(state)
        if isinstance(action, np.ndarray):
            return action
        else:
            return ptu.to_numpy(action).flatten()
        

class BroydenAgent:
    def __init__(self, input_dim, output_dim):
        """
            Shape Controller based on Broyden's rule
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.jacobian_model = np.ones((output_dim, input_dim)) * 1.0
        self.inner_n = 0
        self.target_dlo_shape = None
        self.weight = np.eye(output_dim)


    def set_target_q(self, target_q):
        """
            set target shape
        """
        self.target_dlo_shape = target_q

    def clamp(self, action):

        return action
    
    def update_weight(self, new_weight):
        """update the weight matrix for inverse kinematic controller

        Args:
            new_weight (_type_): new weight
        """
        self.weight = new_weight

    def select_action(self, state, alpha=1.0):
        """
            traditional inverse kinematics controller
        """
        dlo_shape = state
        # # compute Jacobian
        # U, S, Vh = np.linalg.svd( self.jacobian_model )
        # min_s = np.min(S)
        jac = self.jacobian_model + 0.2 * np.eye(self.output_dim, self.input_dim) # regularization
        jac_pesuinv = np.linalg.pinv((jac.transpose() @ self.weight @ jac)) @ jac.transpose() @ self.weight
        delta_dlo_shape = np.clip(self.target_dlo_shape - dlo_shape, -alpha, alpha)
        u_inverse = jac_pesuinv @ delta_dlo_shape
        logpdf = None
        return u_inverse, logpdf

    def update(self, delta_s, delta_x):
        """
            update the kinematic model via Broyden's rule
        """
        self.jacobian_model += 0.1 * (delta_s - self.jacobian_model @ delta_x) / (np.sum(delta_x**2)) @ delta_x.T
        self.inner_n += 1
        return None
        

    def evaluate(self, state):
        action, _ = self.select_action(state)
        return action

    def CollectBootstrapData(self):
        """
            bootstrap using random policy
        """
        pass


@jit
def sdf_box(p: jnp.ndarray, obs_param: dict):
    """
        signed distance function for rectangle box
    """
    obs_center = obs_param.get('box_center', None)
    obs_dx = obs_param.get('box_dx', None)
    obs_dy = obs_param.get('box_dy', None)
    obs_dz = obs_param.get('box_dz', None)
    b = jnp.array([obs_dx, obs_dy, obs_dz])
    # q = jnp.abs( p - obs_center ) - b
    # sdf_val = jnp.linalg.norm(jnp.maximum(q, 0.0)) + jnp.minimum(jnp.maximum(q[0],jnp.maximum(q[1], q[2])), 0.0)
    sdf_val = jnp.linalg.norm(p - obs_center) - jnp.linalg.norm(b)
    return sdf_val

barrier_func = sdf_box

def barrier_func_tmp(x: jnp.ndarray, idx: int, c: jnp.ndarray, size: List[float]):
    p = x[idx]
    obs_param = {
        'box_center': c, 
        'box_dx': size[0],
        'box_dy': size[1],
        'box_dz': size[2]
        }
    d = barrier_func(p, obs_param)
    return d


barrier_func_grad = grad(barrier_func_tmp, argnums=[0,])


class MultiBoxCbfQpAgent:
    def __init__(self, 
                 num_feat:int, 
                 num_grasp:int,
                 box_center:np.ndarray,
                 box_size: List[List[float]], 
                 Q:np.ndarray, 
                 umax:np.ndarray=np.ndarray,
                 umin:np.ndarray=np.ndarray,
                 gamma:float=0.2, 
                 use_closet_point:bool=False,
                 verbose:bool=False):
        """
        Control Barrier Function Based Quadratic Programming Controller

        Args:
            num_feat (int): number of feature points
            
            num_grasp (int): number of grasping points
            
            box_center (np.ndarray): positions of obstacle centers (shape: ``num_obs x 3``)
            
            box_size (list of float): box obstacle sizes
            
            Q (np.ndarray): symmetric cost matrix for cbf-qp
            
            gamma (float, optional): decay rate. Defaults to ``0.2``
            
            use_closet_point (bool, optional): whether only use the point that is closet to the obstacle. Defaults to ``False``
            
            verbose (bool, optional): print QP solver information. Defaults to ``False``
                
        """
        self.num_feat = num_feat
        self.num_grasp = num_grasp
        self.num_obs = box_center.shape[0]
        
        self.dim_u = 3 * self.num_grasp
        self.dim_x = 3 * self.num_feat
        
        self.use_closet_point = use_closet_point
        self.verbose = verbose
        
        self.box_centers = box_center
        self.box_sizes = box_size
        
        # positive definite matrix
        self.P = 1 / 2. * ( Q.T + Q )
        self.gamma = gamma
        
        # barrier function values for each <feature point, obstacle> pair
        self.hs = np.zeros(shape=(self.num_obs, self.num_feat))
        self.ph_ps = np.zeros(shape=(self.num_obs, self.num_feat, self.dim_x))

        # initial guess of control
        self.Uinit = np.zeros(shape=(self.dim_u,))

        self.umax = umax
        self.umin = umin


    def _get_h_value(self, q:np.ndarray):
        """
            compute barrier function value.

            Args:
                q: configuration vector: ``q = [dlo_shape (x); gripper_pos (y)]``
        """
        x = q[0:self.dim_x].reshape((self.num_feat, -1))
        for i in range(self.num_obs):
            obs_param = {
                'box_center': jau.from_numpy(self.box_centers[i]), 
                'box_dx': self.box_sizes[i][0],
                'box_dy': self.box_sizes[i][1],
                'box_dz': self.box_sizes[i][2]
                }
            hs_i = vmap(barrier_func, in_axes=[0, None])(jau.from_numpy(x), obs_param)
            self.hs[i] = jau.to_numpy(hs_i)
        return self.hs
        

    def _get_h_grad(self, q:np.ndarray):
        """
            Get ``partial_h / partial_x``

            Args:
                q (np.ndarray): configuration vector: ``q = [dlo_shape (x); gripper_pos (y)]``
        """
        x_shaped = q[0:self.dim_x].reshape((self.num_feat, -1))
        for i in range(self.num_obs):
            tmp_res, = vmap(barrier_func_grad, in_axes=[None, 0, None, None])(jau.from_numpy(x_shaped), jnp.arange(0, self.num_feat), jau.from_numpy(self.box_centers[i]), self.box_sizes[i])
            self.ph_ps[i] = jau.to_numpy(tmp_res.reshape(self.num_feat, self.dim_x))
        
        return self.ph_ps


    def solve_control(self, 
                      jac_matrix:np.ndarray, 
                      q:np.ndarray, 
                      uref:np.ndarray):
        """
            sovle cbf-qp problem
        """
        hs = self._get_h_value(q)
        ph_ps = self._get_h_grad(q)

        lb = np.array([-0.25]*3*self.num_grasp)
        ub = np.array([ 0.25]*3*self.num_grasp)

        # QP parameters
        P_qp = self.P
        q_qp = -uref.T @ P_qp # (P_qp + 0.1 * np.eye(P_qp.shape[0]))

        min_idx = np.unravel_index(np.argmin(hs), hs.shape)

        min_h_val = self.hs[min_idx]
        
        if self.use_closet_point:
            LJ_h = (ph_ps[min_idx] @ jac_matrix).reshape(1, self.dim_u)
            G = -LJ_h
            h = np.array([[self.gamma * min_h_val]])
            u_opt = solve_qp(P=P_qp, q=q_qp, G=G, h=h, lb=lb, ub=ub, solver="osqp", eps_abs=1e-3, initvals=self.Uinit,  verbose=self.verbose)
            
        else:
            LJ_h = (ph_ps @ jac_matrix).reshape(-1, self.dim_u)
            G = -LJ_h
            h = self.gamma * self.hs.reshape(-1, 1)

            u_opt = solve_qp(P=P_qp, q=q_qp, G=G, h=h, lb=lb, ub=ub, solver="osqp", eps_abs=1e-3, initvals=self.Uinit, verbose=self.verbose)
        
        # when CBF-QP is infeasible
        if u_opt is None:
            print(f'Infeasible CBF-QP. Use Original Control!')
            u_opt = uref
            
        self.Uinit = u_opt  
        return u_opt, min_h_val