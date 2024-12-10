import torch
import torch.nn as nn
import torch.optim as optim
from st_dlo_planning.neural_mpc_tracker.mpc_solver import GradientMPCSolver
from typing import Callable
import numpy as np
import dill
import st_dlo_planning.utils.pytorch_utils as ptu


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