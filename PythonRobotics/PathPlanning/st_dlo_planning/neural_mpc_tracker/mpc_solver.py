"""Gradient-based MPC solver"""
import torch
import torch.nn as nn
import numpy as np
from typing import Callable
import copy

import st_dlo_planning.utils.pytorch_utils as ptu
from st_dlo_planning.neural_mpc_tracker.gdm_dataset import normalize_data, unnormalize_data


class DiscreteModelEnv:
    def __init__(self, dynamic_model, dlo_len, nx, nu, device, stats=None):
        """
        A discrete system model wrapper.

        Args:
            model: the approximated model to wrap.
            dlo_len: dlo length
            nx: dim of keypoints
            nu: dim of control input
            device: GPU/CPU device
            stats: dataset stats to normalie/unnormalize data
        """
        self.dynamic_model = dynamic_model
        self.device = device
        self.nx = nx
        self.nu = nu
        self.dim_q = nx + nu
        self.dlo_len = dlo_len
        self.stats = stats

        self.dynamic_model.eval()
        self.dynamic_model.to(self.device)

    def step(self, 
             state: np.ndarray, 
             delta_eefPos: np.ndarray) -> np.ndarray:
        """
            take a step prediction of the model environment
        """
        # delta_ee_pos need be normalized, while state does not need normalization
        if self.stats is not None:
            _, normalized_delta_ee_pos = normalize_data(self.stats, 
                                                        delta_shape=None, 
                                                        delta_ee_pos=delta_eefPos)
        else:
            normalized_delta_ee_pos = delta_eefPos

        with torch.no_grad():
            state_tensor = ptu.from_numpy(state, self.device)[None]
            normalized_delta_ee_pos_tensor = ptu.from_numpy(normalized_delta_ee_pos, self.device)[None]
            dlo_len_tensor = ptu.from_numpy(np.array([self.dlo_len,]), device=self.device)

            delta_shape = self.dynamic_model(state_tensor, normalized_delta_ee_pos_tensor, dlo_len_tensor)
        # unnormalize the prediction
        delta_shape_np = ptu.to_numpy(delta_shape[0])
        if self.stats is not None:
            unnormalized_delta_shape, _ = unnormalize_data(self.stats, 
                                                        normalized_delta_shape=delta_shape_np, 
                                                        normalized_delta_ee_pos=None)
        else:
            unnormalized_delta_shape = delta_shape_np
            
        delta_state = np.concatenate([unnormalized_delta_shape * 1.0, delta_eefPos])
        next_state = state + delta_state
        return next_state

    def render(self, render_mode="human"):
        print('Not Implemented Yet.')


# ============================== Gradient Based Solver ======================================= # 
class GradientMPCSolver:
    def __init__(self,
                 system_model: nn.Module,
                 path_cost_func: Callable,
                 final_cost_func: Callable,
                 dlo_len: float,
                 nx: int,
                 nu: int,
                 dt: float, 
                 device,
                 u_max: np.ndarray,
                 u_min: np.ndarray,
                 discount_factor: float = 1.0,
                 horizon: int = 10,
                 tol: float = 1e-3,
                 lr: float = 0.001,
                 max_iter: int = 2):
        
        self.model = system_model

        self.dlo_len = torch.tensor(dlo_len, device=device)
        self.path_cost_func = path_cost_func
        self.final_cost_func = final_cost_func
        
        self._lr = lr
        self._horizon = horizon
        
        self._nx = nx
        self._nu = nu
        self._umax = u_max
        self._umin = u_min
        self._discount_factor = discount_factor
        self._dt = dt
        self._device = device
        self._max_iter = max_iter
        self._tol = tol

        # initial guess for U, i.e. delta eef positions
        self.U = torch.zeros(self._horizon, self._nu, 
                             requires_grad=True, device=self._device)

        # MPC numerical optimizer
        self.optimizer = torch.optim.AdamW([self.U,], self._lr, weight_decay=0.001)

        
    def plan(self, x_init, xref):
        """ 
            plan a sequence of trajectory via gradient-based optimization
        """
        inter_target_shape = ptu.from_numpy(xref, device=self._device)

        dlo_len_b = self.dlo_len[None]
        loss_last = 1e4
        for iter in range(self._max_iter):
            cumulative_loss = 0.0
            x = ptu.from_numpy(x_init, self._device)
            for t in range(self._horizon):
                x_b = x[None]
                # u_b = torch.tanh(self.U[t][None]) * self._umax[0]
                u_b = self.U[t][None]
                
                delta_X = self.model(x_b, u_b, dlo_len_b)[0]
                delta_eef = self.U[t]
                delta_x = torch.cat([delta_X, delta_eef], dim=0)

                x_next = x + delta_x
                cumulative_loss = cumulative_loss \
                                + self._discount_factor ** t * self.path_cost_func(x_next, self.U[t], inter_target_shape)
                # forward simulate the system dynamics
                x = x_next
                
            cumulative_loss = cumulative_loss + self._discount_factor ** t * self.final_cost_func(x, inter_target_shape)

            if cumulative_loss.data.item() < loss_last:
                self.optimizer.zero_grad()
                cumulative_loss.backward()
                self.optimizer.step()
                
                loss_np = ptu.to_numpy(cumulative_loss)
                print(f'MPC Iter: {iter} with objective {loss_np}')
                if loss_last - loss_np < self._tol:
                    break
                loss_last = loss_np
            else:
                break
        return ptu.to_numpy(self.U), loss_np