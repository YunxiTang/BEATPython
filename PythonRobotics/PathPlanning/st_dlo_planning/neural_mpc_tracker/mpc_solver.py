"""Gradient-based MPC solver"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable
import copy

import st_dlo_planning.utils.pytorch_utils as ptu
from st_dlo_planning.neural_mpc_tracker.gdm_dataset import (
    normalize_data,
    unnormalize_data,
)
from st_dlo_planning.neural_mpc_tracker.modelling_gdm import GDM


class DiscreteModelEnv:
    def __init__(self, dynamic_model, nx, nu, device, stats=None):
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

        self.dynamic_model.eval()
        self.dynamic_model.to(self.device)

        self.stats = stats

    def step(self, state: np.ndarray, delta_eefPos: np.ndarray) -> np.ndarray:
        """
        take a step prediction of the model environment
        """
        # delta_ee_pos need be normalized, while state does not need normalization
        if self.stats is not None:
            _, normalized_delta_ee_pos = normalize_data(
                self.stats, delta_shape=None, delta_ee_pos=delta_eefPos
            )
        else:
            normalized_delta_ee_pos = delta_eefPos

        with torch.no_grad():
            state_tensor = ptu.from_numpy(state, self.device)[None]
            normalized_delta_ee_pos_tensor = ptu.from_numpy(
                normalized_delta_ee_pos, self.device
            )[None]
            dlo_len_tensor = ptu.from_numpy(
                np.array(
                    [
                        self.dlo_len,
                    ]
                ),
                device=self.device,
            )

            delta_shape = self.dynamic_model(
                state_tensor, normalized_delta_ee_pos_tensor, dlo_len_tensor
            )
        # unnormalize the prediction
        delta_shape_np = ptu.to_numpy(delta_shape[0])
        if self.stats is not None:
            unnormalized_delta_shape, _ = unnormalize_data(
                self.stats,
                normalized_delta_shape=delta_shape_np,
                normalized_delta_ee_pos=None,
            )
        else:
            unnormalized_delta_shape = delta_shape_np

        delta_state = np.concatenate([unnormalized_delta_shape * 1.0, delta_eefPos])
        next_state = state + delta_state
        return next_state

    def render(self, render_mode="human"):
        print("Not Implemented Yet.")


# ============================== Gradient Based Solver ======================================= #
def relaxed_log_barrier(x, delta=0.01, phi=1.0):
    delta = torch.tensor(delta)
    if x > delta:
        val = -phi * torch.log(x)
    else:
        val = 1 / 2 * (((x - 2 * delta) / delta) ** 2 - 1) - torch.log(delta)
    return val


class GradientMPCSolver:
    def __init__(
        self,
        system_model: nn.Module,
        path_cost_func: Callable,
        final_cost_func: Callable,
        dlo_len: float,
        num_eef: int,
        dt: float,
        device,
        umin=None,
        umax=None,
        discount_factor: float = 1.0,
        horizon: int = 10,
        tol: float = 1e-3,
        lr: float = 0.001,
        max_iter: int = 2,
    ):
        self.model = system_model

        self.dlo_len = torch.tensor(dlo_len, device=device)
        self.path_cost_func = path_cost_func
        self.final_cost_func = final_cost_func

        self._lr = lr
        self._horizon = horizon

        self._num_eef = num_eef
        self._discount_factor = discount_factor
        self._dt = dt
        self._device = device
        self._max_iter = max_iter
        self._tol = tol

        # initial guess for U, i.e. delta eef positions
        self.U = torch.zeros(
            self._horizon, self._num_eef, 3, requires_grad=True, device=self._device
        )

        self.umin = ptu.from_numpy(umin)
        self.umax = ptu.from_numpy(umax)
        self.umin = self.umin.reshape(self._num_eef, 3)
        self.umax = self.umax.reshape(self._num_eef, 3)

        # MPC numerical optimizer
        self.optimizer = torch.optim.AdamW(
            [
                self.U,
            ],
            self._lr,
            weight_decay=0.001,
        )

    def plan(self, dlo_kp_2d_init, eef_states_init, dlo_kp_ref_2d):
        """
        plan a sequence of trajectory via gradient-based optimization
        """
        inter_target_shape = ptu.from_numpy(dlo_kp_ref_2d, device=self._device)
        loss_last = float("inf")

        phi = 100.0
        for iter in range(self._max_iter):
            cumulative_loss = 0.0
            dlo_kp_2d = ptu.from_numpy(dlo_kp_2d_init, self._device)
            eef_states = ptu.from_numpy(eef_states_init, self._device)

            for t in range(self._horizon):
                dlo_kp_2d_b = dlo_kp_2d[None]
                eef_states_b = eef_states[None]
                u_b = self.U[t][None]  # torch.tanh(self.U[t][None]) * self.umax[None]

                delta_dlo_kp_2d = self.model(dlo_kp_2d_b, eef_states_b, u_b)[
                    0
                ]  # dlo_kp, eef_states, delta_eef_states
                delta_eef = u_b[0]  # torch.clamp(self.U[t], self.umin, self.umax)

                next_dlo_kp_2d = dlo_kp_2d + delta_dlo_kp_2d
                next_eef_states = eef_states + delta_eef * 1.0

                cumulative_loss = (
                    cumulative_loss
                    + self._discount_factor** t
                    * self.path_cost_func(
                        next_dlo_kp_2d,
                        next_eef_states,
                        self.U[t],
                        inter_target_shape,
                        phi,
                    )
                )
                # forward simulate the system dynamics
                dlo_kp_2d = next_dlo_kp_2d
                eef_states = next_eef_states

            cumulative_loss = (
                cumulative_loss
                + self._discount_factor** t
                * self.final_cost_func(dlo_kp_2d, eef_states, inter_target_shape, phi)
            )
            loss_np = ptu.to_numpy(cumulative_loss)
            if loss_last - loss_np > self._tol or iter < 2:
                self.optimizer.zero_grad()
                cumulative_loss.backward()
                self.optimizer.step()
                # phi = 5 * phi

                # print(f'MPC Iter: {iter} with objective {loss_np}')
                loss_last = loss_np
            else:
                break
        return ptu.to_numpy(self.U), loss_np
