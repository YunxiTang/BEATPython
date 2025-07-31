"""Control Barrier Function (CBF) - QP Controller For Multiple Obstacle"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
import st_dlo_planning.utils.jax_utils as jau
import numpy as np
from qpsolvers import solve_qp
from typing import List


def get_normal_vector(p1, p2):
    delta = p2 - p1
    return jnp.array([-delta[1], delta[0]])


def lse_fn(x, beta):
    exp_tmp = jnp.exp(beta * x)
    sum_tmp = jnp.sum(exp_tmp)
    log_tmp = jnp.log(sum_tmp)
    return 1.0 / beta * log_tmp


@jax.jit
def sdf_box(query_point: jnp.ndarray, vertixes):
    """
    sdf for box
    """
    vertex1 = vertixes[0]
    vertex2 = vertixes[1]
    vertex3 = vertixes[2]
    vertex4 = vertixes[3]

    # compute the normal direction vector
    n12 = get_normal_vector(vertex1, vertex2)
    n23 = get_normal_vector(vertex2, vertex3)
    n34 = get_normal_vector(vertex3, vertex4)
    n41 = get_normal_vector(vertex4, vertex1)

    sdf1 = jnp.dot(query_point - vertex1, n12 / jnp.linalg.norm(n12))
    sdf2 = jnp.dot(query_point - vertex2, n23 / jnp.linalg.norm(n23))
    sdf3 = jnp.dot(query_point - vertex3, n34 / jnp.linalg.norm(n34))
    sdf4 = jnp.dot(query_point - vertex4, n41 / jnp.linalg.norm(n41))

    sdf_val = lse_fn(jnp.array([sdf1, sdf2, sdf3, sdf4]), beta=100.0)
    # sdf_val = jnp.max(jnp.array([sdf1, sdf2, sdf3, sdf4]))
    return sdf_val


barrier_func = sdf_box


def barrier_func_tmp(x: jnp.ndarray, idx: int, vertixes):
    p = x[idx]
    d = barrier_func(p, vertixes)
    return d


barrier_func_grad = jax.jit(
    jax.grad(
        barrier_func_tmp,
        argnums=[
            0,
        ],
    )
)


class MultiCBFQP:
    def __init__(
        self,
        num_feat: int,
        num_grasp: int,
        obstacle_vertixs: List[np.ndarray],
        Q: np.ndarray,
        umax: np.ndarray,
        umin: np.ndarray,
        gamma: float = 0.2,
        use_closet_point: bool = False,
        verbose: bool = False,
    ):
        """
        Control Barrier Function Based Quadratic Programming Controller

        Args:
            num_feat (int): number of feature points

            num_grasp (int): number of grasping points

            obstacle_center (np.ndarray): positions of obstacle centers (shape: ``num_obs x 3``)

            obstacle_radius (list of float): radiuss of obstacle sizes

            Q (np.ndarray): symmetric cost matrix for cbf-qp

            gamma (float, optional): decay rate. Defaults to ``0.2``

            use_closet_point (bool, optional): whether only use the point that is closet to the obstacle. Defaults to ``False``

            verbose (bool, optional): print QP solver information. Defaults to ``False``

        """
        self.num_feat = num_feat
        self.num_grasp = num_grasp
        self.num_obs = len(obstacle_vertixs)

        self.dim_x = 2 * self.num_feat
        self.dim_u = 3 * self.num_grasp

        self.use_closet_point = use_closet_point
        self.verbose = verbose

        self.obs_vertixs = jnp.array(obstacle_vertixs)

        # positive definite matrix
        self.P = 1 / 2.0 * (Q.T + Q)
        self.gamma = gamma

        # barrier function values for each <feature point, obstacle> pair
        self.hs = np.zeros(shape=(self.num_obs, self.num_feat))
        self.ph_ps = np.zeros(shape=(self.num_obs, self.num_feat, self.dim_x))

        # initial guess of control
        self.Uinit = np.zeros(shape=(self.dim_u,))

        self.umax = umax
        self.umin = umin

    def _get_h_value(self, dlo_kp: np.ndarray):
        """
        compute barrier function value.

        Args:
            q: configuration vector: ``q = [dlo_shape (x); gripper_pos (y)]``
        """
        x = dlo_kp
        for i in range(self.num_obs):
            hs_i = vmap(barrier_func, in_axes=[0, None])(
                jau.from_numpy(x), jau.from_numpy(self.obs_vertixs[i])
            )
            self.hs[i] = jau.to_numpy(hs_i)
        return self.hs

    def _get_h_grad(self, dlo_kp: np.ndarray):
        """
        Get ``partial_h / partial_x``

        Args:
            q (np.ndarray): configuration vector: ``q = [dlo_shape (x); gripper_pos (y)]``
        """
        x_shaped = dlo_kp
        for i in range(self.num_obs):
            (tmp_res,) = vmap(barrier_func_grad, in_axes=[None, 0, None])(
                jau.from_numpy(x_shaped),
                jnp.arange(0, self.num_feat),
                jau.from_numpy(self.obs_vertixs[i]),
            )
            self.ph_ps[i] = jau.to_numpy(tmp_res.reshape(self.num_feat, self.dim_x))
        return self.ph_ps

    def get_h_and_min_h(self, dlo_kp):
        hs = self._get_h_value(dlo_kp)
        return hs, np.min(hs)

    def solve_control(
        self, jac_matrix: np.ndarray, dlo_kp: np.ndarray, uref: np.ndarray
    ):
        """
        sovle cbf-qp problem
        """

        hs = self._get_h_value(dlo_kp)
        ph_ps = self._get_h_grad(dlo_kp)

        lb = self.umin
        ub = self.umax

        # QP parameters
        P_qp = self.P
        q_qp = -uref.T @ P_qp

        min_h_val = np.min(hs)

        LJ_h = (ph_ps @ jac_matrix).reshape(-1, self.dim_u)
        G = -LJ_h
        h = self.gamma * self.hs.reshape(-1, 1)
        u_opt = solve_qp(
            P=P_qp,
            q=q_qp,
            G=G,
            h=h,
            lb=lb,
            ub=ub,
            solver="osqp",
            eps_abs=1e-3,
            initvals=self.Uinit,
            verbose=self.verbose,
        )

        # when CBF-QP is infeasible
        if u_opt is None:
            print(f"Infeasible CBF-QP. Use Zero Control!")
            u_opt = 0.0 * uref

        self.Uinit = u_opt
        return u_opt, min_h_val
