import cyipopt
from cyipopt import minimize_ipopt
import numpy as np
from typing import Callable
import copy

from st_dlo_planning.utils import PathSet, smooth_trajectory, compute_enery

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev

jax.config.update("jax_enable_x64", True)     # enable fp64
jax.config.update('jax_platform_name', 'cpu') # use the CPU instead of GPU


class DLOOptProblem():

    def __init__(self, pathset: PathSet, k1: float, k2: float):
        self.pathset = pathset
        self.virtual_k1 = k1
        self.virtual_k2 = k2
        
        self.T = self.pathset.T               # discrete resolution
        self.num_path = self.pathset.num_path # exactly the $n$ used in the paper
        
        # decision_variable_dim
        self.decision_variable_dim = self.T * self.num_path
        
        # constraint dim
        self.num_constraint = self.num_path * 2 + self.num_path * (self.T - 1)
        
    def _assemble_shape(self, sigma):
        dlo_shape = []
        for waypoints in self.pathset:
            feature_point = smooth_trajectory(sigma, waypoints, k=30)
            dlo_shape.append(feature_point)
        return np.concatenate(dlo_shape, axis=0)
    
    def _compute_potential_energy(self, dlo_shape):
        u = compute_enery(dlo_shape, self.virtual_k1, self.virtual_k2, self.segment_len)
        return u
    
    def _path_constraint(self, dlo_shape):
        pass
    
    def _terminal_constraint(self, dlo_shape):
        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def objective(self, x: np.ndarray):
        """
            objective function E = \sum^T_{t=0} U_t
        """
        sigmas = x.reshape(self.T, self.path_num)
        # accumulate potential energy [u_0, ..., u_T]
        Loss = 0.0
        for t in range(self.T): 
            dlo_shape = self._assemble_shape(sigmas[t])
            u = self._compute_potential_energy(dlo_shape)
            Loss += u
        return Loss

    
    def gradient(self, x):
        '''
            Approximate the gradient of the objective function 
            with finite difference method (FDM) 
            (TODO: replace with JAX AD to compute gradients)
        '''
        delta = 1e-3
        gradient_list = np.zeros((self.decision_variable_dim,))
        for i in range(len(x)):
            e = np.zeros((self.decision_variable_dim, ))
            e[i] = delta
            x_upper = x + e
            x_lower = x - e
            obj_upper = self.objective(x_upper)
            obj_lower = self.objective(x_lower)
            gradient = (obj_upper - obj_lower) / (2. * delta)
            gradient_list[i] = gradient
        return gradient_list

    def constraints(self, x):
        """
            Returns the constraints.
        """
        sigma = x.reshape(self.T, self.num_path)
        sigma_0 = sigma[0, :]          
        sigma_T = sigma[self.T-1, :]  
        
        # === equality constraints (fix initial shape and terminal shape) ====
        init_eq = sigma_0 - np.zeros_like(sigma_T)  # in shape of (self.num_path, )
        term_eq = sigma_T - np.ones_like(sigma_T)   # in shape of (self.num_path, )
        
        # === inequality constraints ====
        # for each path, we force that: sigma_T > sigma_{T-1} > ... > sigma_1 > sigma_0 
        path_ineq = np.diff(sigma, axis=0)          
        path_ineq = path_ineq.flatten()             # in shape of ((self.T-1) x self.num_path, )
        constraints = np.concatenate([init_eq, term_eq, path_ineq])
        return constraints

    # def jacobian(self, x):
    #     """
    #         The callback for calculating the Jacobian
    #     """
    #     return None

    # def hessianstructure(self):
    #     """ 
    #         an example callback for hessian structure
    #     """
    #     return None

    # def hessian(self, x, lagrange, obj_factor):
    #     """
    #         The callback for calculating the Hessian
    #     """
    #     return None

    # def intermediate(self, 
    #                  alg_mod, 
    #                  iter_count, 
    #                  obj_value, 
    #                  inf_pr, 
    #                  inf_du,
    #                  mu,
    #                  d_norm,
    #                  regularization_size,
    #                  alpha_du,
    #                  alpha_pr,
    #                  ls_trials):
    #     """
    #         intermediate callback.
    #     """
    #     print(f"Iteration #{iter_count}: Objective value: {obj_value}.")


class TcDLOSolver:
    """
        Temporal DLO Configuration Optimization using IPOPT Solver From Path Set
    """
    def __init__(self, pathset: PathSet, k1: float, k2: float, 
                 tol: float = 1e-3, max_iter: int = 30):
        
        self.pathset = pathset
        self.T = pathset.T
        self.num_path = pathset.num_path
        
        self.k1 = k1
        self.k2 = k2
        
        self._tol = tol
        self._max_iter = max_iter
        
        # NLP problem related
        self.lb = np.repeat([0.], self.pathset.num_path * self.pathset.T)
        self.ub = np.repeat([1.], self.pathset.num_path * self.pathset.T)
        
        self.cl = np.array([0.,] * self.num_path + [0.,] * self.num_path + [0.,] * (self.T-1) * self.num_path)
        self.cu = np.array([0.,] * self.num_path + [0.,] * self.num_path + [0.5,] * (self.T-1) * self.num_path)
        
        # initialize the decision variables
        self.init_sigmas = np.repeat( np.linspace(0.0, 1.0, self.pathset.T), self.pathset.num_path)
        
    def solve(self):
        """ 
            plan a DLO configuration sequence
        """
        problem = DLOOptProblem(self.pathset, self.k1, self.k2)
        
        nlp = cyipopt.Problem(
            n=len(self.init_sigmas),
            m=problem.num_constraint,
            problem_obj=problem,
            lb=self.lb,
            ub=self.ub,
            cl=self.cl,
            cu=self.cu
            )

        # ================ Set solver options =======================
        nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('tol', self._tol)
        nlp.add_option('max_iter', self._max_iter)
        nlp.add_option('print_level', 0)
        # nlp.add_option('gradient_approximation', 'finite-difference-values')
        # nlp.add_option('jacobian_approximation', 'finite-difference-values')
        # nlp.add_option('hessian_approximation', 'limited-memory')
        # nlp.add_option('max_cpu_time', 0.5)
        # nlp.addOption('derivative_test', 'second-order')

        # Solve the problem
        opt_sigmas, info = nlp.solve(self.init_sigmas)
            
        return opt_sigmas
    
    
if __name__ == '__main__':
    print('testing')