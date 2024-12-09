import cyipopt
from cyipopt import minimize_ipopt
import numpy as np
from typing import Callable
import copy

from st_dlo_planning.utils import PathSet, compute_enery

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev

jax.config.update("jax_enable_x64", True)     # enable fp64
jax.config.update('jax_platform_name', 'cpu') # use the CPU instead of GPU


class DloOptProblem():
    '''
        objective function E = \sum^T_{t=0} U_t
        objective(x):
            sigmas = x.reshape(self.T+1, self.num_path)
            # accumulate potential energy [u_0, ..., u_T]
            Loss = 0.0
            for t in range(self.T-1): 
                sigma = sigmas[t]
                dlo_shape = self._assemble_shape(sigma)
                u = self._compute_potential_energy(dlo_shape)
                Loss += u
            return Loss
    '''
    def __init__(self, pathset: PathSet, k1: float, k2: float):
        self.pathset = pathset
        self.virtual_k1 = k1
        self.virtual_k2 = k2
        
        self.T = self.pathset.T               # discrete resolution
        self.num_path = self.pathset.num_path # exactly the $n$ used in the paper
        
        # decision_variable_dim
        self.decision_variable_dim = (self.T + 1) * self.num_path
        
        # constraint dim
        self.num_constraint = self.num_path * 2 + self.num_path * self.T

        self.seg_len = pathset.seg_len

        self.gradient = jax.jit(jax.grad(self.objective))
        self.jacobian = jax.jit(jax.jacfwd(self.constraints))

        self.init_shape = self.pathset.query_dlo_shape(jnp.array([0.]*self.num_path))
        self.goal_shape = self.pathset.query_dlo_shape(jnp.array([1.]*self.num_path))
    
        
    def _assemble_shape(self, sigma):
        dlo_shape = self.pathset.query_dlo_shape(sigma)
        return dlo_shape
    
    def _compute_potential_energy(self, dlo_shape):
        u = compute_enery(dlo_shape, self.virtual_k1, self.virtual_k2, self.seg_len)
        return u
    
    def _path_constraint(self, dlo_shape):
        pass
    
    def _terminal_constraint(self, dlo_shape):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def objective(self, x):
        """
            objective function E = \sum^T_{t=0} U_t
        """
        sigmas = jnp.reshape(x, (self.T+1, self.num_path))
        @jax.jit
        def _sigma_to_energy(carry, sigma):
            dlo_shape = self._assemble_shape(sigma)
            u = self._compute_potential_energy(dlo_shape) #+ 0.1 * jnp.linalg.norm(dlo_shape-self.goal_shape)
            new_carry = u + carry
            return new_carry, u
        loss, _ = jax.lax.scan(_sigma_to_energy, 0.0, sigmas, length=self.T+1)
        regularization = jnp.sum( jnp.diff(sigmas, axis=0) ** 2 )
        return loss + 1.5 * regularization


    @partial(jax.jit, static_argnums=(0,))
    def constraints(self, x):
        """
            Returns the constraints.
        """
        sigma = jnp.reshape(x, (self.T+1, self.num_path))
        sigma_0 = sigma[0, :]          
        sigma_T = sigma[self.T, :]  
        
        # === equality constraints (fix initial shape and terminal shape) ====
        init_eq = sigma_0 - jnp.zeros_like(sigma_0)  # in shape of (self.num_path, )
        term_eq = sigma_T - jnp.ones_like(sigma_T)   # in shape of (self.num_path, )
        
        # === inequality constraints ====
        # for each path, we force that: sigma_T > sigma_{T-1} > ... > sigma_1 > sigma_0 
        path_ineq = jnp.diff(sigma, axis=0)          
        path_ineq = jnp.reshape(path_ineq, (self.T) * self.num_path)            
        constraints = jnp.concatenate([init_eq, term_eq, path_ineq])
        return constraints

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

    def intermediate(self, 
                     alg_mod, 
                     iter_count, 
                     obj_value, 
                     inf_pr, 
                     inf_du,
                     mu,
                     d_norm,
                     regularization_size,
                     alpha_du,
                     alpha_pr,
                     ls_trials):
        """
            intermediate callback.
        """
        if iter_count % 5 == 0:
            print(f"Iteration #{iter_count}: Obj. Val.: {obj_value}.")


class TcDloSolver:
    """
        Temporal DLO Configuration Optimization using IPOPT Solver From Path Set
    """
    def __init__(self, pathset: PathSet, k1: float, k2: float, 
                 tol: float = 1e-6, max_iter: int = 300):
        
        self.pathset = pathset
        self.T = pathset.T
        self.num_path = pathset.num_path
        
        self.k1 = k1
        self.k2 = k2
        
        self._tol = tol
        self._max_iter = max_iter
        
        # NLP problem related
        self.lb = np.repeat([0.], self.pathset.num_path * (self.pathset.T + 1))
        self.ub = np.repeat([1.], self.pathset.num_path * (self.pathset.T + 1))
        
        self.cl = np.array([0.,] * self.num_path + [0.,] * self.num_path + [0.002,] * (self.T * self.num_path))
        self.cu = np.array([0.,] * self.num_path + [0.,] * self.num_path + [0.05,] * (self.T * self.num_path))
        
        # initialize the decision variables
        # self.init_sigmas = np.ones((self.pathset.T + 1) * self.pathset.num_path) * 0.0
        self.init_sigmas = np.repeat( np.linspace(0.0, 1.0, self.pathset.T + 1, endpoint=True), self.pathset.num_path)
        
    def solve(self):
        """ 
            plan a DLO configuration sequence
        """
        problem = DloOptProblem(self.pathset, self.k1, self.k2)
        
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
        # nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('tol', self._tol)
        nlp.add_option('max_iter', self._max_iter)
        nlp.add_option('print_level', 0)
        # nlp.add_option('hessian_approximation', 'limited-memory')
        # nlp.add_option('max_cpu_time', 0.5)
        # nlp.addOption('derivative_test', 'second-order')
        # nlp.add_option('gradient_approximation', 'finite-difference-values')
        # nlp.add_option('jacobian_approximation', 'finite-difference-values')

        # Solve the problem
        opt_sigmas, info = nlp.solve(self.init_sigmas)
            
        return opt_sigmas, info
    