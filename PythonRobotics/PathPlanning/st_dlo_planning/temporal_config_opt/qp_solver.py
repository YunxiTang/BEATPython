'''
    QP solver to polish/smooth the DLO shape 
'''
import jax
import jax.numpy as jnp
from optax import adamw
from jax import random, vmap, grad
import numpy as np
import qpsolvers
from qpsolvers import solve_qp

@jax.jit
def elastic_enery(intermediate_points, init_intermediate_points, end_point1, end_point2, k1, k2, segment_len):
    dlo_shape = jnp.concatenate([end_point1, intermediate_points, end_point2], axis=0)
    num_feature = dlo_shape.shape[0]
    U1 = 0.0
    for i in range(num_feature-1):
        U1 = U1 + k1/2. * (jnp.linalg.norm(dlo_shape[i+1] - dlo_shape[i]) - segment_len) ** 2
        
    U2 = 0.0
    for j in range(num_feature-2):
        U2 = U2 + k2/2. * (jnp.linalg.norm(dlo_shape[j+2] - dlo_shape[j]) - 2 * segment_len) ** 2

    U3 = 0.0
    for j in range(num_feature-3):
        U3 = U3 + k2/2. * (jnp.linalg.norm(dlo_shape[j+3] - dlo_shape[j]) - 3 * segment_len) ** 2

    U4 = 0.0
    for j in range(num_feature-4):
        U4 = U4 + k2/2. * (jnp.linalg.norm(dlo_shape[j+4] - dlo_shape[j]) - 4 * segment_len) ** 2

    U5 = 0.0
    for j in range(num_feature-5):
        U5 = U5 + k2/2. * (jnp.linalg.norm(dlo_shape[j+5] - dlo_shape[j]) - 5 * segment_len) ** 2
        
    reg = 0.01 * jnp.mean((intermediate_points - init_intermediate_points)**2)
    return U1 + U2 + U3 + U4 + U5


val_and_grad_fn = jax.value_and_grad(elastic_enery)


def polish_dlo_shape(raw_shape, k1, k2, segment_len, iter:int=500, verbose:bool=False):
    intermidiate_points = raw_shape[1:-1]
    endpoint1 = raw_shape[0][None]
    endpoint2 = raw_shape[-1][None]
    init_intermediate_points = intermidiate_points
    energy_pre = 10000.

    for i in range(iter):
        energy, grads = val_and_grad_fn(intermidiate_points, init_intermediate_points, endpoint1, endpoint2, k1, k2, segment_len)
        intermidiate_points = intermidiate_points - 0.005 * grads
        if i % 20 == 0 and verbose:
            print(f'{i}: {energy}')
        if energy_pre - energy <= 1e-5:
            break
        energy_pre = energy
    new_shape = jnp.concatenate([endpoint1, intermidiate_points, endpoint2], axis=0)
    return new_shape
