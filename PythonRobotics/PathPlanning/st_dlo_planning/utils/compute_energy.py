import jax
import jax.numpy as jnp

@jax.jit
def compute_enery(dlo_shape, k1, k2, segment_len):
    num_feature = dlo_shape.shape[0]
    U1 = 0.0
    for i in range(num_feature-1):
        U1 = U1 + k1/2. * (jnp.linalg.norm(dlo_shape[i+1] - dlo_shape[i]) - segment_len) ** 2
        
    U2 = 0.0
    for j in range(num_feature-2):
        U2 = U2 + k2/2. * (jnp.linalg.norm(dlo_shape[j+2] - dlo_shape[j]) - 2 * segment_len) ** 2
        
    return U1 + U2

if __name__ == '__main__':
    import numpy as np
    x = np.linspace(0., 0.05 * 11, 10)[None] #np.random.uniform(0, 1.0, (10, 3))
    x = np.repeat(x, 3, axis=0).T
    print(x.shape)
    # exit()
    u = compute_enery(x, 1.0, 0.5, 0.05)
    print(u)