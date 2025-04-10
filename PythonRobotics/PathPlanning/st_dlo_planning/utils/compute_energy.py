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

    U3 = 0.0
    for j in range(num_feature-3):
        U3 = U3 + k2/2. * (jnp.linalg.norm(dlo_shape[j+3] - dlo_shape[j]) - 3 * segment_len) ** 2

    U4 = 0.0
    for j in range(num_feature-4):
        U4 = U4 + k2/2. * (jnp.linalg.norm(dlo_shape[j+4] - dlo_shape[j]) - 4 * segment_len) ** 2

    U5 = 0.0
    for j in range(num_feature-5):
        U5 = U5 + k2/2. * (jnp.linalg.norm(dlo_shape[j+5] - dlo_shape[j]) - 5 * segment_len) ** 2

    Ux = 0.0
    for j in range(num_feature-6):
        Ux = Ux + k2/2. * (jnp.linalg.norm(dlo_shape[j+6] - dlo_shape[j]) - 6 * segment_len) ** 2
        
    return U1 + U2 + U3 + U4 #+ Ux

if __name__ == '__main__':
    import numpy as np
    seg_len = 0.05
    x = np.array([[0.0, seg_len, 0.0],
                  [0.0, 2*seg_len, 0.0],
                  [0.0, 3*seg_len, 0.0]])
    
    print(x.shape)
    # exit()
    u = compute_enery(x, 1.0, 0.5, seg_len)
    print(u)