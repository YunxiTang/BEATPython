# # # estimating virtual stiffnesses for spring-mass model
# # if __name__ == '__main__':
# #     import sys
# #     import os
# #     import pathlib
# #     import matplotlib.pyplot as plt
# #     import numpy as np
# #     import jax.numpy as jnp
# #     import seaborn as sns
# #     import zarr
# #     import flax
# #     import flax.linen as nn

# #     sns.set_theme('paper')


# #     ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)

# #     sys.path.append(ROOT_DIR)
# #     os.chdir(ROOT_DIR)

# #     from st_dlo_planning.utils.world_map import plot_circle
# #     from st_dlo_planning.utils.path_interpolation import visualize_shape
# #     import jax
# #     jax.config.update("jax_enable_x64", True)     # enable fp64
# #     jax.config.update('jax_platform_name', 'gpu') # use the CPU instead of GPU


# #     def get_elastic_enery(ks, dlo_shape, segment_len):
# #         k1 = ks[0]
# #         k2 = ks[1]
# #         num_feature = dlo_shape.shape[0]

# #         # r = k1 / k2 # nn.relu(k1) / (nn.relu(k2) + 0.001)
# #         U1 = 0.0
# #         for i in range(num_feature-1):
# #             U1 = U1 + k1/2. * (jnp.linalg.norm(dlo_shape[i+1] - dlo_shape[i]) - segment_len) ** 2

# #         U2 = 0.0
# #         for j in range(num_feature-2):
# #             U2 = U2 + k2/2. * (jnp.linalg.norm(dlo_shape[j+2] - dlo_shape[j]) - 2 * segment_len) ** 2
# #         return U1 + U2

# #     loss_grad_fn = jax.jit(jax.value_and_grad(get_elastic_enery, argnums=[0,]))

# #     zarr_root = zarr.open('/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/results/gdm_mj/train/task03_10.zarr')
# #     dlo_len = zarr_root['meta']['dlo_len'][0]
# #     keypoints = zarr_root['data']['dlo_keypoints'][:]
# #     keypoints = keypoints.reshape(-1, 13, 3)
# #     print(keypoints.shape)

# #     straight_shape = keypoints[0]

# #     seg_len = np.linalg.norm(straight_shape[0] - straight_shape[1])

# #     ks = jnp.array([10., 10.])
# #     n = keypoints.shape[0]
# #     iter = 2000
# #     ks_measure = []
# #     for i in range(n):
# #         shape_sample = keypoints[i]

# #         loss_pre = 1e5
# #         for j in range(iter):
# #             loss, grads = loss_grad_fn(ks, shape_sample, seg_len)
# #             ks = ks - 0.1 * grads[0]
# #             if loss > loss_pre:
# #                 print(j)
# #                 break
# #             loss_pre = loss
# #         ks_measure.append(ks)
# #         print(i, ks)
# #         print(' +++++++++++++++++++++++++++++++ ')


# import numpy as np
# from scipy.optimize import minimize

# # Define keypoint positions (example data)
# keypoints = np.array([
#     [0.0, 0.0],
#     [1.0, 0.5],
#     [2.0, 0.8],
#     [3.0, 0.5],
#     [4.0, 0.0]
# ])

# # Define rest lengths (example data)
# rest_lengths = np.array([1.0, 1.0, 1.0, 1.0])  # Assuming uniform rest lengths

# # Number of keypoints and springs
# n = len(keypoints)
# num_springs = n - 1

# # Function to compute the total potential energy
# def total_potential_energy(k_values):
#     """
#     Compute the total potential energy of the system.
#     """
#     total_energy = 0.0
#     for i in range(num_springs):
#         x1 = keypoints[i]
#         x2 = keypoints[i + 1]
#         current_length = np.linalg.norm(x2 - x1)
#         delta_L = current_length - rest_lengths[i]
#         total_energy += 0.5 * k_values[i] * delta_L ** 2
#     return total_energy

# # Initial guess for stiffness values (uniform stiffness)
# k_initial = np.ones(num_springs) * 15 # Example: [1.0, 1.0, 1.0, 1.0]

# # Bounds for stiffness values (must be positive)
# bounds = [(1e-1, None) for _ in range(num_springs)]  # k_i > 0

# # Optimize to find stiffness values
# result = minimize(total_potential_energy, k_initial, bounds=bounds)

# # Extract optimized stiffness values
# optimized_stiffness = result.x
# print("Optimized stiffness values:", optimized_stiffness)
