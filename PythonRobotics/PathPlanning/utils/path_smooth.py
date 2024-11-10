import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial


def compute_enery(dlo_shape, k1, k2):
    pass


# Define sigmoid function with control parameter k for sharpness of transition
def sigmoid(x, k):
    return 1 / (1 + jnp.exp(-k * x))

# Define a function to compute smooth transitions for each segment
def weight_function(t, t_i, t_next, k):
    return sigmoid(t - t_i, k) * sigmoid(-(t - t_next), k)

# Function to compute x and y positions at time t with smooth transitions
@jax.jit
def smooth_trajectory(t, waypoints, k=30):
    
    # Step 1: Calculate distances between consecutive waypoints
    distances = jnp.linalg.norm( jnp.diff( waypoints, axis=0 ), axis=1, ord=2)

    # Step 2: Compute cumulative distance (chord length) along the path
    cumulative_distances = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(distances)])

    # Step 3: Normalize cumulative distances to be between 0 and 1
    total_length = cumulative_distances[-1]
    normalized_distances = cumulative_distances / total_length

    # Number of segments
    n = len(normalized_distances) - 1

    point_smooth = jnp.zeros(shape=(waypoints.shape[1],))
    weight_sum = 0.0

    # Iterate over each segment
    for i in range(n):
        # Normalized distances for the current segment
        t_i, t_next = normalized_distances[i], normalized_distances[i + 1]
        # Linear interpolation for point in segment i
        point_i = waypoints[i] + (waypoints[i + 1] - waypoints[i]) * (t - t_i) / (t_next - t_i)

        # Smooth transition weight for the segment
        w_i = weight_function(t, t_i, t_next, k)
        
        # Accumulate the weighted positions and weight sum
        point_smooth += w_i * point_i
        weight_sum += w_i

    # Normalize by total weight to get the smooth transition
    point_smooth /= weight_sum

    return point_smooth

vec_smooth_traj_fn = jax.vmap(smooth_trajectory, in_axes=[0, None, None])
# Generate smooth trajectory over normalized "time" (0 to 1 based on chord length)
sigmas = jnp.linspace(0, 1, 100)

# Define waypoints in Cartesian coordinates (x, y)
waypoints1 = jnp.array([[0.25, 0.0], [0.35, 0.5], [0.35, 1.0], [0.25, 1.5]])
waypoints2 = jnp.array([[0.5, 0.0], [0.5, 0.5], [0.5, 1.0], [0.5, 1.5]])
waypoints3 = jnp.array([[0.75, 0.0], [0.65, 0.5], [0.65, 1.0], [0.75, 1.5]])

trajectory1 = vec_smooth_traj_fn(sigmas, waypoints1, 30)
trajectory2 = vec_smooth_traj_fn(sigmas, waypoints2, 30)
trajectory3 = vec_smooth_traj_fn(sigmas, waypoints3, 30)


point_query_fn1 = partial(smooth_trajectory, waypoints=waypoints1, k=30)
point_query_fn2 = partial(smooth_trajectory, waypoints=waypoints2, k=30)
point_query_fn3 = partial(smooth_trajectory, waypoints=waypoints3, k=30)


# Extract x and y components for plotting
plt.plot(trajectory1[:, 0], trajectory1[:, 1], 'r-', label="Smooth Path")
plt.plot(waypoints1[:, 0], waypoints1[:, 1], 'k-.', label="Raw Path")
plt.scatter(waypoints1[:, 0], waypoints1[:, 1], color='k', label="Waypoints")

plt.plot(trajectory2[:, 0], trajectory2[:, 1], 'r-')
plt.plot(waypoints2[:, 0], waypoints2[:, 1], 'k-.')
plt.scatter(waypoints2[:, 0], waypoints2[:, 1], color='k')

plt.plot(trajectory3[:, 0], trajectory3[:, 1], 'r-')
plt.plot(waypoints3[:, 0], waypoints3[:, 1], 'k-.')
plt.scatter(waypoints3[:, 0], waypoints3[:, 1], color='k')

p1 = point_query_fn1(0.1)[None]
p2 = point_query_fn2(0.05)[None]
p3 = point_query_fn3(0.1)[None]
shape = jnp.concatenate([p1, p2, p3], axis=0)
plt.plot(shape[:, 0], shape[:, 1], 'm-', linewidth=3, label='DLO Shape 1')

p1 = point_query_fn1(0.5)[None]
p2 = point_query_fn2(0.45)[None]
p3 = point_query_fn3(0.3)[None]
shape = jnp.concatenate([p1, p2, p3], axis=0)
plt.plot(shape[:, 0], shape[:, 1], 'g-', linewidth=3, label='DLO Shape 2')

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Path Smoothing")
plt.axis('equal')
plt.show()
