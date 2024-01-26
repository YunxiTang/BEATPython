if __name__ == '__main__':
    
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import fcl
import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils import plot_rectangle, plot_circle
from itertools import combinations

fig, ax = plt.subplots()

g1 = fcl.Box(1, 1, 0.0)
t1 = fcl.Transform()
o1 = fcl.CollisionObject(g1, t1)

g2 = fcl.Box(10, 10, 0.0)
t2 = fcl.Transform(jnp.array([20., 10, 0.0]))
o2 = fcl.CollisionObject(g2, t2)

cylinder_g = fcl.Cylinder(15., 0.0)
cylinder_t = fcl.Transform(jnp.array([50., 50, 0.0]))
cylinder = fcl.CollisionObject(cylinder_g, cylinder_t)

request = fcl.DistanceRequest()
result = fcl.DistanceResult()

ret = fcl.distance(cylinder, o1, request, result)

ax.plot([result.nearest_points[0][0], result.nearest_points[1][0]],
        [result.nearest_points[0][1], result.nearest_points[1][1]], 'k-.')

ret = fcl.distance(cylinder, o2, request, result)

ax.plot([result.nearest_points[0][0], result.nearest_points[1][0]],
        [result.nearest_points[0][1], result.nearest_points[1][1]], 'k-.')

request = fcl.DistanceRequest()
result = fcl.DistanceResult()

ret = fcl.distance(o2, o1, request, result)
ax.plot([result.nearest_points[0][0], result.nearest_points[1][0]],
        [result.nearest_points[0][1], result.nearest_points[1][1]], 'k-.')

plot_rectangle(1, 1, 0, 0, ax)
plot_rectangle(10, 10, 20., 10, ax)
# plot_rectangle(30.0, 30.0, 100.0, 90.0, ax)
plot_circle(50, 50, 15, ax)
ax.set_aspect('equal', adjustable='box')
plt.show()


xx = [1, 4., 7., 2.]
for ele in combinations(xx, 2):
    print(ele)