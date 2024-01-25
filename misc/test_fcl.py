import fcl
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 


g1 = fcl.Box(1, 1, 0.0)
t1 = fcl.Transform()
o1 = fcl.CollisionObject(g1, t1)

g2 = fcl.Box(1, 1, 0.0)
t2 = fcl.Transform(jnp.array([2., 0., 0.0]))
o2 = fcl.CollisionObject(g2, t2)

request = fcl.DistanceRequest()
result = fcl.DistanceResult()

ret = fcl.distance(o1, o2, request, result)
print(result.nearest_points)


#define Matplotlib figure and axis
fig, ax = plt.subplots()

ax.plot([result.nearest_points[0][0], result.nearest_points[1][0]],
        [result.nearest_points[0][1], result.nearest_points[1][1]], 'k-.')
ax.add_patch(Rectangle((-0.5, -0.5), 1, 1))
ax.add_patch(Rectangle((1.5, -0.5), 1, 1))
ax.set_aspect('equal', adjustable='box')
plt.show()
