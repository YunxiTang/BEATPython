from pydrake.all import (
    PiecewiseQuaternionSlerp,
    RandomGenerator,
    UniformlyRandomQuaternion,
)

g = RandomGenerator()
q1 = UniformlyRandomQuaternion(g)
q2 = UniformlyRandomQuaternion(g)

print(q1.wxyz(), "\n", q2.wxyz())

alpha = 0.5
slerp1 = PiecewiseQuaternionSlerp([0, 1], [q1, q2])

print(q1.slerp(0.5, q2).wxyz())
print(slerp1.orientation(alpha).wxyz())
