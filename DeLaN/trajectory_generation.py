import numpy as np
from math import pi


class Trajectory(object):
    def __init__(self, Atitude, frequnecy):
        self.w = frequnecy
        self.A = Atitude

    def trajectory(self, t):
        q0 = self.A * np.sin(self.w * pi * t)
        qd0 = self.A * self.w * pi * np.cos(self.w * pi * t)
        qdd0 = - self.A * self.w * self.w * pi * pi * np.sin(self.w * pi * t)

        q = np.array([q0, q0]).reshape(2,)
        qd = np.array([qd0, qd0]).reshape(2,)
        qdd = np.array([qdd0, qdd0]).reshape(2,)
        return q, qd, qdd
