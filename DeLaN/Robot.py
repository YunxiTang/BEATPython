"""Define a robot example: Pendubot"""

import numpy as np
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
from math import *
import matplotlib.animation as animation


class Pendubot():
    def __init__(self):
        """default simulation settings"""

        self.t_span = (0.0, 30.0)
        self.initial_state = np.array([-pi/4, 0.0, 0.0, 0.0], dtype=np.float)
        self.dt = 0.002

        # model parameters
        self.l1 = 1
        self.l2 = 1
        self.lc1 = 0.5
        self.lc2 = 0.5
        self.m1 = 1
        self.m2 = 1
        self.I1 = 0.33
        self.I2 = 0.33
        self.b1 = 0.
        self.b2 = 0.
        self.g = 9.81

    def forward_kinematics(self, state):
        """forward kinematics of robot"""
        x = state
        q1 = x[:, 0]
        q2 = x[:, 1]
        l1 = self.l1
        l2 = self.l2

        x1 = l1 * np.cos(q1)
        y1 = l1 * np.sin(q1)
        x2 = x1 + l2 * np.cos(q1 + q2)
        y2 = y1 + l2 * np.sin(q1 + q2)

        return x1, y1, x2, y2

    def jacobian_compute(self, current_state):
        x = current_state
        """Jacobian Computing"""
        l1 = self.l1
        l2 = self.l2
        q1 = x[0]
        q2 = x[1]
        J11 = - l2 * np.sin(q1 + q2) - l1 * np.sin(q1)
        J12 = - l2 * np.sin(q1 + q2)
        J21 = l2 * np.cos(q1 + q2) + l1 * np.cos(q1)
        J22 = l2 * np.cos(q1 + q2)
        J = np.array([[J11, J12],
                      [J21, J22]])
        return J

    def EoM(self, current_state):
        """Equations of Motion"""
        x = current_state
        q = x[0:2]
        qd = x[2:4]

        s = np.sin(q)
        c = np.cos(q)

        s12 = np.sin(np.sum(q))
        c12 = np.cos(np.sum(q))

        M11 = self.m1 * (self.lc1 ** 2) + self.m2 * (self.l1 ** 2 + self.lc2 ** 2
                                                     + 2 * self.l1 * self.lc2 * c[1]) + self.I1 + self.I2
        M12 = self.m2 * (self.lc2 ** 2 + self.l1 * self.lc2 * c[1]) + self.I2
        M21 = M12
        M22 = self.m2 * self.lc2 ** 2 + self.I2
        M = np.array([[M11, M12],
                      [M21, M22]], dtype=np.float64)

        C11 = -2 * self.m2 * self.l1 * self.lc2 * s[1] * qd[1]
        C12 = -self.m2 * self.l1 * self.lc2 * s[1] * qd[1]
        C21 = self.m2 * self.l1 * self.lc2 * s[1] * qd[0]
        C22 = 0
        C = np.array([[C11, C12],
                      [C21, C22]], dtype=np.float64)

        G1 = self.g * (self.m1 * self.lc1 * c[0] + self.m2 * (self.l1 * c[0] + self.lc2 * c12))
        G2 = self.g * self.m2 * self.lc2 * c12

        G = np.array([G1, G2], dtype=np.float64).reshape((2, 1))

        F = np.array([[self.b1, 0],
                      [0, self.b2]], dtype=np.float64)

        B = np.array([[1, 0],
                      [0, 1]])

        return M, C, G, F, B

    def simulation(self, torque):
        t0 = self.t_span[0]
        tf = self.t_span[1]
        num = int((tf - t0) / self.dt)
        t_lx = np.linspace(t0, tf, num, endpoint=True)
        solx = odeint(dynamics, self.initial_state.reshape(4, ), t_lx, args=(torque, self), tfirst=True)
        return t_lx, solx

    def get_next_state(self, current_state, torque):
        t0 = self.t_span[0]
        tf = t0 + 3 * self.dt
        num = int((tf - t0) / self.dt)
        t_lin = np.linspace(t0, tf, num, endpoint=True)
        sol2 = odeint(dynamics, current_state.reshape(4, ), t_lin, args=(torque, self), tfirst=True)
        next_state = sol2[1, :]
        return next_state


def dynamics(t, s, force, robot_):
    # s = unwrap(s)
    # action = force
    # q = np.reshape(s[0:2], (2, 1))
    # print(q)
    # qd = np.reshape(s[2:], (2, 1))
    # print(qd)
    # M, C, G, F, B = robot_.EoM(s)
    # M_inv = np.linalg.inv(M)
    # print('M_inv', M_inv)
    # print('Actuation', (np.matmul(B, action.reshape(2,))).reshape(2, 1))
    # print('C term', np.matmul(C, qd))
    # print('F term', np.matmul(F, qd))
    # print('G', G)
    # qdd = np.matmul(M_inv, (np.matmul(B, action.reshape(2,))).reshape(2, ) - np.matmul(C, qd) - np.matmul(F, qd) - G)
    # print(qdd)
    # sd = np.append(qd, qdd)
    # return sd
    s = unwrap(s)
    action = force.reshape(2, 1)
    q = np.reshape(s[0:2], (2, 1))
    qd = np.reshape(s[2:4], (2, 1))
    M, C, G, F, B = robot_.EoM(s)
    M_inv = np.linalg.inv(M)
    qdd = np.matmul(M_inv, (np.matmul(B, action) - np.matmul(C, qd) - np.matmul(F, qd) - G))
    sd = np.append(qd, qdd)
    return sd


def plotter(t_l, sol):
    plt.plot(t_l, sol[:, 0] / pi * 180, color="blue", linewidth=2.0, linestyle="-", label="q1")
    plt.plot(t_l, sol[:, 1] / pi * 180, color="red", linewidth=2.0, linestyle="-.", label="q2")
    plt.grid('on')
    plt.legend()
    plt.show()


def unwrap(state):
    """unwrap the joint angles to the limited range [-pi pi]"""
    s = state
    s[0] = s[0] - 2*pi*floor((s[0] + pi)/(2*pi))
    s[1] = s[1] - 2*pi*floor((s[1] + pi)/(2*pi))
    new_s = s
    return new_s


def show_movie(robot, sol):
    x1, y1, x2, y2 = robot.forward_kinematics(sol)
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], '-o', lw=2.5)
    time_template = 'time = %.2fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i * robot.dt))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, range(1, len(sol), 2),
                                  interval=robot.dt * 1000, blit=True, init_func=init)
    plt.show()


if __name__ == '__main__':
    robot = Pendubot()
    tau = np.array([[12],
                    [0]])
    print(tau)
    t_l, sol = robot.simulation(tau)
    s_now = np.array([-pi, 0.0, 0.0, 0.0])
    s_ = robot.get_next_state(s_now, tau)
    print(s_)
    plotter(t_l, sol)
    show_movie(robot, sol)
