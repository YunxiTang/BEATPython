from Robot import Pendubot
from Robot import dynamics
import numpy as np
from Network import DeLaN
from trajectory_generation import Trajectory
from math import pi

import torch
import torch.optim as opt
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import torch.utils.data as Data

import torch.utils.data as Data

MAX_TORQUE = 20
BATCH_SIZE = 20

robot = Pendubot()
Model_net = DeLaN(2, 950)
des_tra = Trajectory(2, 1)

# hyper parameters
RUN_TIME = 100
TRAIN_STEP = 2000
LR = 0.001
optimizer = opt.Adam(Model_net.parameters(), lr=LR, weight_decay=0.01)
loss_func = nn.MSELoss()

Kp = np.array([[250, 0],
               [0, 250]])
Kd = np.array([[70, 0],
               [0, 70]])
current_time = 0
Time = []
dt = robot.dt
Loss = []
Q1 = []
Q2 = []
Qd1 = []
Qd2 = []

Qs1 = []
Qs2 = []
Qds1 = []
Qds2 = []

E = []
Ed = []
# initial state
state = robot.initial_state

# train the model
while True:
    q = state[0:2]
    qd = state[2:]

    q_s, qd_s, qdd_s = des_tra.trajectory(current_time)
    # print(torch.tensor(qd_s).reshape(2,))
    e = q_s - q
    ed = qd_s - qd
    # PD Controller
    tau_1 = np.matmul(Kp, e) + np.matmul(Kd, ed)
    # inverse dynamics
    tau_0 = Model_net.foward(torch.tensor(q_s, dtype=torch.float, requires_grad=True).reshape(2,),
                             torch.tensor(qd_s, dtype=torch.float).reshape(2, ),
                             torch.tensor(qdd_s, dtype=torch.float).reshape(2, ))
    tau = torch.add(tau_0, torch.tensor(tau_1, dtype=torch.float))

    # tau = sat(tau.detach().numpy(), MAX_TORQUE)
    sd = dynamics(current_time, state, tau.detach().numpy(), robot)
    qdd = torch.tensor(sd[2:], dtype=torch.float)
    tau_hat = Model_net.foward(torch.tensor(q, dtype=torch.float, requires_grad=True).reshape(2,),
                               torch.tensor(qd, dtype=torch.float).reshape(2,),
                               qdd.reshape(2,))

    loss = loss_func(tau_hat, tau.clone().detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    next_state = robot.get_next_state(state, tau.detach())

    Time.append(current_time)
    Loss.append(loss)
    Q1.append(state[0])
    Q2.append(state[1])
    Qd1.append(state[2])
    Qd2.append(state[3])
    Qs1.append(q_s[0])
    Qs2.append(q_s[1])
    Qds1.append(qd_s[0])
    Qds2.append(qd_s[1])
    E.append(e/pi*180)
    Ed.append(ed/pi*180)

    current_time = current_time + dt
    state = next_state

    print("Current Time: ", current_time, "Loss :", loss.detach().numpy())
    print('Tracking Errors:', 'e:', e, 'ed:', ed)
    if current_time > 5:
        break

trained_model = torch.save(Model_net, 'Model.pkl')

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(Time, Loss, color="blue", linewidth=2.0, linestyle="-", label="Loss")
plt.grid('on')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(Time, Q1, color="red", linewidth=2.0, linestyle="-", label="q1")
plt.plot(Time, Q2, color="green", linewidth=2.0, linestyle="-", label="q2")
plt.plot(Time, Qs1, color="red", linewidth=2.0, linestyle="-.", label="qs1")
plt.plot(Time, Qs2, color="green", linewidth=2.0, linestyle="-.", label="qs2")
plt.grid('on')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(Time, Qd1, color="red", linewidth=2.0, linestyle="-", label="qd1")
plt.plot(Time, Qd2, color="green", linewidth=2.0, linestyle="-", label="qd2")
plt.plot(Time, Qds1, color="red", linewidth=2.0, linestyle="-.", label="qds1")
plt.plot(Time, Qds2, color="green", linewidth=2.0, linestyle="-.", label="qds2")
plt.grid('on')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(Time, E, color="red", linewidth=2.0, linestyle="-", label='e')
plt.grid('on')
plt.legend()

plt.show()






