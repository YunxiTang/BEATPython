import warp as wp
import numpy as np

wp.init()

# 系统参数
T = 10  # 时间步数
dt = 0.1  # 时间间隔
x0 = 0.0  # 初始位置
x_goal = 1.0  # 目标位置

# 控制变量：T 个加速度值
u = wp.array(np.zeros(T, dtype=np.float32), requires_grad=True)
x = wp.zeros(T + 1, dtype=wp.float32, requires_grad=True)  # 状态轨迹
loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)


@wp.kernel
def rollout(x0: float, u: wp.array(dtype=float), x: wp.array(dtype=float)):
    tid = wp.tid()
    if tid == 0:
        x[0] = x0
    # wp.sync_threads()

    if tid < u.shape[0]:
        x[tid + 1] = x[tid] + u[tid] * dt  # 简单积分模型


@wp.kernel
def terminal_loss(x: wp.array(dtype=float), x_goal: float, loss: wp.array(dtype=float)):
    # 使用 L2 loss 到目标
    delta = x[x.shape[0] - 1] - x_goal  # 避免使用 x[-1]
    wp.atomic_add(loss, 0, delta * delta)  # 显式平方


# 自动微分求导
with wp.Tape() as tape:
    wp.launch(rollout, dim=T, inputs=[x0, u, x])
    wp.launch(terminal_loss, dim=1, inputs=[x, x_goal, loss])

    # 反向传播
    tape.backward(loss)

# 查看梯度
print("loss:", loss.numpy()[0])
print("u grad:", u.grad.numpy())
