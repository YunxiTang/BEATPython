import warp as wp
import warp.sim
import warp.sim.render
import numpy as np

wp.init()

# ====== 核函数部分 ======


@wp.kernel
def apply_gravity(
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    gravity: wp.vec3,
    dt: float,
):
    i = wp.tid()
    if inv_mass[i] > 0.0:
        vel[i] += gravity * dt


@wp.kernel
def predict_position(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    pred: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    pred[i] = pos[i] + vel[i] * dt


@wp.kernel
def solve_distance_constraint(
    pred: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    indices: wp.array(dtype=int),
    rest_lengths: wp.array(dtype=float),
    stiffness: float,
    dt: float,
):
    i = wp.tid()
    i0 = indices[i * 2 + 0]
    i1 = indices[i * 2 + 1]

    p0 = pred[i0]
    p1 = pred[i1]

    w0 = inv_mass[i0]
    w1 = inv_mass[i1]

    d = p1 - p0
    l = wp.length(d)
    if l > 1e-6:
        n = d / l
        c = l - rest_lengths[i]
        alpha = stiffness / (dt * dt)
        denom = w0 + w1 + alpha
        lambda_c = -c / denom

        dp0 = lambda_c * w0 * n
        dp1 = -lambda_c * w1 * n

        pred[i0] += dp0
        pred[i1] += dp1


@wp.kernel
def update_velocity(
    pos: wp.array(dtype=wp.vec3),
    pred: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    vel[i] = (pred[i] - pos[i]) / dt
    pos[i] = pred[i]


# ====== XPBD 初始化参数 ======

num_points = 30
spacing = 0.05
stiffness = 1.0
gravity = wp.vec3(0.0, -9.8, 0.0)
dt = 1.0 / 60.0
iters = 10
steps = 3000

device = "cpu"

# 初始化位置、速度、质量
init_pos = [wp.vec3(i * spacing, 0.0, 0.0) for i in range(num_points)]
inv_mass_host = [1.0 for _ in range(num_points)]
inv_mass_host[0] = 0.0  # 固定端点

# 初始位置和状态变量
pos = wp.array(init_pos, dtype=wp.vec3, device=device)
vel = wp.zeros(num_points, dtype=wp.vec3, device=device)
pred = wp.zeros_like(pos)
inv_mass = wp.array(inv_mass_host, dtype=float, device=device)

# 距离约束（相邻点）
indices_host = []
rest_lengths_host = []
for i in range(num_points - 1):
    indices_host += [i, i + 1]
    rest_lengths_host.append(spacing)

indices = wp.array(indices_host, dtype=int, device=device)
rest_lengths = wp.array(rest_lengths_host, dtype=float, device=device)

# ====== 构造 ModelBuilder 用于可视化 ======

builder = wp.sim.ModelBuilder()
for i in range(num_points):
    builder.add_particle(
        pos=init_pos[i],
        vel=(0.0, 0.0, 0.0),
        mass=0.0 if inv_mass_host[i] == 0.0 else 1.0,
    )

model = builder.finalize(device=device)
state = model.state()
renderer = wp.sim.render.SimRendererUsd(model, path="xpbd_rope.usd")

# ====== 仿真 + 可视化循环 ======

for step in range(steps):
    # 重力
    wp.launch(
        apply_gravity,
        dim=num_points,
        inputs=[vel, inv_mass, gravity, dt],
        device=device,
    )

    # 预测
    wp.launch(
        predict_position, dim=num_points, inputs=[pos, vel, pred, dt], device=device
    )

    # 多次迭代满足约束
    for _ in range(iters):
        wp.launch(
            solve_distance_constraint,
            dim=num_points - 1,
            inputs=[pred, inv_mass, indices, rest_lengths, stiffness, dt],
            device=device,
        )

    # 更新速度 & 位置
    wp.launch(
        update_velocity, dim=num_points, inputs=[pos, pred, vel, dt], device=device
    )

    # 将位置写入 state 以便渲染器访问
    state.particle_q = pos

    renderer.begin_frame(step * dt)
    renderer.render(state)
    renderer.end_frame()
renderer.save()
print("XPBD 绳子仿真完成 🎉")
