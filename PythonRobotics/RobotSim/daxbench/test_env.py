import jax
import jax.numpy as jnp
from daxbench.core.envs import ShapeRopeEnv
import numpy as np
import matplotlib.pyplot as plt
from daxbench.core.utils.util import get_expert_start_end_mpm
import time


def visualize_shape(dlo: np.ndarray, ax, clr=None):
    '''
        visualize a rope shape
    '''
    if clr is None:
        clr = 0.5 + 0.5 * np.random.random(3)

    num_kp = dlo.shape[0]

    for i in range(num_kp):
        ax.scatter(dlo[i][0], dlo[i][1], dlo[i][2], color=clr,  marker='o', s=45)
    for i in range(num_kp-1):
        ax.plot3D([dlo[i][0], dlo[i+1][0]], 
                  [dlo[i][1], dlo[i+1][1]], 
                  [dlo[i][2], dlo[i+1][2]], color=clr, linewidth=3.0)
    ax.axis('equal')


def demo():
    print(jax.devices())
    env = ShapeRopeEnv(batch_size=1, seed=1)
    # env.collect_goal()
    # env.collect_expert_demo(10)
    # actions = jnp.zeros((env.batch_size, 6))
    # obs, state = env.reset(env.simulator.key)
    print("time start")
    start_time = time.time()
    for it in range(100):
        # state = env.auto_reset(env.init_state, state, state.key)
        obs, state = env.reset(env.simulator.key)
        for i in range(6):
            actions = get_expert_start_end_mpm(state.x, size=512)
            # obs, reward, done, info = env.step_diff(actions, state)
            obs, reward, done, info = env.step_with_render(actions, state)
            state = info["state"]
            print("it", it, "step", i, time.time() - start_time)
            # print("iou", calc_IOU(state.x[0], env.conf.goal_path))
            # print("reward", reward)
        print(time.time() - start_time)


if __name__ == '__main__':
    # Crreate the environments
    env = ShapeRopeEnv(batch_size=1, seed=1)
    obs, state = env.reset(env.simulator.key)

    # Actions to be simulated in each environment
    actions = jnp.array(
        [
            [0.4, 0, 0.4, 0.6, 0, 0.6],
        ]
    )

    obs, reward, done, info = env.step_diff(actions, state)
    next_state = info["state"]
    # image = env.render(next_state, visualize=True)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal')
    ax.scatter(0.0, 0.0, 0.0, color='r',  marker='*', s=100)
    next_state = np.array(next_state)
    print(next_state.shape)
    # visualize_shape(next_state[0:50:-1,:], ax, clr='r')
