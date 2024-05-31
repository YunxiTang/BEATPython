import jax
import jax.numpy as jnp
from daxbench.core.envs import ShapeRopeEnv
from daxbench.core.envs.shape_rope_env import DefaultConf
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
        ax.scatter(dlo[i][0], dlo[i][2], dlo[i][1], color=clr,  marker='o', s=45)
    for i in range(num_kp-1):
        ax.plot3D([dlo[i][0], dlo[i+1][0]], 
                  [dlo[i][2], dlo[i+1][2]], 
                  [dlo[i][1], dlo[i+1][1]], color=clr, linewidth=3.0)
    ax.axis('equal')


def demo():
    print(jax.devices())
    env_cfg = DefaultConf()
    env_cfg.rope_hardness = 2.5
    env_cfg.rope_width = [0.25, 0.006, 0.006]
    env = ShapeRopeEnv(batch_size=1, seed=1, conf=env_cfg)
    # env.collect_goal()
    # env.collect_expert_demo(10)
    # actions = jnp.zeros((env.batch_size, 6))
    # obs, state = env.reset(env.simulator.key)
    print("time start")
    start_time = time.time()
    for it in range(100):
        obs, state = env.reset(env.simulator.key)
        for i in range(6):
            actions = get_expert_start_end_mpm(state.x, size=512)
            print(actions)
            obs, reward, done, info = env.step_with_render(actions, state)
            state = info["state"]
            print("it", it, "step", i, time.time() - start_time)
            
        print(time.time() - start_time)


def test():
    # Create the environments
    env_cfg = DefaultConf()
    env_cfg.rope_hardness = 0.5
    env_cfg.rope_width = [0.5, 0.006, 0.006]
    env_cfg.dt = 1e-4
    env = ShapeRopeEnv(batch_size=2, seed=10, conf=env_cfg)
    obs, state = env.reset(env.simulator.key)

    # Actions to be simulated in each environment
    actions = jnp.array(
        [
            [0.4, 0, 0.4, 0.6, 0, 0.6],
            [0.2, 0, 0.5, 0.4, 0, 0.4]
        ]
    )

    obs, reward, done, info = env.step_diff(actions, state)
    next_state = info["state"]
    image = env.render(next_state, visualize=True)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal')
    ax.scatter(0.0, 0.0, 0.0, color='r',  marker='*', s=100)
    next_state = np.array(next_state[0])
    print(next_state[0, :, :].shape)
    visualize_shape(next_state[0, 0:582:30, :], ax, clr='r')
    visualize_shape(next_state[1, 0:582:30, :], ax, clr='r')
    plt.show()


def test1():
    # Create the environments
    states = []
    env_cfg = DefaultConf()
    env_cfg.rope_hardness = 0.5
    env_cfg.rope_width = [0.5, 0.006, 0.006]
    env_cfg.dt = 1e-4
    env = ShapeRopeEnv(batch_size=2, seed=10, conf=env_cfg)
    obs, state = env.reset(env.simulator.key)
    N = 10
    for _ in range(N):
        acts = env.random_policy(2, radius=0.05)
        acts[:, 1] = 0
        state, reward, done, info = env.step_diff(acts, state)
        # state, reward, done, info = self.step_diff(actions, self.state)
        state = info["state"]
        states.append(np.array(state[0]))
    
    print(len(states))
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal')
    ax.scatter(0.0, 0.0, 0.0, color='r',  marker='*', s=100)
    for idx in range(N):
        visualize_shape(states[idx][0, 0:582:30, :], ax, clr='r')
        visualize_shape(states[idx][1, 0:582:30, :], ax, clr='r')
    plt.show()


if __name__ == '__main__':
    # demo()
    test1()
