'''
     Langevin dynamic sampling
'''
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random


@jax.jit
def energy_func(x:jnp.ndarray):
    center0 = jnp.array([0.3, 0.3])
    center1 = jnp.array([1.2, 1.2])
    center2 = jnp.array([2.5, 2.5])

    dist0 = jnp.linalg.norm(x - center0) - 0.3
    dist1 = jnp.linalg.norm(x - center1) - 0.05
    dist2 = jnp.linalg.norm(x - center2) - 0.4
    energy = jnp.minimum(jax.nn.relu(dist0), 
                         jnp.minimum(jax.nn.relu(dist1), 
                                     jax.nn.relu(dist2))
                         )
    return energy

energy_func_grad = jax.grad(energy_func)


if __name__ == '__main__':
    from matplotlib import cm
    x = jnp.linspace(0.0, 3.0, num=200)
    y = jnp.linspace(0.0, 3.0, num=200)

    X, Y = jnp.meshgrid(x, y)
    
    key = jax.random.key(0)
    enery_map = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            en = energy_func(jnp.array([x[i], y[j]]))
            enery_map[i,j] = en

    # sns.set_theme('paper')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, enery_map, cmap=cm.coolwarm, alpha=0.3)
    ax.contour(X, Y, enery_map, zdir='z', offset=0, cmap='coolwarm')

    num_sample = 2400
    res = []
    for sample in range(num_sample):
        if random.random() < 0.95 and sample > 1:
            point = random.choice(res)
        else:
            point = np.random.uniform(0.0, 3.0, size=[2,])

        for k in range(30):
            noise = np.random.normal(0.0, 0.05, size=[2,])
            point = point - 0.02 * energy_func_grad(point) + noise
            
        res.append(point)

    ax.scatter([ele[0] for ele in res], 
               [ele[1] for ele in res], 
               len(res)*[0,], color=[0.5, 0.4, sample/num_sample])
    plt.show()
    