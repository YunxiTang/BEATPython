# signed distance function
import jax.numpy as jnp
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import jax

pi = jnp.pi

@jax.jit
def sdf_sphere(p: jnp.ndarray, geom_param: dict) -> float:
    """
        signed distance function of sphere
    """
    geom_center = geom_param.get('sphere_center', None)
    geom_r = geom_param.get('sphere_size', None)
    sdf_value = jnp.linalg.norm(p-geom_center) - geom_r
    return sdf_value


@jax.jit
def sdf_2d_box(p: jnp.ndarray, geom_param: dict) -> float:
    transform = geom_param.get('box_transform', None)
    b = jnp.array( geom_param.get('box_size', None) )
    p_homo = jnp.array([p[0], p[1], 1])
    p = jnp.linalg.inv(transform) @ p_homo
    d = jnp.abs(p[0:2]) - b 
    return jnp.linalg.norm(jnp.maximum(d, 0.0)) + jnp.minimum(jnp.maximum(d[0], d[1]), 0.0)
    

if __name__ == '__main__':
    num = 50
    xs = jnp.linspace(-3.0, 3.0, num)
    ys = jnp.linspace(-3.0, 3.0, num)
    
    trans = jnp.array(
        [[jnp.cos(0. * pi / 4.), -jnp.sin(0.0 * pi / 4.), 0.0],
         [jnp.sin(0. * pi / 4.), jnp.cos(0. * pi / 4.), 0.0],
         [0, 0, 1]]
        )
    
    box = {
        'box_transform': trans,
        'box_size': [0.5, 0.2]
        }
    
    ball = {
        'sphere_center': jnp.array([0.0, 0.0]),
        'sphere_size': 0.1
    }
    
    
    sdf_map = []
    for i in range(num):
        for j in range(num):
            p = jnp.array([xs[i], ys[j]])
            sdf_map.append(sdf_2d_box(p, box))
    sdf_map = jnp.array(sdf_map).reshape(num, num).transpose()
    sns.set()
    ax = sns.heatmap(sdf_map)
    plt.grid()
    plt.axis('equal')
    plt.show()
    