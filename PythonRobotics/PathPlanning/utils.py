import matplotlib.pyplot as plt
import jax.numpy as jnp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_circle(x, y, size, color="-b"):
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * jnp.cos(jnp.deg2rad(d)) for d in deg]
    yl = [y + size * jnp.sin(jnp.deg2rad(d)) for d in deg]
    plt.plot(xl, yl, color)


def plot_box(center, size, ax, clr='gray'):
    center_x, center_y, center_z = center
    size_x, size_y, size_z = size
    half_x = size_x / 2.
    half_y = size_y / 2.
    half_z = size_z / 2.

    faces = Poly3DCollection([
        [[center_x-half_x, center_y-half_y, center_z-half_z], 
         [center_x+half_x, center_y-half_y, center_z-half_z], 
         [center_x+half_x, center_y+half_y, center_z-half_z], 
         [center_x-half_x, center_y+half_y, center_z-half_z]
         ], # bottom
        [[center_x-half_x, center_y-half_y, center_z+half_z], 
         [center_x+half_x, center_y-half_y, center_z+half_z], 
         [center_x+half_x, center_y+half_y, center_z+half_z], 
         [center_x-half_x, center_y+half_y, center_z+half_z]
         ], # top
        [[center_x-half_x, center_y-half_y, center_z-half_z], 
         [center_x+half_x, center_y-half_y, center_z-half_z], 
         [center_x+half_x, center_y-half_y, center_z+half_z], 
         [center_x-half_x, center_y-half_y, center_z+half_z]
         ], # front
        [[center_x-half_x, center_y+half_y, center_z-half_z], 
         [center_x+half_x, center_y+half_y, center_z-half_z], 
         [center_x+half_x, center_y+half_y, center_z+half_z], 
         [center_x-half_x, center_y+half_y, center_z+half_z]
         ], # back
        [[center_x-half_x, center_y+half_y, center_z-half_z], 
         [center_x-half_x, center_y+half_y, center_z+half_z], 
         [center_x-half_x, center_y-half_y, center_z+half_z], 
         [center_x-half_x, center_y-half_y, center_z-half_z]
         ], # left
        [[center_x+half_x, center_y+half_y, center_z-half_z], 
         [center_x+half_x, center_y+half_y, center_z+half_z], 
         [center_x+half_x, center_y-half_y, center_z+half_z], 
         [center_x+half_x, center_y-half_y, center_z-half_z]
         ] # right
    ])

    faces.set_alpha(0.8)
    faces.set_facecolor(clr)
    faces.set_edgecolor('k')
    ax.add_collection3d(faces)