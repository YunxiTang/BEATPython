import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_box(center, size, ax):
    center_x, center_y, center_z = center
    size_x, size_y, size_z = size
    half_x = size_x / 2.0
    half_y = size_y / 2.0
    half_z = size_z / 2.0

    faces = Poly3DCollection(
        [
            [
                [center_x - half_x, center_y - half_y, center_z - half_z],
                [center_x + half_x, center_y - half_y, center_z - half_z],
                [center_x + half_x, center_y + half_y, center_z - half_z],
                [center_x - half_x, center_y + half_y, center_z - half_z],
            ],  # bottom
            [
                [center_x - half_x, center_y - half_y, center_z + half_z],
                [center_x + half_x, center_y - half_y, center_z + half_z],
                [center_x + half_x, center_y + half_y, center_z + half_z],
                [center_x - half_x, center_y + half_y, center_z + half_z],
            ],  # top
            [
                [center_x - half_x, center_y - half_y, center_z - half_z],
                [center_x + half_x, center_y - half_y, center_z - half_z],
                [center_x + half_x, center_y - half_y, center_z + half_z],
                [center_x - half_x, center_y - half_y, center_z + half_z],
            ],  # front
            [
                [center_x - half_x, center_y + half_y, center_z - half_z],
                [center_x + half_x, center_y + half_y, center_z - half_z],
                [center_x + half_x, center_y + half_y, center_z + half_z],
                [center_x - half_x, center_y + half_y, center_z + half_z],
            ],  # back
            [
                [center_x - half_x, center_y + half_y, center_z - half_z],
                [center_x - half_x, center_y + half_y, center_z + half_z],
                [center_x - half_x, center_y - half_y, center_z + half_z],
                [center_x - half_x, center_y - half_y, center_z - half_z],
            ],  # left
            [
                [center_x + half_x, center_y + half_y, center_z - half_z],
                [center_x + half_x, center_y + half_y, center_z + half_z],
                [center_x + half_x, center_y - half_y, center_z + half_z],
                [center_x + half_x, center_y - half_y, center_z - half_z],
            ],  # right
        ]
    )

    faces.set_alpha(0.5)
    faces.set_facecolor("gray")
    ax.add_collection3d(faces)


ax = plt.axes(projection="3d")
plot_box(center=[100, 100, 70], size=[20, 40, 140], ax=ax)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

lim = 200
ax.set_xlim([0, lim])
ax.set_ylim([0, lim])
ax.set_zlim([0, lim])

plt.show()
