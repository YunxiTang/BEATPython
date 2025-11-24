import numpy as np
import plotly.graph_objects as go
import trimesh

# Load the STL file
mesh = trimesh.load_mesh('/home/yxtang/CodeBase/PythonCourse/PythonRobotics/RobotSim/isaac_sim/assets/urdf/ycb/011_banana/collision.obj')

# Extract vertices and faces
vertices = mesh.vertices
faces = mesh.faces

import plotly.graph_objects as go

# Create a Mesh3d object
mesh_plotly = go.Mesh3d(
    x=vertices[:, 0],  # x-coordinates of vertices
    y=vertices[:, 1],  # y-coordinates of vertices
    z=vertices[:, 2],  # z-coordinates of vertices
    i=faces[:, 0],     # Indices of the first vertex of each triangle
    j=faces[:, 1],     # Indices of the second vertex of each triangle
    k=faces[:, 2],     # Indices of the third vertex of each triangle
    opacity=0.7,       # Set opacity for better visualization
    # color='lightblue'  # Set mesh color
)

# Calculate the range for each axis
x_range = [np.min(vertices[:, 0]), np.max(vertices[:, 0])]
y_range = [np.min(vertices[:, 1]), np.max(vertices[:, 1])]
z_range = [np.min(vertices[:, 2]), np.max(vertices[:, 2])]

# Find the maximum range across all axes
max_range = max(
    x_range[1] - x_range[0],
    y_range[1] - y_range[0],
    z_range[1] - z_range[0]
)

# Center the ranges around the data
x_center = np.mean(x_range)
y_center = np.mean(y_range)
z_center = np.mean(z_range)

# Set equal ranges for all axes
x_range_equal = [x_center - max_range / 2, x_center + max_range / 2]
y_range_equal = [y_center - max_range / 2, y_center + max_range / 2]
z_range_equal = [z_center - max_range / 2, z_center + max_range / 2]

# Create a figure
fig = go.Figure(data=[mesh_plotly])

# Update layout for better visualization
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

# Set equal axis ranges
fig.update_layout(
    scene=dict(
        xaxis=dict(range=x_range_equal, title='X'),
        yaxis=dict(range=y_range_equal, title='Y'),
        zaxis=dict(range=z_range_equal, title='Z'),
        aspectmode='manual',  # Manually set aspect ratio
        aspectratio=dict(x=1, y=1, z=1)  # Equal aspect ratio
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

# Show the figure
fig.show()