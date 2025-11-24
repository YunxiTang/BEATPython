import mujoco
import mujoco.viewer
from mujoco.renderer import Renderer
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

XML = """
<mujoco>
  <default>
    <geom mass=".01" solref="-1000 0"/>
  </default>
  <extension>
      <plugin plugin="mujoco.elasticity.cable"/>
  </extension>
  <visual>
    <rgba haze="0 0 0 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1=".2 .2 .2" rgb2=".3 .3 .3" markrgb=".8 .8 .8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="2 2" reflectance=".2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.0" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <camera name="camera" pos="0 -2 .5" axisangle="1 0 0 90" fovy="45"/>

    <composite type="cable" curve="s" count="41 1 1" size="1.0" offset="-.2 0 .5" initial="none">
        <plugin plugin="mujoco.elasticity.cable">
            <!--Units are in Pa (SI)-->
            <config key="twist" value="0"/>
            <config key="bend" value="4e6"/>
            <config key="vmax" value="0.05"/>
        </plugin>
        <joint kind="main" damping=".1"/>
        <geom type="capsule" size=".02" rgba=".8 .2 .1 1" condim="1"/>
    </composite>

    <body name="slider" pos=".7 0 .6">
        <joint type="slide" axis="1 0 0" damping=".2"/>
        <geom size=".02"/>
    </body>

    <body name="slider2" pos=".7 0.2 0.9">
        <joint type="slide" axis="1 0 0" damping=".2"/>
        <geom size=".02"/>
    </body>

  </worldbody>
  <equality>
      <connect name="right_boundary" body1="B_last" body2="slider" anchor=".025 0 0"/>
  </equality>
  <contact>
      <exclude body1="B_last" body2="slider"/>
  </contact>
</mujoco>
"""

def render_rgbd(renderer: mujoco.Renderer, camera_id: int) -> tuple[np.ndarray, np.ndarray]:
    renderer.update_scene(data, camera=camera_id)
    renderer.enable_depth_rendering()
    depth = np.copy(renderer.render())
    renderer.disable_depth_rendering()
    rgb = np.copy(renderer.render())
    return rgb, depth

def rgbd_to_pointcloud(rgb: np.ndarray, depth: np.ndarray, intr: np.ndarray, 
                       extr: np.ndarray, depth_trunc: float = 2.0):
    cc, rr = np.meshgrid(np.arange(width), np.arange(height), sparse=True)
    valid = (depth > 0) & (depth < depth_trunc)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (cc - intr[0, 2]) / intr[0, 0], 0)
    y = np.where(valid, z * (rr - intr[1, 2]) / intr[1, 1], 0)
    xyz = np.vstack([e.flatten() for e in [x, y, z]]).T
    color = rgb.transpose([2, 0, 1]).reshape((3, -1)).T / 255.0
    mask = np.isnan(xyz[:, 2])
    xyz = xyz[~mask]
    color = color[~mask]
    xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    xyz_t = (extr @ xyz_h.T).T
    xyzrgb = np.hstack([xyz_t[:, :3], color])
    return xyzrgb


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Define camera parameters and init renderer.
    height = 480
    width = 640
    fps = 30
    camera_id = model.cam("camera").id
    renderer = Renderer(model, height=height, width=width)

    # Intrinsic matrix.
    fov = model.cam_fovy[camera_id]
    theta = np.deg2rad(fov)
    fx = width / 2 / np.tan(theta / 2)
    fy = height / 2 / np.tan(theta / 2)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    intr = np.array([[fx, 0, cx], 
                     [0, fy, cy], 
                     [0, 0, 1]])

    # Extrinsic matrix.
    cam_pos = data.cam_xpos[camera_id]
    cam_rot = data.cam_xmat[camera_id].reshape(3, 3)
    extr = np.eye(4)
    extr[:3, :3] = cam_rot.T
    extr[:3, 3] = cam_pos

    # Simulate for 10 seconds and capture RGB-D images at fps Hz.
    xyzrgbs: list[np.ndarray] = []
    mujoco.mj_resetData(model, data)
    while data.time < 10.0:
        mujoco.mj_step(model, data)
        if len(xyzrgbs) < data.time * fps:
            rgb, depth = render_rgbd(renderer, camera_id)
            xyzrgb = rgbd_to_pointcloud(rgb, depth, intr, extr)
            xyzrgbs.append(xyzrgb)
    renderer.close()

    # Visualize in open3d.
    print(xyzrgbs[200].shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgbs[0][:, :3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrgbs[0][:, 3:])
    o3d.visualization.draw_geometries([pcd])

    # Perform DBSCAN clustering
    points = xyzrgbs[0][:, :3]
    dbscan = DBSCAN(eps=0.1, min_samples=10)  # Set parameters as needed
    labels = dbscan.fit_predict(points)
    unique_labels = set(labels)
    colors = plt.get_cmap("tab10", len(unique_labels))
    # Assign colors based on cluster labels
    colored_points = np.zeros((points.shape[0], 3))  # Initialize an array for colored points

    for k in unique_labels:
        if k == -1:
            # Noise points are colored black
            colored_points[labels == k] = [0, 0, 0]  # Black for noise
        else:
            colored_points[labels == k] = colors(k)[:3]  # Use color map for clusters

    # Assign colors to the point cloud
    # Create an Open3D PointCloud objects
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colored_points)

    # Visualize the clustered point cloud
    o3d.visualization.draw_geometries([point_cloud], window_name="Clustered Point Cloud")