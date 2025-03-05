import mujoco
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np
import itertools
from PIL import Image
import open3d as o3d

def compute_camera_matrix(renderer, data):
    """Returns the 3x4 camera matrix."""
    # If the camera is a 'free' camera, we get its position and orientation
    # from the scene data structure. It is a stereo camera, so we average over
    # the left and right channels. Note: we call `self.update()` in order to
    # ensure that the contents of `scene.camera` are correct.
    renderer.update_scene(data)
    pos = np.mean([camera.pos for camera in renderer.scene.camera], axis=0)
    z = -np.mean([camera.forward for camera in renderer.scene.camera], axis=0)
    y = np.mean([camera.up for camera in renderer.scene.camera], axis=0)
    rot = np.vstack((np.cross(y, z), y, z))
    fov = model.vis.global_.fovy

    # Translation matrix (4x4).
    translation = np.eye(4)
    translation[0:3, 3] = -pos

    # Rotation matrix (4x4).
    rotation = np.eye(4)
    rotation[0:3, 0:3] = rot

    # Focal transformation matrix (3x4).
    focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * renderer.height / 2.0
    focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

    # Image matrix (3x3).
    image = np.eye(3)
    image[0, 2] = (renderer.width - 1) / 2.0
    image[1, 2] = (renderer.height - 1) / 2.0
    return image @ focal @ rotation @ translation


xml = """
<mujoco model="Cable">
    <extension>
        <plugin plugin="mujoco.elasticity.cable"/>
    </extension>

    <statistic center="0 0 .3" extent="1"/>

    <visual>
        <global elevation="-30"/>
    </visual>

    <compiler autolimits="true"/>

    <size memory="2M"/>

    <worldbody>
        <composite type="cable" curve="s" count="41 1 1" size="1" offset="-.2 0 .6" initial="none">
        <plugin plugin="mujoco.elasticity.cable">
            <!--Units are in Pa (SI)-->
            <config key="twist" value="1e7"/>
            <config key="bend" value="4e6"/>
            <config key="vmax" value="0.05"/>
        </plugin>
        <joint kind="main" damping=".015"/>
        <geom type="capsule" size=".005" rgba=".8 .2 .1 1" condim="1"/>
        </composite>
        <body name="slider" pos=".7 0 .6">
        <joint type="slide" axis="1 0 0" damping=".1"/>
        <geom size=".01"/>
        </body>
    </worldbody>
    <equality>
        <connect name="right_boundary" body1="B_last" body2="slider" anchor=".025 0 0"/>
    </equality>
    <contact>
        <exclude body1="B_last" body2="slider"/>
    </contact>
    <actuator>
        <motor site="S_last" gear="0 0 0 1 0 0" ctrlrange="-.03 .03"/>
    </actuator>
    </mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.Renderer(model) as renderer:
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    rgb_img = renderer.render()
    img = Image.fromarray(rgb_img)
    # img.show('rgb')

with mujoco.Renderer(model) as renderer:
    renderer.enable_depth_rendering() # update renderer to render depth
    renderer.update_scene(data) # reset the scene
    depth = renderer.render() # depth is a float array, in meters.

    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    # Assign the points to the PointCloud object
    # point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)

    depth -= depth.min() # Shift nearest values to the origin.
    depth /= 2*depth[depth <= 1].mean() # Scale by 2 mean distances of near rays.
    pixels = 255*np.clip(depth, 0, 1) # Scale to [0, 255]
    img = Image.fromarray(pixels.astype(np.uint8))
    img.show('depth')

with mujoco.Renderer(model) as renderer:
    renderer.disable_depth_rendering()

    # update renderer to render segmentation
    renderer.enable_segmentation_rendering()

    # reset the scene
    renderer.update_scene(data)

    seg = renderer.render()

    # Display the contents of the first channel, which contains object
    # IDs. The second channel, seg[:, :, 1], contains object types.
    geom_ids = seg[:, :, 0]
    # Infinity is mapped to -1
    geom_ids = geom_ids.astype(np.float64) + 1
    # Scale to [0, 1]
    geom_ids = geom_ids / geom_ids.max()
    pixels = 255*geom_ids
    img = Image.fromarray(pixels)
    img.show('segmentation')