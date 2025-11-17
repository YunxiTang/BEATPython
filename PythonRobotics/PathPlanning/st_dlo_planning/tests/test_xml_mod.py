import xml.etree.ElementTree as ET
from omegaconf import OmegaConf
import numpy as np
import zarr
import pathlib
import seaborn as sns


def hex_to_rgba(hex_code):
    """
    Convert a hex color code to an RGBA tuple.
    Supports both #RRGGBB and #RRGGBBAA formats.
    """
    hex_code = hex_code.lstrip("#")  # Remove the '#' if present

    # Determine if the hex code includes an alpha channel
    if len(hex_code) == 6:
        hex_code += "FF"  # Add default alpha (fully opaque)

    # Convert hex to integer values
    r = int(hex_code[0:2], 16)  # Red
    g = int(hex_code[2:4], 16)  # Green
    b = int(hex_code[4:6], 16)  # Blue
    a = int(hex_code[6:8], 16)  # Alpha

    # Normalize to range [0, 1] (optional, commonly used in graphics)
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    a_norm = a / 255.0

    return (r_norm, g_norm, b_norm, a_norm)  # Return normalized RGBA


h = 0.4

map_case = "camera_ready_maze2"  #'map_case8.yaml' #
cfg_path = f"/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/envs/map_cfg/{map_case}.yaml"
map_cfg_file = OmegaConf.load(cfg_path)

obstacles = map_cfg_file.obstacle_info.obstacles

print(obstacles)
# Load the XML file
file_dir = "/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/envs/planar_cable_deform/assets/"
raw_file = f"{file_dir}dual_hand_thin_03.xml"

tree = ET.parse(raw_file)
root = tree.getroot()

clrs = sns.color_palette("tab10", n_colors=max(3, len(obstacles))).as_hex()

# mocap body
worldbody = root.find("worldbody")
obstacle_body = ET.SubElement(
    worldbody, "body", name="obstacles", mocap="true", pos=f"0 0 {h}"
)
k = 0
for obstacle in obstacles:
    size_x = obstacle[0] / 2
    size_y = obstacle[1] / 2
    pos_x = obstacle[2]
    pos_y = obstacle[3]
    theta = obstacle[4] * np.pi
    rgba = hex_to_rgba(clrs[k])
    ET.SubElement(
        obstacle_body,
        "geom",
        type="box",
        name=f"obs{k}",
        contype="0",
        conaffinity="0",
        size=f"{size_x} {size_y} 0.08",
        pos=f"{pos_x} {pos_y} 0",
        euler=f"0 0 {theta}",
        rgba=f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}",
    )
    k += 1

result_path = pathlib.Path(__file__).parent.parent.joinpath(
    "results", f"{map_case}_optimal_shape_seq.npy"
)
planned_shape_seq = np.load(result_path, mmap_mode="r")
planned_shape_seq = np.copy(planned_shape_seq.reshape(-1, 13, 3))
init_dlo_shape = planned_shape_seq[0]
goal_dlo_shape = planned_shape_seq[-1]

for i in range(13):
    pos_x = init_dlo_shape[i, 0]
    pos_y = init_dlo_shape[i, 1]

    ET.SubElement(
        obstacle_body,
        "geom",
        type="sphere",
        name=f"init_kp{i}",
        contype="0",
        conaffinity="0",
        size="0.005",
        pos=f"{pos_x} {pos_y} 0",
        rgba="0 1 0 0.5",
    )

for i in range(13):
    pos_x = goal_dlo_shape[i, 0]
    pos_y = goal_dlo_shape[i, 1]

    ET.SubElement(
        obstacle_body,
        "geom",
        type="sphere",
        name=f"goal_kp{i}",
        contype="0",
        conaffinity="0",
        size="0.005",
        pos=f"{pos_x} {pos_y} 0",
        rgba="0 1 0 0.8",
    )

# Save the modified XML
mod_file = f"{file_dir}dual_hand_thin_03_mod.xml"
tree.write(mod_file)
