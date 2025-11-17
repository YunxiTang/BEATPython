import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import fcl
from typing import Callable
from itertools import combinations
from typing import NamedTuple
from dataclasses import dataclass
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation as R

from st_dlo_planning.utils.pytorch_utils import from_numpy

import seaborn as sns
from omegaconf import OmegaConf
import xml.etree.ElementTree as ET
import pathlib


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


def load_mapCfg_to_mjcf(map_case: str, init_dlo_shape, goal_dlo_shape):
    h = 0.4

    cfg_path = f"/home/yxtang/CodeBase/PythonCourse/PythonRobotics/PathPlanning/st_dlo_planning/envs/map_cfg/{map_case}.yaml"
    map_cfg = OmegaConf.load(cfg_path)

    obstacles = map_cfg.obstacle_info.obstacles

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

    ET.SubElement(
        obstacle_body,
        "geom",
        type="box",
        name="wall_left",
        contype="0",
        conaffinity="0",
        size=f"{map_cfg.workspace.robot_size / 2} {map_cfg.workspace.map_ymax / 2} 0.02",
        pos=f"{-map_cfg.workspace.robot_size / 2} {map_cfg.workspace.map_ymax / 2} 0",
        rgba="0.5 0.5 0.8 0.3",
    )
    ET.SubElement(
        obstacle_body,
        "geom",
        type="box",
        name="wall_right",
        contype="0",
        conaffinity="0",
        size=f"{map_cfg.workspace.robot_size / 2} {map_cfg.workspace.map_ymax / 2} 0.02",
        pos=f"{map_cfg.workspace.map_ymax + map_cfg.workspace.robot_size / 2} {map_cfg.workspace.map_ymax / 2} 0",
        rgba="0.5 0.5 0.8 0.3",
    )
    ET.SubElement(
        obstacle_body,
        "geom",
        type="box",
        name="wall_down",
        contype="0",
        conaffinity="0",
        size=f"{map_cfg.workspace.map_xmax / 2} {map_cfg.workspace.robot_size / 2} 0.02",
        pos=f"{map_cfg.workspace.map_xmax / 2} {-map_cfg.workspace.robot_size / 2} 0",
        rgba="0.5 0.5 0.8 0.3",
    )
    ET.SubElement(
        obstacle_body,
        "geom",
        type="box",
        name="wall_up",
        contype="0",
        conaffinity="0",
        size=f"{map_cfg.workspace.map_xmax / 2} {map_cfg.workspace.robot_size / 2} 0.02",
        pos=f"{map_cfg.workspace.map_xmax / 2} {map_cfg.workspace.map_ymax + map_cfg.workspace.robot_size / 2} 0",
        rgba="0.5 0.5 0.8 0.3",
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
            size=f"{size_x} {size_y} 0.02",
            pos=f"{pos_x} {pos_y} 0",
            euler=f"0 0 {theta}",
            rgba=f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}",
        )
        k += 1

    # for i in range(13):
    #     pos_x = init_dlo_shape[i, 0]
    #     pos_y = init_dlo_shape[i, 1]

    #     ET.SubElement(obstacle_body, "geom",
    #                 type="sphere", name=f'init_kp{i}', contype="0", conaffinity="0",
    #                 size="0.005", pos=f"{pos_x} {pos_y} 0",
    #                 rgba='0 1 0 0.5')

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
            rgba="0 0.5 1 0.8",
        )

    # Save the modified XML
    mod_file = f"{file_dir}dual_hand_thin_03_mod.xml"
    tree.write(mod_file)


def lse_fn(x, beta=1.0):
    exp_tmp = np.exp(beta * x)
    sum_tmp = np.sum(exp_tmp)
    log_tmp = np.log(sum_tmp)
    return 1.0 / beta * log_tmp


def draw_fake_eef(eef_state, sz, ax):
    x = eef_state[0]
    y = eef_state[1]
    theta = eef_state[2]


class Point(NamedTuple):
    """
    2D point
    """

    x: float
    y: float

    def to_list(self):
        return [self.x, self.y]

    def to_numpy(self):
        return np.array([self.x, self.y])


class Passage(NamedTuple):
    """
    Passage in Passage Map
    """

    vrtx1: list
    vrtx2: list
    min_dist: float


def line(p1: Point, p2: Point):
    A = p1.y - p2.y
    B = p2.x - p1.x
    C = p1.x * p2.y - p2.x * p1.y
    return A, B, -C


def get_intersection_point(p1: Point, q1: Point, p2: Point, q2: Point):
    x1, y1 = p1.x, p1.y
    x2, y2 = q1.x, q1.y
    x3, y3 = p2.x, p2.y
    x4, y4 = q2.x, q2.y
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

    if denom == 0:  # parallel
        return None
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    if ua < 0 or ua > 1:  # out of range
        return None
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if ub < 0 or ub > 1:  # out of range
        return None
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return Point(x, y)


def get_orientation(p: Point, q: Point, r: Point):
    """
    get the orientation of an ordered triplet (p, q, r)
    Return:
        0 : Collinear points
        1 : Clockwise points
        2 : Counterclockwise points
    See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ for details.
    """
    val = ((r.x - q.x) * (q.y - p.y)) - ((r.y - q.y) * (q.x - p.x))
    if val > 0.0:
        # Clockwise orientation
        return 1
    elif val < 0.0:
        # Counterclockwise orientation
        return 2
    else:
        # Collinear orientation
        return 0


def check_on_segment(p: Point, q: Point, r: Point):
    """
    Given three collinear points (p, q, r),
    check if q lies on line segment <p, r>
    """
    if (
        (q.x <= max(p.x, r.x))
        and (q.x >= min(p.x, r.x))
        and (q.y <= max(p.y, r.y))
        and (q.y >= min(p.y, r.y))
    ):
        return True
    return False


def check_intersection(p1: Point, q1: Point, p2: Point, q2: Point):
    """
    returns true if the line segment <p1, q1> and <p2, q2> intersect.
    Find the four orientations required for the general and special cases
    """
    o1 = get_orientation(p1, q1, p2)
    o2 = get_orientation(p1, q1, q2)
    o3 = get_orientation(p2, q2, p1)
    o4 = get_orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and check_on_segment(p1, p2, q1):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and check_on_segment(p1, q2, q1):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and check_on_segment(p2, p1, q2):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and check_on_segment(p2, q1, q2):
        return True

    # if none of the cases
    return False


def plot_rectangle(
    size_x, size_y, pos_x, pos_y, ax, angle: float = 0.0, center=True, color="b"
):
    if center:
        angle = angle * 180 / np.pi
        ax.add_patch(
            Rectangle(
                [pos_x, pos_y],
                size_x / 2.0,
                size_y / 2.0,
                angle=angle,
                rotation_point="xy",
                color=color,
            )
        )
        ax.add_patch(
            Rectangle(
                (pos_x, pos_y),
                -size_x / 2.0,
                size_y / 2.0,
                angle=angle,
                rotation_point="xy",
                color=color,
            )
        )
        ax.add_patch(
            Rectangle(
                (pos_x, pos_y),
                -size_x / 2.0,
                -size_y / 2.0,
                angle=angle,
                rotation_point="xy",
                color=color,
            )
        )
        ax.add_patch(
            Rectangle(
                (pos_x, pos_y),
                size_x / 2.0,
                -size_y / 2.0,
                angle=angle,
                rotation_point="xy",
                color=color,
            )
        )
    else:
        pass


def plot_circle(x, y, radius, ax, color="-b"):
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + radius * jnp.cos(jnp.deg2rad(d)) for d in deg]
    yl = [y + radius * jnp.sin(jnp.deg2rad(d)) for d in deg]
    ax.plot(xl, yl, color)


class Block:
    """
    city building block
    """

    def __init__(
        self,
        size_x,
        size_y,
        size_z,
        pos_x,
        pos_y,
        pos_z=None,
        angle: float = 0.0,
        clr=None,
        is_wall: bool = False,
    ):
        self._size_x = size_x
        self._size_y = size_y
        self._size_z = size_z

        self._pos_x = pos_x
        self._pos_y = pos_y
        self._pos_z = pos_z if pos_z is not None else size_z / 2.0

        self.angle = angle
        self.rotation = R.from_matrix(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1.0],
            ]
        ).as_matrix()

        # collision property
        self.geom = fcl.Box(self._size_x, self._size_y, self._size_z)
        T = np.array([self._pos_x, self._pos_y, self._pos_z])
        self.tf = fcl.Transform(self.rotation, T)
        # tf = fcl.Transform(T)
        self._collision_obj = fcl.CollisionObject(self.geom, self.tf)

        self._wall = is_wall

        if self._wall:
            # visualization property
            self._color = "gray"
        else:
            self._color = clr

    @staticmethod
    def get_normal_vector(p1, p2):
        delta = p2 - p1
        return np.array([-delta[1], delta[0]])

    def get_2d_sdf_val(self, query_point):
        # get vertices of the box in x-y plane in clockwise order.
        tf = self.rotation
        obs_center = np.array([self._pos_x, self._pos_y, 0.0])
        vertex1_o = np.array([-self._size_x / 2, self._size_y / 2, 0.0])
        vertex2_o = np.array([self._size_x / 2, self._size_y / 2, 0.0])
        vertex3_o = np.array([self._size_x / 2, -self._size_y / 2, 0.0])
        vertex4_o = np.array([-self._size_x / 2, -self._size_y / 2, 0.0])

        vertex1 = obs_center + tf @ vertex1_o
        vertex2 = obs_center + tf @ vertex2_o
        vertex3 = obs_center + tf @ vertex3_o
        vertex4 = obs_center + tf @ vertex4_o

        vertex1 = vertex1[0:2]
        vertex2 = vertex2[0:2]
        vertex3 = vertex3[0:2]
        vertex4 = vertex4[0:2]

        # compute the normal direction vector
        n12 = self.get_normal_vector(vertex1, vertex2)
        n23 = self.get_normal_vector(vertex2, vertex3)
        n34 = self.get_normal_vector(vertex3, vertex4)
        n41 = self.get_normal_vector(vertex4, vertex1)

        sdf1 = np.dot(query_point - vertex1, n12 / np.linalg.norm(n12))
        sdf2 = np.dot(query_point - vertex2, n23 / np.linalg.norm(n23))
        sdf3 = np.dot(query_point - vertex3, n34 / np.linalg.norm(n34))
        sdf4 = np.dot(query_point - vertex4, n41 / np.linalg.norm(n41))

        # print(np.max([sdf1, sdf2, sdf3, sdf4]))

        sdf_val = np.max(
            [sdf1, sdf2, sdf3, sdf4]
        )  # lse_fn(np.array([sdf1, sdf2, sdf3, sdf4]), beta=500.)

        return sdf_val, [vertex1, vertex2, vertex3, vertex4]


@dataclass
class MapCfg:
    resolution: float = 0.1
    map_xmin: float = 0.0
    map_xmax: float = 200.0
    map_ymin: float = 0.0
    map_ymax: float = 200.0

    map_zmin: float = 99.9
    map_zmax: float = 100.1

    robot_size: float = 0.1

    dim: int = 3


class WorldMap:
    """
    the world map of tasks
    """

    def __init__(self, map_cfg: MapCfg):
        self._obstacle = []
        self.map_cfg = map_cfg
        self._resolution = map_cfg.resolution
        self._s = fcl.Sphere(map_cfg.robot_size)

        self._finalized = False

        # add walls
        wall_left = Block(
            map_cfg.robot_size,
            map_cfg.map_ymax - map_cfg.map_ymin,
            map_cfg.map_zmax,
            map_cfg.map_xmin - map_cfg.robot_size / 2,
            (map_cfg.map_ymax + map_cfg.map_ymin) / 2,
            is_wall=True,
        )

        wall_right = Block(
            map_cfg.robot_size,
            map_cfg.map_ymax - map_cfg.map_ymin,
            map_cfg.map_zmax,
            map_cfg.map_xmax + map_cfg.robot_size / 2,
            (map_cfg.map_ymax + map_cfg.map_ymin) / 2,
            is_wall=True,
        )

        wall_down = Block(
            map_cfg.map_xmax - map_cfg.map_xmin,
            map_cfg.robot_size,
            map_cfg.map_zmax,
            (map_cfg.map_xmax + map_cfg.map_xmin) / 2,
            map_cfg.map_ymin - map_cfg.robot_size / 2,
            is_wall=True,
        )

        wall_up = Block(
            map_cfg.map_xmax - map_cfg.map_xmin,
            map_cfg.robot_size,
            map_cfg.map_zmax,
            (map_cfg.map_xmax + map_cfg.map_xmin) / 2,
            map_cfg.map_ymax + map_cfg.robot_size / 2,
            is_wall=True,
        )

        self.add_obstacle(wall_left)
        self.add_obstacle(wall_right)
        self.add_obstacle(wall_down)
        self.add_obstacle(wall_up)

        # passages
        self._passages = []

    def add_obstacle(self, obstacle: Block):
        self._obstacle.append(obstacle)

    def finalize(self):
        self._collision_instances = [
            obstacle._collision_obj for obstacle in self._obstacle
        ]

        self._collision_manager = fcl.DynamicAABBTreeCollisionManager()
        self._collision_manager.registerObjects(self._collision_instances)
        self._collision_manager.setup()

        self._finalized = True

        self.construct_passage()
        self.filter_passage()

        return None

    def get_obs_vertixs(self):
        obs_vrtxs = []
        for obs in self._obstacle:
            tf = obs.rotation
            obs_center = np.array([obs._pos_x, obs._pos_y, 0.0])
            obs_center = np.array([obs._pos_x, obs._pos_y, 0.0])
            _size_x = obs._size_x + 0.02
            _size_y = obs._size_y + 0.02
            vertex1_o = np.array([-_size_x / 2, _size_y / 2, 0.0])
            vertex2_o = np.array([_size_x / 2, _size_y / 2, 0.0])
            vertex3_o = np.array([_size_x / 2, -_size_y / 2, 0.0])
            vertex4_o = np.array([-_size_x / 2, -_size_y / 2, 0.0])

            vertex1 = obs_center + tf @ vertex1_o
            vertex2 = obs_center + tf @ vertex2_o
            vertex3 = obs_center + tf @ vertex3_o
            vertex4 = obs_center + tf @ vertex4_o

            vertex1 = vertex1[0:2][None]
            vertex2 = vertex2[0:2][None]
            vertex3 = vertex3[0:2][None]
            vertex4 = vertex4[0:2][None]

            obs_vrtxs.append(
                np.concatenate([vertex1, vertex2, vertex3, vertex4], axis=0)
            )
        return obs_vrtxs

    def get_obs_tensor_info(self, device):
        obs_info = []
        for obs in self._obstacle:
            tf = obs.rotation
            obs_center = np.array([obs._pos_x, obs._pos_y, 0.0])
            _size_x = obs._size_x + 0.02
            _size_y = obs._size_y + 0.02
            vertex1_o = np.array([-_size_x / 2, _size_y / 2, 0.0])
            vertex2_o = np.array([_size_x / 2, _size_y / 2, 0.0])
            vertex3_o = np.array([_size_x / 2, -_size_y / 2, 0.0])
            vertex4_o = np.array([-_size_x / 2, -_size_y / 2, 0.0])

            vertex1 = obs_center + tf @ vertex1_o
            vertex2 = obs_center + tf @ vertex2_o
            vertex3 = obs_center + tf @ vertex3_o
            vertex4 = obs_center + tf @ vertex4_o

            vertex1 = vertex1[0:2]
            vertex2 = vertex2[0:2]
            vertex3 = vertex3[0:2]
            vertex4 = vertex4[0:2]

            # compute the normal direction vector
            n12 = from_numpy(obs.get_normal_vector(vertex1, vertex2), device)
            n23 = from_numpy(obs.get_normal_vector(vertex2, vertex3), device)
            n34 = from_numpy(obs.get_normal_vector(vertex3, vertex4), device)
            n41 = from_numpy(obs.get_normal_vector(vertex4, vertex1), device)

            vertex1 = from_numpy(vertex1[0:2], device)
            vertex2 = from_numpy(vertex2[0:2], device)
            vertex3 = from_numpy(vertex3[0:2], device)
            vertex4 = from_numpy(vertex4[0:2], device)

            obs_info.append(
                {
                    "normal_vec": [n12, n23, n34, n41],
                    "vertex": [vertex1, vertex2, vertex3, vertex4],
                }
            )
        return obs_info

    def construct_passage(self):
        assert self._finalized, "world_map is not finalized!"
        for obs_pair in combinations(self._obstacle, 2):
            if not (obs_pair[0]._wall and obs_pair[1]._wall):
                # 1
                request1 = fcl.DistanceRequest()
                result1 = fcl.DistanceResult()
                fcl.distance(
                    obs_pair[0]._collision_obj,
                    obs_pair[1]._collision_obj,
                    request1,
                    result1,
                )

                # 2
                request2 = fcl.DistanceRequest()
                result2 = fcl.DistanceResult()
                fcl.distance(
                    obs_pair[1]._collision_obj,
                    obs_pair[0]._collision_obj,
                    request2,
                    result2,
                )

                if result1.min_distance <= result2.min_distance:
                    self._passages.append(
                        Passage(
                            result1.nearest_points[0],
                            result1.nearest_points[1],
                            result1.min_distance,
                        )
                    )
                else:
                    self._passages.append(
                        Passage(
                            result2.nearest_points[0],
                            result2.nearest_points[1],
                            result2.min_distance,
                        )
                    )

    def filter_passage(self):
        """
        filter out the useless passage
        """
        assert self._finalized, "world_map is not finalized!"
        self._filtered_passages = []
        for passage in self._passages:
            passage_center = [
                (passage.vrtx1[0] + passage.vrtx2[0]) / 2.0,
                (passage.vrtx1[1] + passage.vrtx2[1]) / 2.0,
            ]

            passage_half_length = passage.min_dist / 2.0 - 0.0001
            cylinder_g = fcl.Cylinder(passage_half_length, 0.001)
            cylinder_t = fcl.Transform(
                np.array([passage_center[0], passage_center[1], 0.0])
            )
            cylinder = fcl.CollisionObject(cylinder_g, cylinder_t)
            collision = False

            for obs in self._obstacle:
                request = fcl.CollisionRequest()
                result = fcl.CollisionResult()
                fcl.collide(cylinder, obs._collision_obj, request, result)
                #
                collision = collision or result.is_collision

            if not collision:
                self._filtered_passages.append(passage)

        return self._filtered_passages

    def compute_clearance(self, state):
        """
        compute the minimal clearance from obstacles
        """
        assert self._finalized, "world_map is not finalized!"
        T = np.array([state[0], state[1], state[2]])
        tf = fcl.Transform(T)
        robot = fcl.CollisionObject(self._s, tf)

        clearance = []
        for i in range(len(self._collision_instances)):
            request = fcl.DistanceRequest()
            result = fcl.DistanceResult()
            obs = self._collision_instances[i]
            ret = fcl.distance(robot, obs, request, result)
            clearance.append(ret)
        cel = np.min(clearance)
        return cel

    def sample_validate_position(self):
        """
        sample a validate position
        """
        while True:
            print("*****")
            position_candidate = np.random.uniform(
                [self.map_cfg.map_xmin, self.map_cfg.map_ymin, self.map_cfg.map_zmin],
                [self.map_cfg.map_xmin, self.map_cfg.map_ymin, self.map_cfg.map_zmax],
                size=(3,),
            )
            if self.check_pos_collision(position_candidate):
                break
        return position_candidate

    def check_passage_intersection(self, parent_state, child_state):
        """
        check whether the <parent_state, child_state> pair intersects with some passage in the map
        """
        for passage in self._filtered_passages:
            intersection = check_intersection(
                Point(passage.vrtx1[0], passage.vrtx1[1]),
                Point(passage.vrtx2[0], passage.vrtx2[1]),
                Point(parent_state[0], parent_state[1]),
                Point(child_state[0], child_state[1]),
            )
            if intersection:
                return passage
        return None

    def get_path_intersection(self, sinle_path, use_extension: bool = False):
        """
        get the intersection between the path and passage
        """
        intersects = []
        path_len = sinle_path.shape[0]
        for i in range(path_len - 1):
            for passage in self._filtered_passages:
                path_1 = Point(sinle_path[i, 0], sinle_path[i, 1])
                path_2 = Point(sinle_path[i + 1, 0], sinle_path[i + 1, 1])

                direction_p = np.array(
                    [
                        passage.vrtx2[0] - passage.vrtx1[0],
                        passage.vrtx2[1] - passage.vrtx1[1],
                    ]
                )
                direction_p = direction_p / np.linalg.norm(direction_p)
                direction_n = np.array(
                    [
                        passage.vrtx1[0] - passage.vrtx2[0],
                        passage.vrtx1[1] - passage.vrtx2[1],
                    ]
                )
                direction_n = direction_n / np.linalg.norm(direction_n)

                extended_vrtx1 = (
                    np.array([passage.vrtx1[0], passage.vrtx1[1]]) + 250.0 * direction_n
                )
                extended_vrtx2 = (
                    np.array([passage.vrtx2[0], passage.vrtx2[1]]) + 250.0 * direction_p
                )

                passage_1 = Point(passage.vrtx1[0], passage.vrtx1[1])
                passage_2 = Point(passage.vrtx2[0], passage.vrtx2[1])

                if use_extension:
                    extended_passage_1 = Point(extended_vrtx1[0], extended_vrtx1[1])
                    extended_passage_2 = Point(extended_vrtx2[0], extended_vrtx2[1])
                    point = get_intersection_point(
                        path_1, path_2, extended_passage_1, extended_passage_2
                    )
                else:
                    point = get_intersection_point(path_1, path_2, passage_1, passage_2)

                if point:
                    # if check_on_segment(path_1, point, path_2) and check_on_segment(passage_1, point, passage_2):
                    intersects.append(
                        {
                            "passage": passage,
                            "passage_width": passage.min_dist,
                            "point": point,
                        }
                    )
        return intersects

    def check_pos_collision(self, state):
        assert self._finalized, "world_map is not finalized!"
        T = np.array([state[0], state[1], state[2]])
        tf = fcl.Transform(T)
        robot = fcl.CollisionObject(self._s, tf)

        req = fcl.CollisionRequest()
        rdata = fcl.CollisionData(request=req)
        self._collision_manager.collide(robot, rdata, fcl.defaultCollisionCallback)
        validate = not rdata.result.is_collision
        return validate

    def check_line_collision(
        self, start_state: np.ndarray, end_state: np.ndarray
    ) -> bool:
        assert self._finalized, "world_map is not finalized!"
        state_distance = np.linalg.norm(start_state - end_state)
        N = int(state_distance / (self._resolution))
        ratios = np.linspace(0.0, 1.0, num=N)

        for ratio in ratios:
            state_sample = (1 - ratio) * start_state + ratio * end_state
            res = self.check_pos_collision(state_sample)
            if res:
                return True
        return False

    def visualize_map(self, ax=None, show_wall: bool = False):
        from ..spatial_pathset_gen.utils import plot_box

        if ax is None:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(projection="3d")
        ax.set_xlim(self.map_cfg.map_xmin, self.map_cfg.map_xmax)
        ax.set_ylim(self.map_cfg.map_ymin, self.map_cfg.map_ymax)
        ax.set_zlim(0, self.map_cfg.map_zmax)
        ax.set_aspect("equal")
        ax.xaxis.set_ticks(
            np.arange(self.map_cfg.map_xmin, self.map_cfg.map_xmax, 50.0)
        )
        ax.yaxis.set_ticks(
            np.arange(self.map_cfg.map_ymin, self.map_cfg.map_ymax, 50.0)
        )
        ax.zaxis.set_ticks(
            np.arange(self.map_cfg.map_zmin, self.map_cfg.map_zmax, 50.0)
        )
        ax.set(xlabel="$x (m)$", ylabel="$y (m)$", zlabel="$z (m)$")
        ax.grid(True)

        # obstacle visulization
        for obstacle in self._obstacle:
            if show_wall:
                plot_box(
                    center=(obstacle._pos_x, obstacle._pos_y, obstacle._pos_z),
                    size=(obstacle._size_x, obstacle._size_y, obstacle._size_z),
                    ax=ax,
                    clr=obstacle._color,
                )
            else:
                if not obstacle._wall:
                    plot_box(
                        center=(obstacle._pos_x, obstacle._pos_y, obstacle._pos_z),
                        size=(obstacle._size_x, obstacle._size_y, obstacle._size_z),
                        ax=ax,
                        clr=obstacle._color,
                    )
        return ax

    def visualize_passage(self, ax=None, full_passage: bool = True):
        assert self._finalized, "world_map is not finalized!"
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        for obs in self._obstacle:
            plot_rectangle(
                obs._size_x,
                obs._size_y,
                obs._pos_x,
                obs._pos_y,
                ax,
                angle=obs.angle,
                color=obs._color,
            )

        if full_passage:
            for passage in self._passages:
                ax.plot(
                    [passage.vrtx1[0], passage.vrtx2[0]],
                    [passage.vrtx1[1], passage.vrtx2[1]],
                    "k--",
                )

        for passage in self._filtered_passages:
            (p,) = ax.plot(
                [passage.vrtx1[0], passage.vrtx2[0]],
                [passage.vrtx1[1], passage.vrtx2[1]],
                "k--",
                linewidth=2.0,
            )
        return ax


if __name__ == "__main__":
    from pprint import pprint

    cfg = MapCfg()
    print(cfg.resolution)
