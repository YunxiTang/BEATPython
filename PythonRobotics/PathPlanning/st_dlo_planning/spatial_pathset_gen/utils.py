import matplotlib.pyplot as plt
import jax.numpy as jnp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle
from typing import NamedTuple
import numpy as np
# from st_dlo_planning.spatial_pathset_gen.world_map import WorldMap


class Point(NamedTuple):
    """
    2D point
    """

    x: float
    y: float

    def to_list(self):
        return [self.x, self.y]


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


# =============================================================================================
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


def plot_box(center, size, ax, clr="gray"):
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

    faces.set_alpha(0.8)
    faces.set_facecolor(clr)
    faces.set_edgecolor([1.0, 1.0, 1.0])

    ax.add_collection3d(faces)


#
# def deform_path(pathset, world_map: WorldMap):
#     '''
#         deform the pathset to get collision-free pathset
#     '''

#     world_map.get_path_intersection( pathset )
