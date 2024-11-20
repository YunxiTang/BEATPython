import matplotlib.pyplot as plt
import jax.numpy as jnp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle 
from typing import NamedTuple


class Point(NamedTuple):
    '''
        2D point
    '''
    x: float
    y: float

    def to_list(self):
        return [self.x, self.y]
    

class Passage(NamedTuple):
    '''
        Passage in Passage Map
    '''
    vrtx1: list
    vrtx2: list
    min_dist: float
    

def line(p1: Point, p2: Point):
    A = (p1.y - p2.y)
    B = (p2.x - p1.x)
    C = (p1.x * p2.y - p2.x * p1.y)
    return A, B, -C


def get_intersection_point(p1: Point, q1: Point, p2: Point, q2: Point):
    x1,y1 = p1.x, p1.y
    x2,y2 = q1.x, q1.y
    x3,y3 = p2.x, p2.y
    x4,y4 = q2.x, q2.y
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)

    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return Point(x, y)
    

def get_orientation(p: Point, q: Point, r: Point): 
    '''
        get the orientation of an ordered triplet (p, q, r) 
        Return: 
            0 : Collinear points 
            1 : Clockwise points 
            2 : Counterclockwise points
        See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ for details.  
    '''
    val = ((r.x - q.x) * (q.y - p.y)) - ((r.y - q.y) * (q.x - p.x)) 
    if (val > 0.): 
        # Clockwise orientation 
        return 1
    elif (val < 0.): 
        # Counterclockwise orientation 
        return 2
    else: 
        # Collinear orientation 
        return 0
    
 
def check_on_segment(p: Point, q: Point, r: Point): 
    '''
        Given three collinear points (p, q, r), 
        check if q lies on line segment <p, r> 
    '''
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and \
         (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y)) ): 
        return True
    return False

 
def check_intersection(p1: Point, q1: Point, p2: Point, q2: Point): 
    '''
        returns true if the line segment <p1, q1> and <p2, q2> intersect.
        Find the four orientations required for the general and special cases 
    '''
    o1 = get_orientation(p1, q1, p2) 
    o2 = get_orientation(p1, q1, q2) 
    o3 = get_orientation(p2, q2, p1) 
    o4 = get_orientation(p2, q2, q1) 
  
    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return True
  
    # Special Cases 
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1 
    if ((o1 == 0) and check_on_segment(p1, p2, q1)): 
        return True
  
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1 
    if ((o2 == 0) and check_on_segment(p1, q2, q1)): 
        return True
  
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2 
    if ((o3 == 0) and check_on_segment(p2, p1, q2)): 
        return True
  
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2 
    if ((o4 == 0) and check_on_segment(p2, q1, q2)): 
        return True
  
    # if none of the cases 
    return False


# =============================================================================================
def plot_rectangle(size_x, size_y, pos_x, pos_y, ax, center=True, color='b'):
    if center:
        ax.add_patch(Rectangle([pos_x, pos_y], size_x / 2., size_y / 2., color=color))
        ax.add_patch(Rectangle((pos_x, pos_y), -size_x / 2., size_y / 2., color=color))
        ax.add_patch(Rectangle((pos_x, pos_y), -size_x / 2., -size_y / 2., color=color))
        ax.add_patch(Rectangle((pos_x, pos_y), size_x / 2., -size_y / 2., color=color))
    else:
        pass


def plot_circle(x, y, radius, ax, color="-b"):
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + radius * jnp.cos(jnp.deg2rad(d)) for d in deg]
    yl = [y + radius * jnp.sin(jnp.deg2rad(d)) for d in deg]
    ax.plot(xl, yl, color)


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
    faces.set_edgecolor([1.,1.,1.])
    
    ax.add_collection3d(faces)


def transfer_path(pivot_path, start, goal, delta_start=None, delta_goal=None):
    transfered_path = []

    pivot_start = pivot_path[0]
    pivot_goal = pivot_path[-1]
    pivot_path_num = len(pivot_path)

    pivot_path_len = pivot_path[-1].path_len

    if delta_start is None:
        delta_start = start - pivot_start.state
    if delta_goal is None:
        delta_goal = goal - pivot_goal.state

    for i in range(pivot_path_num):
        tmp_state = pivot_path[i].state + pivot_path[i].path_len / pivot_path_len * (delta_goal - delta_start) + delta_start
        tmp_node = SimpleNode(tmp_state, pivot_path[i].path_len)
        transfered_path.append(tmp_node)
    return transfered_path


if __name__ == '__main__':
    p1 = Point(0., 0.)
    q1 = Point(0.49, 0.49)

    p2 = Point(0., 1.)
    q2 = Point(1., 0.)

    print( check_intersection(p1, q1, p2, q2) )

    R = get_intersection_point(Point(0.6, 0.6), Point(1, 1), 
                               Point(0, 1), Point(1, 0))
    if R:
        print("Intersection detected:", R)
    else:
        print("No single intersection point detected")