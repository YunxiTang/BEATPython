import numpy as np
import math
import warp as wp
from config import DEVICE
import os
import json
from plyfile import PlyData, PlyElement


@wp.func
def wp_vec3_mul_element(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    return wp.vec3(a[0] * b[0], a[1] * b[1], a[2] * b[2])

# Reinstate the element-wise vector square root helper function
@wp.func
def wp_vec3_sqrt(a: wp.vec3) -> wp.vec3:
    return wp.vec3(wp.sqrt(a[0]), wp.sqrt(a[1]), wp.sqrt(a[2]))

# Add element-wise vector division helper function
@wp.func
def wp_vec3_div_element(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    # Add small epsilon to denominator to prevent division by zero
    # (although Adam's epsilon should mostly handle this)
    safe_b = wp.vec3(b[0] + 1e-9, b[1] + 1e-9, b[2] + 1e-9)
    return wp.vec3(a[0] / safe_b[0], a[1] / safe_b[1], a[2] / safe_b[2])

@wp.func
def wp_vec3_add_element(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    return wp.vec3(a[0] + b[0], a[1] + b[1], a[2] + b[2])

@wp.func
def wp_vec3_clamp(x: wp.vec3, min_val: float, max_val: float) -> wp.vec3:
    return wp.vec3(
        wp.clamp(x[0], min_val, max_val),
        wp.clamp(x[1], min_val, max_val),
        wp.clamp(x[2], min_val, max_val)
    )

def to_warp_array(data, dtype, shape_check=None, flatten=False):
    if isinstance(data, wp.array):
        return data
    if data is None:
        return None
    # Convert torch tensor to numpy if needed
    if hasattr(data, 'cpu') and hasattr(data, 'numpy'):
        data = data.cpu().numpy()
    if flatten and data.ndim == 2 and data.shape[1] == 1:
        data = data.flatten()
    return wp.array(data, dtype=dtype, device=DEVICE)


def world_to_view(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def projection_matrix(fovx, fovy, znear, zfar):
    tanHalfFovY = math.tan((fovy / 2))
    tanHalfFovX = math.tan((fovx / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def matrix_to_quaternion(matrix):
    """
    Convert a 3x3 rotation matrix to a quaternion in (x, y, z, w) format.
    
    Args:
        matrix: 3x3 rotation matrix
        
    Returns:
        Quaternion as (x, y, z, w) in numpy array of shape (4,)
    """
    # Ensure the input is a proper rotation matrix
    # This is just a simple check that might be helpful during debug
    if np.abs(np.linalg.det(matrix) - 1.0) > 1e-5:
        print(f"Warning: Input matrix determinant is not 1: {np.linalg.det(matrix)}")
    
    trace = np.trace(matrix)
    if trace > 0:
        S = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * S
        x = (matrix[2, 1] - matrix[1, 2]) / S
        y = (matrix[0, 2] - matrix[2, 0]) / S
        z = (matrix[1, 0] - matrix[0, 1]) / S
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        S = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / S
        x = 0.25 * S
        y = (matrix[0, 1] + matrix[1, 0]) / S
        z = (matrix[0, 2] + matrix[2, 0]) / S
    elif matrix[1, 1] > matrix[2, 2]:
        S = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / S
        x = (matrix[0, 1] + matrix[1, 0]) / S
        y = 0.25 * S
        z = (matrix[1, 2] + matrix[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / S
        x = (matrix[0, 2] + matrix[2, 0]) / S
        y = (matrix[1, 2] + matrix[2, 1]) / S
        z = 0.25 * S
    
    # Return as (x, y, z, w) to match Warp's convention
    return np.array([x, y, z, w], dtype=np.float32)



# camera utils
# Y down, Z forward
def load_camera(camera_info):
    """Load camera parameters from camera info dictionary"""
    # Extract camera parameters
    # camera_id = camera_info["camera_id"]
    camera_to_world = np.asarray(camera_info["camera_to_world"], dtype=np.float64)
    
    # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    camera_to_world[:3, 1:3] *= -1
    
    # Calculate world to camera transform
    world_to_camera = np.linalg.inv(camera_to_world).astype(np.float32)
    
    
    # Extract rotation and translation
    R = world_to_camera[:3, :3]
    T = world_to_camera[:3, 3]
    
    
    world_to_camera[3, 3] = 1.
    world_to_camera = world_to_camera.T

    
    width = camera_info.get("width")
    height = camera_info.get("height")
    fx = camera_info.get("focal")
    fy = camera_info.get("focal")
    cx = width / 2
    cy = height / 2
    
    # Calculate field of view from focal length
    fovx = 2 * np.arctan(width / (2 * fx))
    fovy = 2 * np.arctan(height / (2 * fy))
    
    # Create view matrix
    view_matrix = world_to_view(R=R, t=T)
    
    # Create projection matrix
    znear = 0.01
    zfar = 100.0
    proj_matrix = projection_matrix(fovx=fovx, fovy=fovy, znear=znear, zfar=zfar).T
    full_proj_matrix = world_to_camera @ proj_matrix
    
    # Calculate other parameters
    tan_fovx = np.tan(fovx * 0.5)
    tan_fovy = np.tan(fovy * 0.5)
    
    camera_center = np.linalg.inv(world_to_camera)[3, :3]
    
    # Handle camera type and distortion
    camera_model = camera_info.get("camera_model", "OPENCV")
    if camera_model == "OPENCV" or camera_model is None:
        camera_type = 0  # PERSPECTIVE
    elif camera_model == "OPENCV_FISHEYE":
        camera_type = 1  # FISHEYE
    else:
        raise ValueError(f"Unsupported camera_model '{camera_model}'")
    
    # Get distortion parameters
    distortion_params = []
    for param_name in ["k1", "k2", "p1", "p2", "k3", "k4"]:
        distortion_params.append(camera_info.get(param_name, 0.0))
    
    camera_params = {
        'R': R,
        'T': T,
        'camera_center': camera_center,
        'view_matrix': view_matrix,
        'proj_matrix': proj_matrix,
        'full_proj_matrix': full_proj_matrix,
        'tan_fovx': tan_fovx,
        'tan_fovy': tan_fovy,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'width': width,
        'height': height,
        'camera_to_world': camera_to_world,
        'world_to_camera': world_to_camera,
        'camera_type': camera_type,
        'distortion_params': np.array(distortion_params, dtype=np.float32)
    }
    
    return camera_params

def load_camera_from_json(input_path, camera_id=0):
    """Load camera parameters from camera.json file"""
    camera_file = os.path.join(os.path.dirname(input_path), "cameras.json")
    if not os.path.exists(camera_file):
        print(f"Warning: No cameras.json found in {os.path.dirname(input_path)}, using default camera")
        return None
    
    try:
        with open(camera_file, 'r') as f:
            cameras = json.load(f)
        
        # Find camera with specified ID, or use the first one
        camera = next((cam for cam in cameras if cam["id"] == camera_id), cameras[0])
        
        # Use load_camera to process the camera parameters
        return load_camera(camera)
        
    except Exception as e:
        print(f"Error loading camera from cameras.json: {e}")
        return None
    
    
    
    
    
    
# ============= pointcloud ===================
# from plyfile import PlyData, PlyElement

# Function to save point cloud to PLY file
def save_ply(params, filepath, num_points, colors=None):
    # Get numpy arrays
    positions = params['positions'].numpy()
    scales = params['scales'].numpy()
    rotations = params['rotations'].numpy()
    opacities = params['opacities'].numpy()
    shs = params['shs'].numpy()
    
    # Handle colors - either provided or computed from SH coefficients
    if colors is not None:
        # Use provided colors
        if hasattr(colors, 'numpy'):
            colors_np = colors.numpy()
        else:
            colors_np = colors
    else:
        # Compute colors from SH coefficients (DC term only for simplicity)
        # SH DC coefficients are stored in the first coefficient (index 0)
        colors_np = np.zeros((num_points, 3), dtype=np.float32)
        for i in range(num_points):
            # Get DC term from SH coefficients
            sh_dc = shs[i * 16]  # First SH coefficient contains DC term
            # Convert from SH to RGB (simplified - just use DC term)
            colors_np[i] = np.clip(sh_dc + 0.5, 0.0, 1.0)  # Add 0.5 offset and clamp
    
    # Create vertex data
    vertex_data = []
    for i in range(num_points):
        # Basic properties
        vertex = (
            positions[i][0], positions[i][1], positions[i][2],
            scales[i][0], scales[i][1], scales[i][2],
            opacities[i]
        )
        
        # Add rotation quaternion elements
        quat = rotations[i]
        rot_elements = (quat[0], quat[1], quat[2], quat[3])  # x, y, z, w
        vertex += rot_elements
        
        # Add RGB colors (convert to 0-255 range)
        color_255 = (
            int(np.clip(colors_np[i][0] * 255, 0, 255)),
            int(np.clip(colors_np[i][1] * 255, 0, 255)),
            int(np.clip(colors_np[i][2] * 255, 0, 255))
        )
        vertex += color_255
        
        # Add SH coefficients
        sh_dc = tuple(shs[i * 16][j] for j in range(3))
        vertex += sh_dc
        
        # Add remaining SH coefficients
        sh_rest = []
        for j in range(1, 16):
            for c in range(3):
                sh_rest.append(shs[i * 16 + j][c])
        vertex += tuple(sh_rest)
        
        vertex_data.append(vertex)
    
    # Define the structure of the PLY file
    vertex_type = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('opacity', 'f4')
    ]
    
    # Add rotation quaternion elements
    vertex_type.extend([('rot_x', 'f4'), ('rot_y', 'f4'), ('rot_z', 'f4'), ('rot_w', 'f4')])
    
    # Add RGB color fields
    vertex_type.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    # Add SH coefficients
    vertex_type.extend([('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')])
    
    # Add remaining SH coefficients
    for i in range(45):  # 15 coeffs * 3 channels
        vertex_type.append((f'f_rest_{i}', 'f4'))
    
    vertex_array = np.array(vertex_data, dtype=vertex_type)
    el = PlyElement.describe(vertex_array, 'vertex')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the PLY file
    PlyData([el], text=False).write(filepath)
    print(f"Point cloud saved to {filepath}")
    