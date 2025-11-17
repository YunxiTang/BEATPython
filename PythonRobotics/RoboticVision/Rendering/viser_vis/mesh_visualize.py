import viser
import os
import numpy as np
import trimesh
import viser.transforms as tf
import time


def main():
    mesh_file_path = '/home/tyx/CodeBase/BEATPython/PythonRobotics/RoboticControl/franka/collision/finger_0.obj'
    mesh = trimesh.load_mesh(
        mesh_file_path
    )
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    print(f'number of vertices {vertices.shape} || number of faces: {faces.shape}')
    
    viser_server = viser.ViserServer()
    
    viser_server.scene.add_grid(
        name='/plane',
        plane='xy'
    )
    
    mesh_handle = viser_server.scene.add_mesh_trimesh(
        name='/obj1', mesh=mesh, scale=1., 
        wxyz=tf.SO3.from_x_radians(np.pi/2).wxyz, position=(0, 0, 0)
    )
    
    mesh_handle2 = viser_server.scene.add_mesh_trimesh(
        name='/obj2', mesh=mesh, scale=1., position=(0, 0, 0)
    )
    
    while True:
        time.sleep(0.1)


if __name__ == '__main__':
    main()