

if __name__ == "__main__":

    import numpy as np
    import open3d 
    from open3d import *

    print("Let\'s draw some primitives")
    world_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=30, origin=[0, 0, 0])

    ground = open3d.geometry.TriangleMesh.create_box(width = 200.0, height = 0.001, depth = 200)
    ground.compute_vertex_normals()
    ground.paint_uniform_color([0.5, 0.5, 0.5])

    box1 = open3d.geometry.TriangleMesh.create_box(width = 20.0, height = 120.0, depth = 15.0)
    box1.compute_vertex_normals()
    box1.paint_uniform_color([0.4, 0.4, 0.4])
    T = np.array([[1., 0., 0., 50.],
                  [0., 1., 0., 0.0],
                  [0., 0., 1., 50.],
                  [0., 0., 0., 1.]])
    box1.transform(T)

    box2 = open3d.geometry.TriangleMesh.create_box(width = 20.0, height = 80.0, depth = 25.0)
    box2.compute_vertex_normals()
    box2.paint_uniform_color([0.4, 0.4, 0.4])
    T = np.array([[1., 0., 0., 80.],
                  [0., 1., 0., 0.0],
                  [0., 0., 1., 100.],
                  [0., 0., 0., 1.]])
    box2.transform(T)

    points = [[0, 0, 0],
              [200, 200, 200]]
    lines = [[0, 1],]
    colors = [[0, 1, 0] for i in range(len(lines))]

    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.utility.Vector3dVector(points)
    line_set.lines = open3d.utility.Vector2iVector(lines)
    line_set.colors = open3d.utility.Vector3dVector(colors)

    open3d.visualization.draw_geometries([world_frame, ground, box1, box2, line_set,])