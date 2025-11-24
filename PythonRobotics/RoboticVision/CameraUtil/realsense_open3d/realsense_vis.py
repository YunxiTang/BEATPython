import pyrealsense2 as rs
import open3d as o3d
import math
import time
import numpy as np


# config the camera
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream( rs.stream.depth, 640, 480, rs.format.z16, 30 )
config.enable_stream( rs.stream.color, 1280, 720, rs.format.bgr8, 30 )

profile = pipeline.start(config)

point_cloud = rs.pointcloud()
points = rs.points()

align = rs.align(rs.stream.color)

try:
    for fid in range(20):
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

    t0 = time.time()

    frames = pipeline.wait_for_frames()
    print('elapsed time of getting frames: {}'.format(time.time()-t0))
    t1 = time.time()
    aligned_frames = align.process(frames)
    print('elapsed time of aligning frames: {}'.format(time.time()-t1))

    profile = aligned_frames.get_profile()
    intrinsics = profile.as_video_stream_profile().get_intrinsics()

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
    )

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    t2 = time.time()
    img_depth = o3d.geometry.Image(depth_image)
    img_color = o3d.geometry.Image(color_image)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

    print('elapsed time: {}'.format(time.time() - t2))
    print('elapsed total time: {}'.format(time.time() - t0))

    pcd.transform([[1,0,0,0],
                   [0,-1,0,0],
                   [0,0,0-1,0],
                   [0,0,0,1]])
    
    o3d.visualization.draw_geometries([pcd])

finally:
    pipeline.stop()
    print('done')
