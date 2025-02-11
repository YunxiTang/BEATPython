import pyrealsense2 as rs
import open3d as o3d
import math
import time
import numpy as np
import cv2

# ================ create & start RealSense pipline ====================
# pipeline serves as a top-level API for stereaming and processing frames
pipe = rs.pipeline()

# ========= config the camera and start the camera ==============
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth,  640, 480, rs.format.z16, 30)

pipe.start(config)

try:
  while(1):
    # block program until frames arrive
    frames = pipe.wait_for_frames()
    
    # get depth frame
    depth_frame = frames.get_depth_frame()
    depth_width = depth_frame.get_width()
    depth_height = depth_frame.get_height()

    # query the distance from the camera to the object in the center of image
    dist_to_center = depth_frame.get_distance(int(depth_width / 2), int(depth_height / 2))

    print('depth frame dim: {} x {}'.format(depth_width, depth_height))
    print('camera is facing an object at: {}'.format(dist_to_center))
    
finally:
    pipe.stop()
