import pyrealsense2 as rs2
import numpy as np
import cv2

if __name__ == '__main__':
    # construct a pipeline
    pipeline = rs2.pipeline()

    # create a default config object
    config = rs2.config()

    # configure depth and color streams
    config.enable_stream(rs2.stream.depth, 640, 480, rs2.format.z16, 30)
    config.enable_stream(rs2.stream.color, 640, 480, rs2.format.bgr8, 30)

    # start streaming with config
    pipeline.start(config)

    try:
        while(1):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            # check frames avaible or not
            if not depth_frame or not color_frame:
                continue
            
            # frames info
            depth_width = depth_frame.get_width()
            depth_height = depth_frame.get_height()
            
            color_width = color_frame.get_width()
            color_height = color_frame.get_height()

            print('depth frame dim: {} x {}'.format(depth_width, depth_height)) # 640 x 480
            print('color frame dim: {} x {}'.format(color_width, color_height)) # 640 x 480
            print('+-------------------------------------+')

            # convert images to np array
            depth_img = np.asanyarray(depth_frame.get_data()) # shape (480, 640)
            color_img = np.asanyarray(color_frame.get_data()) # shape (480, 640, 3)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            scaled_depth_img = cv2.convertScaleAbs(depth_img, alpha=0.5)
            depth_colormap = cv2.applyColorMap(scaled_depth_img, cv2.COLORMAP_RAINBOW) # shape (480, 640, 3)

            # Stack both images horizontally
            images = np.hstack((color_img, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

    finally:
        pipeline.stop()

    