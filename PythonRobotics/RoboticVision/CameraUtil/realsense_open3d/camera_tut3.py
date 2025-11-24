'''
    Stream Alignment (depth-2-color) - Demonstrate a way of performing background removal 
    by aligning depth images to color images,
    and performing simple calculation to strip the background.
'''
import pyrealsense2 as rs2
import cv2
import numpy as np
import einops


if __name__ == '__main__':
    pipeline = rs2.pipeline()
    config = rs2.config()
    config.enable_stream(rs2.stream.depth, 640, 360, rs2.format.z16, 30)
    config.enable_stream(rs2.stream.color, 640, 480, rs2.format.bgr8, 30)

    # The pipeline profile includes a device and a selection of active streams, with specific profiles. 
    # Streams may belong to more than one sensor of the device. 
    profile = pipeline.start(config)

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs2.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the target stream type
    align_to = rs2.stream.color
    aligner = rs2.align(align_to)

    # Streaming loop
    try:
        while(1):
            frames = pipeline.wait_for_frames()
            raw_depth_frame = frames.get_depth_frame()

            # Align the depth frame to color frame
            aligned_frames = aligner.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_img = np.asanyarray(aligned_depth_frame.get_data())
            color_img = np.asanyarray(color_frame.get_data())

            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = einops.repeat(depth_img, 'h w -> h w c', c=3)
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_img)
            
            # Render images
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))
            cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Align Example', images)
            cv2.waitKey(1)

    finally:
        pipeline.stop()


