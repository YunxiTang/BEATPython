'''
    collect image and store as dataset
'''
import pyrealsense2 as rs2
import cv2
import numpy as np
import time

from diffusion_planner.common.logger import ZarrLogger


if __name__ == '__main__':

    logger = ZarrLogger(path_to_save='/home/yxtang/CodeBase/PythonCourse/dataset/img_test.zarr', 
                        ks=['time', 'color_img', 'depth_img'],
                        chunk_size=1,
                        dtype='f4')

    pipeline = rs2.pipeline()
    config = rs2.config()
    config.enable_stream(rs2.stream.depth, 640, 360, rs2.format.z16, 15)
    config.enable_stream(rs2.stream.color, 640, 480, rs2.format.bgr8, 15)

    profile = pipeline.start(config)

    align_to = rs2.stream.color
    aligner = rs2.align(align_to)

    k = 0
    # streaming loop
    try:
        for _ in range(2000):
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
            
            tc = time.time()
            logger.log_dict_data({'time': tc, 'color_img': color_img, 'depth_img': depth_img})
            print(time.time() - tc)

            # render images
            cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Align Example', color_img)
            cv2.waitKey(1)

        logger.save_data()

    finally:
        pipeline.stop()