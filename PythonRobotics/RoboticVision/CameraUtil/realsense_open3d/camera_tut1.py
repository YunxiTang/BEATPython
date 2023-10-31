import pyrealsense2 as rs2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    pipeline = rs2.pipeline()
    pipeline.start()

    try:
        frames = pipeline.wait_for_frames()
        
        depth_frame = frames.get_depth_frame()
        depth_width = depth_frame.get_width()
        depth_height = depth_frame.get_height()
        
        print('depth frame dim: {} x {}'.format(depth_width, depth_height))

        depth_np = np.asanyarray(depth_frame.get_data())
        fig = plt.figure(1)
        plt.matshow(depth_np, 1)
        plt.show()

    except Exception as e:
        print(e)
        exit()