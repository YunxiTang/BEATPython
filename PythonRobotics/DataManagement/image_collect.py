import zarr
import numpy as np
import os
import json


class ImageCollector:
    '''
        collect and save RGB-D image as np.ndarray.
        TODO: support more image formats
    '''
    def __init__(self, 
                 directory: str, 
                 dataset_name: str, 
                 image_width: int,
                 image_height: int,
                 camera_fps: int,
                 capacity: int = 10000, 
                 chunk_size: int = 100,
                 mode: str = 'w',
                 overwrite: bool = True,
                 formate: str = None):
        
        if not os.path.exists(directory):
            print("ImageCollector: making new directory at {}".format(directory))
            os.makedirs(directory)

        # the base directory
        self.directory = directory
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join( self.directory, self.dataset_name )

        self.capacity = capacity
        self.chunk_size = chunk_size

        # disk cache for collected
        self.root = zarr.open(self.dataset_path, mode)
        self.data = self.root.create_group('data', overwrite=overwrite)

        self.image_width = image_width
        self.image_height = image_height
        self.camera_fps = camera_fps

        self.rgb_imgs_list = []
        self.depth_imgs_list = []

        self.frame_count = 0


    def add_rgb_and_depth_img(self, rgb_img, depth_img):
        self.rgb_imgs_list.append(rgb_img)
        self.depth_imgs_list.append(depth_img)
        self.frame_count += 1


    def flush_to_disk(self):
        rgb_imgs = self.data.create_dataset('rgb_imgs', 
                                            shape=(self.frame_count, self.image_width, self.image_height, 3), 
                                            dtype='f4', 
                                            chunks=(self.chunk_size, None, None, None), 
                                            overwrite=True)
        
        depth_imgs = self.data.create_dataset('depth_imgs', 
                                              shape=(self.frame_count, self.image_width, self.image_height), 
                                              dtype='f4', 
                                              chunks=(self.chunk_size, None, None), 
                                              overwrite=True)
        rgb_imgs[:] = np.array(self.rgb_imgs_list)
        depth_imgs[:] = np.array(self.depth_imgs_list)



if __name__ == '__main__':
    import pyrealsense2 as rs2
    import cv2
    import einops
    import time

    img_width, img_height = 640, 360
    fps = 30
    collector = ImageCollector(
        './dataset',
        'image_test_v0',
        image_width=360,
        image_height=640,
        camera_fps=30
    )

    pipeline = rs2.pipeline()
    config = rs2.config()
    config.enable_stream(rs2.stream.depth, img_width, img_height, rs2.format.z16, fps)
    config.enable_stream(rs2.stream.color, img_width, img_height, rs2.format.bgr8, fps)

    # The pipeline profile includes a device and a selection of active streams, with specific profiles. 
    # Streams may belong to more than one sensor of the device. 
    profile = pipeline.start(config)

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

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

            depth_img = np.asanyarray(aligned_depth_frame.get_data()).copy()
            color_img = np.asanyarray(color_frame.get_data()).copy()
            print(depth_img.shape)
            # save
            ts = time.time()
            collector.add_rgb_and_depth_img(color_img, depth_img)
            print(time.time()-ts)

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
        collector.flush_to_disk()

