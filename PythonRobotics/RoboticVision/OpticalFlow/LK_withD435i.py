import numpy as np
import cv2 as cv
import pyrealsense2 as rs2
import einops

# =============================== algo params ====================================
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 5,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# ============================== setup camera =====================================
pipeline = rs2.pipeline()
config = rs2.config()

config.enable_stream(rs2.stream.color, 640, 480, rs2.format.bgr8, 30)
config.enable_stream(rs2.stream.depth, 640, 480, rs2.format.z16, 30 )

aligner = rs2.align(rs2.stream.color)

profile = pipeline.start(config)

device = profile.get_device()
depth_sensor = device.first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

try:
    raw_frames = pipeline.wait_for_frames()
    frames = aligner.process(raw_frames)

    color_frame = frames.get_color_frame()
    color_img = np.asanyarray(color_frame.get_data())
    old_frame = color_img
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(1):
        raw_frames = pipeline.wait_for_frames()
        frames = aligner.process(raw_frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
            
        # check frames avaible or not
        if not depth_frame or not color_frame:
            print('[INFO]: No frames detected')
            continue

        depth_img = np.asanyarray(depth_frame.get_data())
        
        color_img = np.asanyarray(color_frame.get_data())
        frame_gray = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            color_img = cv.circle(color_img, (int(a), int(b)), 5, color[i].tolist(), -1)
            # get depth of key point
            dist_to_center = depth_frame.get_distance(int(new[0]), int(new[1]))
            tmp = depth_img[int(b), int(a)] * depth_scale
            print('key point: {} with depth {}. {}'.format(new, dist_to_center, tmp))
        print('+-------------------------------------------------+')

        img = cv.add(color_img, mask)
        # image rendering
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_img, alpha=0.03), cv.COLORMAP_JET)
        img = np.hstack((img, depth_colormap))
        cv.namedWindow('LK Track Demo', cv.WINDOW_AUTOSIZE)
        cv.imshow('frame', img)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        
finally:
    pipeline.stop()
    cv.destroyAllWindows()
