import os, sys
import numpy as np
import einops
import cv2 as cv2
import pyrealsense2 as rs2


class LKTracker:
    def __init__(self, init_color_frame, window_name=None, init_keypoint=None):
        self._window_name = (
            window_name if window_name is not None else "Keypoint Selector"
        )
        self._init_color_frame = init_color_frame
        self.prev_img = cv2.cvtColor(init_color_frame, cv2.COLOR_BGR2GRAY)
        self.next_img = None

        # tracked_keypoint in shape of (num_keypoint, 1, 2)
        if init_keypoint is None:
            self.tracked_keypoint = self._select_keypoint()
        else:
            self.tracked_keypoint = init_keypoint

        self.num_keypoint = self.tracked_keypoint.shape[0]

        # parameters for lucas kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def _click_event(self, event, x, y, flags=None, params=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._selected_points.append((x, y))
        return self._selected_points

    def _select_keypoint(self):
        # key point selection
        self._selected_points = []
        img = self._init_color_frame
        while 1:
            cv2.imshow(self._window_name, self._init_color_frame)
            cv2.setMouseCallback(self._window_name, self._click_event)
            for i in range(len(self._selected_points)):
                img = cv2.circle(img, self._selected_points[i], 3, [255, 0, 0], -1)
            key = cv2.waitKey(1)

            if key & 0xFF == ord("q") or key == 27:
                break
        cv2.destroyWindow(self._window_name)
        points_np = np.array(self._selected_points, dtype=np.float32)[None, ...]
        points_np = einops.rearrange(points_np, "a b c -> b a c")
        return points_np

    def track_keypoint(self, frame):
        # convert into gray image
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p0 = self.tracked_keypoint

        # LK tracking
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_img, frame_gray, p0, None, **self.lk_params
        )

        # select good points
        good_new = np.zeros_like(self.tracked_keypoint)
        if p1 is not None:
            good_new[st == 1] = p1[st == 1]
            good_new[st == 0] = p0[st == 0]

        # update stored historical information
        self.prev_img = frame_gray
        self.tracked_keypoint = good_new

        return self.tracked_keypoint.copy()


if __name__ == "__main__":
    cv2.namedWindow("sample image")
    img_1 = cv2.imread("./imgs/1.jpeg")
    img_2 = cv2.imread("./imgs/2.jpeg")

    # point tracking
    lk_tracker = LKTracker(img_1)

    while 1:
        tracked_keypoints = lk_tracker.track_keypoint(img_2)

        for i in range(tracked_keypoints.shape[0]):
            img_2 = cv2.circle(
                img_2,
                (int(tracked_keypoints[i, 0, 0]), int(tracked_keypoints[i, 0, 1])),
                3,
                [125, 125, 50],
                -1,
            )

        cv2.imshow("sample image", img_2)

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()
