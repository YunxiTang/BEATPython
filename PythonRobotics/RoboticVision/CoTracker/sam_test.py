import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import einops
import torchvision
import sys
sys.path.append("..")
from segment_anything import SamPredictor, sam_model_registry


class LKTracker:
    def __init__(self, init_color_frame, window_name=None, init_keypoint = None):
        self._window_name = window_name if window_name is not None else 'Keypoint Selector'
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
        self.lk_params = dict(winSize  = (15, 15),
                              maxLevel = 3,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        

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

            if key & 0xFF == ord('q') or key == 27:
                break
        cv2.destroyWindow(self._window_name)
        points_np = np.array(self._selected_points, dtype=np.float32)[None,...]
        points_np = einops.rearrange(points_np, 'a b c -> b a c')
        return points_np


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

image = cv2.imread('/home/yxtang/CodeBase/PythonCourse/PythonRobotics/RoboticVision/CoTracker/cable_test.PNG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


sam = sam_model_registry["vit_h"](checkpoint="/home/yxtang/CodeBase/thrid_party/checkpoints/sam_vit_h_4b8939.pth")
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image=image)

lk_tracker = LKTracker(image, 'select_particle')
selected_particlses = lk_tracker._selected_points
input_point = np.array(selected_particlses)
input_label = np.array([1])


plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()  

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True)

print(masks.shape)
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  

