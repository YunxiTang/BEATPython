import cv2
import numpy as np
import time
import zarr
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import einops


dataset_path = "/home/yxtang/CodeBase/PythonCourse/dataset/img_test.zarr"
root = zarr.open(dataset_path, "r")
bgr_img = root["color_img"][:]
depth_img = root["depth_img"][:]
print(root["time"][:])


sns.set_theme()
_, ax = plt.subplots(1, 10)
for i in range(10):
    ele = bgr_img[i]
    img = cv2.cvtColor(ele, cv2.COLOR_BGR2RGB)
    img = img / 255
    ax[i].imshow(img)
plt.show()
