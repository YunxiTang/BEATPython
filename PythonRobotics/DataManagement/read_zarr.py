import zarr
import numpy as np
import tqdm
import einops
from PIL import Image
import cv2
import time


def video_replay(dataset: str, total_frames: int = 1000):
    tmp = zarr.open(dataset, "r")
    print(tmp.tree())

    for i in range(total_frames):
        color_img = tmp["data"]["rgb_imgs"][i]
        depth_img = tmp["data"]["depth_imgs"][i]
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET
        )
        images = np.hstack((color_img, depth_colormap))

        cv2.namedWindow("video_replay", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("video_replay", images.astype(np.uint8))
        cv2.waitKey(1)

    cv2.destroyAllWindows()


video_replay("dataset/image_test_v0.zarr")
# tmp = zarr.open('dataset/image_test_v0.zarr', 'r')
# print(type(tmp))
# print(tmp.tree())

# # img = Image.fromarray(tmp['data']['rgb_imgs'][256].astype('uint8')[:,:,::-1])
# # img.show()

# for i in range(1000):

#     color_img = tmp['data']['rgb_imgs'][i]
#     depth_img = tmp['data']['depth_imgs'][i]
# #     # color_img = einops.rearrange(color_img, 'h w c -> c h w')
# #     # depth_img = einops.repeat(depth_img, 'h w -> h w c', c=3)

#     # color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
#     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03),
#                                        cv2.COLORMAP_JET)
#     images = np.hstack((color_img, depth_colormap))

#     cv2.namedWindow('Test', cv2.WINDOW_AUTOSIZE)
#     cv2.imshow('Test', images.astype(np.uint8))
#     # time.sleep(1/30.)
#     cv2.waitKey(1)

# cv2.destroyAllWindows()
