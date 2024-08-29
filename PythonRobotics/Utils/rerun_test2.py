import rerun as rr
import numpy as np
import time
from PIL import Image
import zarr
from tqdm import tqdm
from rerun.blueprint import (
            Blueprint,
            Horizontal,
            Vertical,
            Spatial2DView,
            Spatial3DView,
            TimeSeriesView,
            Tabs,
            SelectionPanel,
            TimePanel,
            TextDocumentView
        )
import open3d as o3d


num = 2009
video_path = f'/media/yxtang/Extreme SSD/DOM_Reaseach/dobert_dataset/dom_dataset/episode_{num}/camera.zarr'
js_path = f'/media/yxtang/Extreme SSD/DOM_Reaseach/dobert_dataset/dom_dataset/episode_{num}/js.zarr'
kp_tracked_path = f'/media/yxtang/Extreme SSD/DOM_Reaseach/dobert_dataset/dom_dataset/episode_{num}/kp_track.zarr'

video_frames_root = zarr.open(video_path, mode='r')
js_root = zarr.open(js_path, mode='r')
kp_track_root = zarr.open(kp_tracked_path, mode='r')

rgb_img = video_frames_root['rgb_img']
timestamp = video_frames_root['timestamp']
qpos = js_root['qpos']
qvel = js_root['qvel']
kp_tracked = kp_track_root['data']['kp_tracks'][:][0,...]

blue_print = Blueprint(
    Horizontal(
        Vertical(Spatial2DView(name='rgb', origin='/camera', contents=["+ $origin/rgb",]), 
                 Spatial2DView(name='tracked_kp', origin='/camera', contents=["+ $origin/tracked_kp",])),
        Vertical(TimeSeriesView(name='qpos', origin='/robot', contents=["+ $origin/qpos/**",]), 
                 TimeSeriesView(name='qvel', origin='/robot', contents=["+ $origin/qvel/**",]))
    )
)
rr.init(f"ropePush_ep{num}", spawn=True, default_blueprint=blue_print)

for i in range(timestamp.shape[0]):
    rr.set_time_seconds("time", timestamp[i]-timestamp[0])
    rr.log("camera/rgb", rr.Image(rgb_img[i]).compress(jpeg_quality=95))

    for j in range(qpos.shape[1]):
        rr.log(f"robot/qpos/j_{j}", rr.Scalar(qpos[i, j]))
        rr.log(f"robot/qvel/j_{j}", rr.Scalar(qvel[i, j]))
    # print(kp_tracked[i])
    rr.log('camera/tracked_kp', rr.Points2D(kp_tracked[i], radii=5.0) )