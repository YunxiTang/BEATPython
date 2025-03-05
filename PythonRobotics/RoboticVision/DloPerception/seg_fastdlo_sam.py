import os, cv2
from fastdlo.core import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from pprint import pprint
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
import time


def split_fastdlo_res(fastdlo_res):
    nodes = fastdlo_res['nodes']
    res = []
    segment_idx = 1
    segment = []
    # node_idx = 0
    for node_idx, node in nodes.items():
        if 'segment' in node.keys() and node_idx > 0:
            # a new segment appears
            segment_idx += 1
            segment.append(node)
            res.append(segment)

            segment = []
            # segment.append(node)
        else:
            segment.append(node)
    return res


def extract_uniform_keypoints(mask, num_keypoints=10):
    # Step 1: 预处理 Mask（转换为二值图像）
    mask = (mask > 0).astype(np.uint8)  # 确保是二值
    # plt.imshow(mask, cmap="gray")
    # plt.show()

    mask = cv2.erode(mask, kernel=np.ones((5, 5), dtype=np.uint8), iterations=1)
    mask = cv2.dilate(mask, kernel=np.ones((5, 5), dtype=np.uint8), iterations=1)
    # plt.imshow(mask, cmap="gray")
    # plt.show()

    # Step 2: 骨架化（得到单像素宽的中心线）
    skeleton = skeletonize(mask, method='lee').astype(np.uint8)
    # plt.imshow(skeleton, cmap="gray")
    # plt.show()

    # Step 3: 获取骨架的像素点坐标
    yx_points = np.column_stack(np.where(skeleton > 0))  # (N, 2) 形式

    # Step 4: 计算骨架点之间的距离矩阵
    dists = cdist(yx_points, yx_points)
    
    # Step 5: 找到端点（度为1的点）
    neighbor_count = np.sum(dists < 2, axis=0)  # 计算邻近点数
    end_points = yx_points[neighbor_count == 2]

    # Step 6: 选择起点（端点之一）
    start_point = end_points[0]

    # Step 7: 构造一条有序的曲线
    sorted_curve = [start_point]
    remaining_points = set(map(tuple, yx_points))
    remaining_points.remove(tuple(start_point))

    while remaining_points:
        last_point = sorted_curve[-1]
        # 找到下一个最接近的点
        next_point = min(remaining_points, key=lambda p: np.linalg.norm(np.array(p) - last_point))
        sorted_curve.append(next_point)
        remaining_points.remove(next_point)

    sorted_curve = np.array(sorted_curve)  # (M, 2)

    # Step 8: 计算累积弧长
    distances = np.cumsum(np.linalg.norm(np.diff(sorted_curve, axis=0), axis=1))
    distances = np.insert(distances, 0, 0)  # 插入起始点

    # Step 9: 在曲线等间距采样 keypoints
    sample_points = np.linspace(0, distances[-1], num_keypoints)
    keypoints = np.zeros((num_keypoints, 2))

    for i, sp in enumerate(sample_points):
        idx = np.searchsorted(distances, sp)
        keypoints[i] = sorted_curve[idx]

    return keypoints.astype(int)


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


if __name__ == "__main__":
    device = "cuda"
    # ================= fastdlo ===========================
    IMG_W = 640
    IMG_H = 480

    checkpoint_siam = "/media/yxtang/Extreme SSD/fastdlo/weights/CP_similarity.pth"
    checkpoint_seg = "/media/yxtang/Extreme SSD/fastdlo/weights/CP_segmentation.pth"
    p = Pipeline(checkpoint_siam=checkpoint_siam, checkpoint_seg=checkpoint_seg, img_w=IMG_W, img_h=IMG_H)

    # ================= SAM ==================================
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry["vit_h"](checkpoint="/home/yxtang/CodeBase/thrid_party/checkpoints/sam_vit_h_4b8939.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)

    cap = cv2.VideoCapture('/media/yxtang/Extreme SSD/DOM_Reaseach/dobert_cache/dobert_dataset/dom_dataset/episode_1009/episode_1009.mp4')

    frame = 0
    while True:
        ret, source_img = cap.read()

        if source_img is None:
            break
        
        if frame % 15 == 0:
            # COLOR IMG
            source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
            source_img = cv2.resize(source_img, (IMG_W, IMG_H))

            img_out, _, data_res = p.run(source_img=source_img, mask_th=77)

            image4sam = source_img.copy()
            canvas0 = source_img.copy()
            canvas1 = source_img.copy()
            canvas = source_img.copy()

            canvas = cv2.addWeighted(canvas, 0.7, img_out, 1.0, 0.0)
            cv2.imshow("fast_dlo", canvas)
            cv2.waitKey(10)

            grouped_data_res = split_fastdlo_res(data_res)

            clrs = [[0,0,255], [255,0,0], [0,255,0],
                    [125,0,125], [0,125,125], [125,125,0],
                    [50, 125, 50], [150, 25, 150]]
            
            seg_idx = 0
            for segment in grouped_data_res:
                for node in segment:
                    pos = node['pos']
                    # image = cv2.circle(canvas0, (pos[1], pos[0]), radius=3, color=clrs[seg_idx], thickness=-1)
                seg_idx += 1
            # print(seg_idx)
            # cv2.imshow("output0", canvas0)
            # cv2.waitKey(10)

            seg_lens = [len(segment) for segment in grouped_data_res]
            max_len_seg_idx = np.argmax(seg_lens)
            rope_segment = grouped_data_res[max_len_seg_idx]
            prompt_points = []
            for i in range(len(rope_segment)-1):
                pos = rope_segment[i]['pos']
                if i % 10 == 1:
                    prompt_points.append([pos[1], pos[0]])

            input_point = np.array(prompt_points)
            input_label = np.array([1,] * input_point.shape[0])

            ts = time.time()
            predictor.set_image(image=image4sam)
            masks, scores, logits = predictor.predict(point_coords=input_point,
                                                      point_labels=input_label,
                                                      multimask_output=True)
            mask_idx = np.argmax(scores)
            print(time.time() - ts, mask_idx)
            # for i, (mask, score) in enumerate(zip(masks, scores)):
            #     plt.figure(figsize=(5,5))
            #     plt.imshow(image4sam)
            #     show_mask(mask, plt.gca())
            #     show_points(input_point, input_label, plt.gca(), marker_size=25)
            #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            #     plt.axis('off')
            #     plt.show()  
            #     break
            mask = masks[mask_idx]
            keypoints = extract_uniform_keypoints(mask, num_keypoints=50)
            # 可视化结果
            j = 0
            for keypoint in keypoints:
                image = cv2.circle(canvas1, (keypoint[1], keypoint[0]), radius=2, color=(255, 5 + 3 * j, 255-3*j), thickness=-1)
                # plt.plot([pos[1], pos_next[1]], [pos[0], pos_next[0]])
                j += 1

            for input_point_ in input_point:
                image = cv2.circle(canvas1, (input_point_[0], input_point_[1]), radius=5, color=(255, 0, 0), thickness=-1)

            cv2.imshow("rope_state", canvas1)
            cv2.waitKey(10)
        # print('-' * 20)
        frame += 1
        

