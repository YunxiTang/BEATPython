import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

def extract_uniform_keypoints(mask, num_keypoints=10):
    # Step 1: 预处理 Mask（转换为二值图像）
    mask = (mask > 0).astype(np.uint8)  # 确保是二值
    plt.imshow(mask, cmap="gray")
    plt.show()

    mask = cv2.erode(mask, kernel=np.ones((4, 4), dtype=np.uint8), iterations=1)
    mask = cv2.dilate(mask, kernel=np.ones((4, 4), dtype=np.uint8), iterations=1)
    plt.imshow(mask, cmap="gray")
    plt.show()

    # Step 2: 骨架化（得到单像素宽的中心线）
    skeleton = skeletonize(mask, method='lee').astype(np.uint8)
    plt.imshow(skeleton, cmap="gray")
    plt.show()

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


if __name__ == '__main__':
    # 示例：加载 Mask 并提取关键点
    num_keypoint = 40
    mask = cv2.imread("./mask_img.jpg", cv2.IMREAD_GRAYSCALE)  # 读取灰度图
    keypoints = extract_uniform_keypoints(mask, num_keypoints=num_keypoint)

    # 可视化结果
    clrs = sns.color_palette("coolwarm", n_colors=num_keypoint).as_hex()
    plt.imshow(mask, cmap="gray")
    
    plt.scatter(keypoints[:, 1], keypoints[:, 0], c='r', s=20)
    plt.show()
