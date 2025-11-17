import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

import cv2
import torch
import requests
import numpy as np
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

import os
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
import seaborn as sns

# os.environ['TORCH_CUDA_ARCH_LIST'] = True


def extract_uniform_keypoints(mask, num_keypoints=10):
    # Step 1: 预处理 Mask（转换为二值图像）
    mask = (mask > 0).astype(np.uint8)  # 确保是二值
    # plt.imshow(mask, cmap="gray")
    # plt.show()

    mask = cv2.erode(mask, kernel=np.ones((6, 6), dtype=np.uint8), iterations=1)
    mask = cv2.dilate(mask, kernel=np.ones((6, 6), dtype=np.uint8), iterations=1)
    # plt.imshow(mask, cmap="gray")
    # plt.show()

    # Step 2: 骨架化（得到单像素宽的中心线）
    skeleton = skeletonize(mask, method="lee").astype(np.uint8)
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
        next_point = min(
            remaining_points, key=lambda p: np.linalg.norm(np.array(p) - last_point)
        )
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


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


def annotate(
    image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]
) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(
            image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2
        )
        cv2.putText(
            image_cv2,
            f"{label}: {score:.2f}",
            (box.xmin, box.ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color.tolist(),
            2,
        )

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)


def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None,
) -> None:
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis("off")
    if save_name:
        plt.savefig(save_name, bbox_inches="tight")
    plt.show()


def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        "aliceblue",
        "antiquewhite",
        "aqua",
        "aquamarine",
        "azure",
        "beige",
        "bisque",
        "black",
        "blanchedalmond",
        "blue",
        "blueviolet",
        "brown",
        "burlywood",
        "cadetblue",
        "chartreuse",
        "chocolate",
        "coral",
        "cornflowerblue",
        "cornsilk",
        "crimson",
        "cyan",
        "darkblue",
        "darkcyan",
        "darkgoldenrod",
        "darkgray",
        "darkgreen",
        "darkgrey",
        "darkkhaki",
        "darkmagenta",
        "darkolivegreen",
        "darkorange",
        "darkorchid",
        "darkred",
        "darksalmon",
        "darkseagreen",
        "darkslateblue",
        "darkslategray",
        "darkslategrey",
        "darkturquoise",
        "darkviolet",
        "deeppink",
        "deepskyblue",
        "dimgray",
        "dimgrey",
        "dodgerblue",
        "firebrick",
        "floralwhite",
        "forestgreen",
        "fuchsia",
        "gainsboro",
        "ghostwhite",
        "gold",
        "goldenrod",
        "gray",
        "green",
        "greenyellow",
        "grey",
        "honeydew",
        "hotpink",
        "indianred",
        "indigo",
        "ivory",
        "khaki",
        "lavender",
        "lavenderblush",
        "lawngreen",
        "lemonchiffon",
        "lightblue",
        "lightcoral",
        "lightcyan",
        "lightgoldenrodyellow",
        "lightgray",
        "lightgreen",
        "lightgrey",
        "lightpink",
        "lightsalmon",
        "lightseagreen",
        "lightskyblue",
        "lightslategray",
        "lightslategrey",
        "lightsteelblue",
        "lightyellow",
        "lime",
        "limegreen",
        "linen",
        "magenta",
        "maroon",
        "mediumaquamarine",
        "mediumblue",
        "mediumorchid",
        "mediumpurple",
        "mediumseagreen",
        "mediumslateblue",
        "mediumspringgreen",
        "mediumturquoise",
        "mediumvioletred",
        "midnightblue",
        "mintcream",
        "mistyrose",
        "moccasin",
        "navajowhite",
        "navy",
        "oldlace",
        "olive",
        "olivedrab",
        "orange",
        "orangered",
        "orchid",
        "palegoldenrod",
        "palegreen",
        "paleturquoise",
        "palevioletred",
        "papayawhip",
        "peachpuff",
        "peru",
        "pink",
        "plum",
        "powderblue",
        "purple",
        "rebeccapurple",
        "red",
        "rosybrown",
        "royalblue",
        "saddlebrown",
        "salmon",
        "sandybrown",
        "seagreen",
        "seashell",
        "sienna",
        "silver",
        "skyblue",
        "slateblue",
        "slategray",
        "slategrey",
        "snow",
        "springgreen",
        "steelblue",
        "tan",
        "teal",
        "thistle",
        "tomato",
        "turquoise",
        "violet",
        "wheat",
        "white",
        "whitesmoke",
        "yellow",
        "yellowgreen",
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon


def polygon_to_mask(
    polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")
    image = image.resize((640, 480))
    return image


def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]


def refine_masks(
    masks: torch.BoolTensor, polygon_refinement: bool = False
) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


def plot_detections_plotly(
    image: np.ndarray,
    detections: List[DetectionResult],
    class_colors: Optional[Dict[str, str]] = None,
) -> None:
    # If class_colors is not provided, generate random colors for each class
    if class_colors is None:
        num_detections = len(detections)
        colors = random_named_css_colors(num_detections)
        class_colors = {}
        for i in range(num_detections):
            class_colors[i] = colors[i]

    fig = px.imshow(image)

    # Add bounding boxes
    shapes = []
    annotations = []
    for idx, detection in enumerate(detections):
        label = detection.label
        box = detection.box
        score = detection.score
        mask = detection.mask

        polygon = mask_to_polygon(mask)

        fig.add_trace(
            go.Scatter(
                x=[point[0] for point in polygon] + [polygon[0][0]],
                y=[point[1] for point in polygon] + [polygon[0][1]],
                mode="lines",
                line=dict(color=class_colors[idx], width=2),
                fill="toself",
                name=f"{label}: {score:.2f}",
            )
        )

        xmin, ymin, xmax, ymax = box.xyxy
        shape = [
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=xmin,
                y0=ymin,
                x1=xmax,
                y1=ymax,
                line=dict(color=class_colors[idx]),
            )
        ]
        annotation = [
            dict(
                x=(xmin + xmax) // 2,
                y=(ymin + ymax) // 2,
                xref="x",
                yref="y",
                text=f"{label}: {score:.2f}",
            )
        ]

        shapes.append(shape)
        annotations.append(annotation)

    # Update layout
    button_shapes = [dict(label="None", method="relayout", args=["shapes", []])]
    button_shapes = button_shapes + [
        dict(label=f"Detection {idx + 1}", method="relayout", args=["shapes", shape])
        for idx, shape in enumerate(shapes)
    ]
    button_shapes = button_shapes + [
        dict(label="All", method="relayout", args=["shapes", sum(shapes, [])])
    ]

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        # margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        updatemenus=[dict(type="buttons", direction="up", buttons=button_shapes)],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Show plot
    fig.show()


def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    object_detector: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    labels = [label if label.endswith(".") else label + "." for label in labels]
    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]
    filtered_results = []
    for res in results:
        if res.label == "rope.":
            filtered_results.append(res)
    return filtered_results


def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    device,
    polygon_refinement: bool = False,
    segmentator=None,
    processor: Optional[str] = None,
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes,
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results


def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    object_detector: Optional[str] = None,
    processor=None,
    segmentator: Optional[str] = None,
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)
    # print(image)
    detections = detect(image, labels, threshold, object_detector)

    detections = segment(
        image, detections, device, polygon_refinement, segmentator, processor
    )

    return np.array(image), detections


if __name__ == "__main__":
    import time
    import cv2
    import zarr

    device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"

    print(device)
    labels = ["rope."]
    threshold = 0.1

    detector_id = "IDEA-Research/grounding-dino-base"
    segmenter_id = "facebook/sam-vit-large"

    print("load segmentator")
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"
    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id, use_fast=False)

    print("load detector")
    detector_id = (
        detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    )
    object_detector = pipeline(
        model=detector_id,
        task="zero-shot-object-detection",
        device=device,
        use_fast=True,
    )

    root = zarr.open(
        "/media/yxtang/Extreme SSD/GenDloRec/real_world/realworld_dataset-0-20250304.zarr"
    )
    rgb_dataset = root["data"]["rgb"]

    num_keypoint = 50

    i = 0

    while True:
        source_img = rgb_dataset[i][:, :, 0:3]

        if i % 30 == 0:
            cv2.imshow("source_img", source_img)
            cv2.waitKey(10)

            # 转换 BGR -> RGB
            # img_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

            # 转换为 PIL Image
            img_pil = Image.fromarray(source_img)
            ts = time.time()
            image_array, detections = grounded_segmentation(
                image=img_pil,
                labels=labels,
                threshold=threshold,
                polygon_refinement=False,
                object_detector=object_detector,
                processor=processor,
                segmentator=segmentator,
            )

            # print(detections)
            # print(time.time() - ts)
            # plot_detections(image_array, detections, None)

            mask = detections[0].mask
            plt.imshow(mask)
            plt.show()

            keypoints = extract_uniform_keypoints(mask, num_keypoints=num_keypoint)
            # 可视化结果
            j = 0
            for keypoint in keypoints:
                image = cv2.circle(
                    source_img,
                    (keypoint[1], keypoint[0]),
                    radius=2,
                    color=(5, 5 + 3 * j, 255 - 3 * j),
                    thickness=-1,
                )
                # plt.plot([pos[1], pos_next[1]], [pos[0], pos_next[0]])
                j += 1
            cv2.imshow("output", image)
            cv2.waitKey(100)
        i += 1

        # mask_img = Image.fromarray(detections[0].mask * 255)
        # mask_img.save('./mask_img.jpg')
