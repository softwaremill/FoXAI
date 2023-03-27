"""Example of running XAI on YOLOv5 on Object Detection task."""

import os
from typing import List

import cv2
import numpy as np
import torch
from deep_utils import Box
from PIL import Image
from yolo_models.gradcam import GradCAMObjectDetection
from yolo_models.model import WrapperYOLOv5ObjectDetectionModel

from example.gradcam_yolo.yolo_models.object_detector import (
    YOLOv5ObjectDetector,
    find_yolo_layer,
)


def get_res_img(bbox: List[int], mask: torch.Tensor, res_img: np.ndarray) -> np.ndarray:
    mask = (
        mask.squeeze(0)
        .mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    n_heatmat = (Box.fill_outer_box(heatmap, bbox) / 255).astype(np.float32)
    res_img = res_img / 255
    res_img = cv2.add(res_img, n_heatmat)
    res_img = res_img / res_img.max()
    return res_img, n_heatmat


def put_text_box(
    bbox: List[int], cls_name: str, res_img: np.ndarray, tmp_path: str = "temp.jpg"
) -> np.ndarray:
    x1, y1, _, _ = bbox
    # this is a bug in cv2. It does not put box on a converted image from torch unless it's buffered and read again!
    cv2.imwrite(tmp_path, (res_img * 255).astype(np.uint8))
    res_img = cv2.imread(tmp_path)
    res_img = Box.put_box(res_img, bbox)
    res_img = Box.put_text(res_img, cls_name, (x1, y1))
    return res_img


def concat_images(images: List[np.ndarray]) -> np.ndarray:
    w, h = images[0].shape[:2]
    width = w
    height = h * len(images)
    base_img = np.zeros((width, height, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        base_img[:, h * i : h * (i + 1), ...] = img
    return base_img


def main():
    """Run YOLO_v5 XAI and save results."""

    image_path = "example/images/bus.jpg"
    image = Image.open(image_path)
    img_size = (640, 480)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    wrapper_model = WrapperYOLOv5ObjectDetectionModel(
        model=model.model.model, device=device
    )
    model_wrapper = YOLOv5ObjectDetector(
        model=wrapper_model,
        device=device,
        img_size=img_size,
    )
    input_image = model_wrapper.preprocessing(img=np.asarray(image))

    target_layer = find_yolo_layer(
        model=model_wrapper,
        layer_name="model_23_cv3_act",  # last feature extractor layer in YOLOv5
    )
    saliency_method = GradCAMObjectDetection(
        model=model_wrapper,
        target_layer=target_layer,
        img_size=img_size,
    )
    masks, _, [boxes, _, class_names, _] = saliency_method(input_img=input_image)

    img_to_display = (
        input_image.squeeze(0)
        .mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )
    img_to_display = img_to_display[..., ::-1]  # convert to bgr
    images = [img_to_display]

    for i, mask in enumerate(masks):
        res_img = img_to_display.copy()
        bbox, cls_name = boxes[0][i], class_names[0][i]
        res_img, _ = get_res_img(bbox, mask, res_img)
        res_img = put_text_box(bbox, cls_name, res_img)
        images.append(res_img)
    final_image = concat_images(images)
    img_name = "example/gradcam_yolo/yolo_gradcam.png"
    output_path = f"./{img_name}"
    os.makedirs(".", exist_ok=True)
    print(f"[INFO] Saving the final image at {output_path}")
    cv2.imwrite(output_path, final_image)


if __name__ == "__main__":
    main()
