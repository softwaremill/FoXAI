"""Example of running XAI on YOLOv5 on Object Detection task."""

import os
from argparse import ArgumentParser, Namespace
from typing import List

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models._meta import _COCO_CATEGORIES
from torchvision.models.detection import SSD300_VGG16_Weights

from foxai.explainer.computer_vision.object_detection.gradcam import (
    GradCAMObjectDetection,
    ObjectDetectionOutput,
)
from foxai.explainer.computer_vision.object_detection.models.ssd.ssd_object_detector import (
    SSDObjectDetector,
)
from foxai.explainer.computer_vision.object_detection.models.yolov5.model import (
    WrapperYOLOv5ObjectDetectionModel,
)
from foxai.explainer.computer_vision.object_detection.models.yolov5.yolo_object_detector import (
    YOLOv5ObjectDetector,
    get_yolo_layer,
)


def get_heatmap_bbox(
    heatmap: np.ndarray,
    bbox: np.ndarray,
    mask_value: int = 0,
) -> np.ndarray:
    """_summary_

    Code based on https://github.com/pooya-mohammadi/deep_utils/blob/main/deep_utils/utils/box_utils/boxes.py.

    Args:
        heatmap: Heatmap to visualize.
        bbox: Bounding box of detection.
        mask_value: Masking value . Defaults to 0.

    Returns:
        Numpy array with heatmap only present in area of given bounding box.
    """
    # fill the outer area of the selected box
    mask = np.ones_like(heatmap, dtype=np.uint8) * mask_value
    mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = 1
    masked_heatmap = cv2.multiply(heatmap, mask)
    return masked_heatmap


def draw_heatmap_in_bbox(
    bbox: List[int],
    heatmap: torch.Tensor,
    img: np.ndarray,
) -> np.ndarray:
    """Draw heatmap in bounding box on image.

    Args:
        bbox: List of coordinates for bounding box.
        heatmap: Heatmap to display.
        img: Original image.

    Returns:
        Image with displayed heatmap in bounding box area.
    """
    heatmap = (
        heatmap.squeeze(0)
        .mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    masked_heatmap = get_heatmap_bbox(heatmap=heatmap, bbox=bbox).astype(np.float32)

    img = cv2.add(img, masked_heatmap)
    img = img / img.max()
    img = (img * 255).astype(np.uint8)
    return img


def concat_images(images: List[np.ndarray]) -> np.ndarray:
    """Concatenate images into one.

    Args:
        images: List of images to merge.

    Returns:
        Final image.
    """
    w, h = images[0].shape[:2]
    width = w
    height = h * len(images)
    base_img = np.zeros((width, height, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        base_img[:, h * i : h * (i + 1), ...] = img

    return base_img


def parse_args() -> Namespace:
    """Get CLI arguments.

    Returns:
        Namespace with arguments.
    """
    parser = ArgumentParser(prog="GradCAM YOLOv5 example")
    parser.add_argument(
        "--model_name", type=str, default="yolov5s", help="YOLOv5 model name"
    )
    parser.add_argument(
        "--layer_name",
        type=str,
        default="model_23_cv3_act",
        help="YOLOv5 target layer name",
    )
    parser.add_argument("--img_path", type=str, required=True, help="Image to explain")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where explanation image will be saved",
    )
    return parser.parse_args()


def main():
    """Run YOLO_v5 XAI and save results."""
    args = parse_args()

    image = Image.open(args.img_path)
    img_size = (640, 640)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    org_input_image = YOLOv5ObjectDetector.preprocessing(
        img=np.asarray(image),
        new_shape=img_size,
        change_original_ratio=False,
    ).to(device)

    real_input_image_shape = org_input_image.shape[-2:]

    if "ssd" in args.model_name:
        weights = SSD300_VGG16_Weights.COCO_V1
        model = (
            torchvision.models.detection.ssd300_vgg16(weights=weights, pretrained=True)
            .eval()
            .to(device)
        )
        preprocess = weights.transforms()
        model.detections_per_img = 10
        input_image = preprocess(org_input_image).to(device)

        model_wrapper = SSDObjectDetector(model=model, class_names=_COCO_CATEGORIES)
        target_layer = model_wrapper.model.backbone.features[-1]
    else:
        model = torch.hub.load("ultralytics/yolov5", args.model_name, pretrained=True)
        names = model.model.names
        wrapper_model = WrapperYOLOv5ObjectDetectionModel(
            model=model.model.model,
            device=device,
        )
        model_wrapper = YOLOv5ObjectDetector(
            model=wrapper_model,
            device=device,
            img_size=real_input_image_shape,
            names=names,
        )
        target_layer = get_yolo_layer(
            model=model_wrapper,
            layer_name=args.layer_name,
        )

        input_image = org_input_image

    saliency_method = GradCAMObjectDetection(
        model=model_wrapper,
        target_layer=target_layer,
    )
    outputs: ObjectDetectionOutput = saliency_method(input_img=input_image)
    masks = outputs.saliency_maps
    boxes = [pred.bbox for pred in outputs.predictions]
    class_names = [pred.class_name for pred in outputs.predictions]
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
        bbox, cls_name = boxes[i], class_names[i]
        bbox = [int(val) for val in bbox]
        res_img = draw_heatmap_in_bbox(bbox, mask, res_img)
        res_img_tensor = torch.tensor(res_img).transpose(0, 2).transpose(1, 2)
        bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
        res_img_tensor = torchvision.utils.draw_bounding_boxes(
            image=res_img_tensor,
            boxes=torch.tensor([bbox]),
            labels=[cls_name],
        )
        res_img = res_img_tensor.transpose(1, 2).transpose(0, 2).numpy()
        images.append(res_img)

    final_image = concat_images(images)
    img_name = f"{args.output_dir}/yolo_gradcam.png"
    output_path = f"./{img_name}"
    os.makedirs(".", exist_ok=True)
    print(f"[INFO] Saving the final image at {output_path}")
    cv2.imwrite(output_path, final_image)


if __name__ == "__main__":
    main()
