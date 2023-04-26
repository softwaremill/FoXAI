"""Example of running XAI on YOLOv5 on Object Detection task."""

import os
from argparse import ArgumentParser, Namespace

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
from foxai.visualizer import object_detection_visualization


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
    final_image = object_detection_visualization(
        detections=outputs,
        input_image=input_image,
    )
    img_name = f"{args.output_dir}/yolo_gradcam.png"
    output_path = f"./{img_name}"
    os.makedirs(".", exist_ok=True)
    print(f"[INFO] Saving the final image at {output_path}")
    cv2.imwrite(output_path, final_image)


if __name__ == "__main__":
    main()
