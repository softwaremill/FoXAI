"""
Based on code: https://github.com/pooya-mohammadi/yolov5-gradcam.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision
from deep_utils.utils.box_utils.boxes import Box
from torch import nn
from yolo_models.model import WrapperYOLOv5ObjectDetectionModel
from yolo_models.utils import box_iou, resize_image, xywh2xyxy


@dataclass
class PredictionOutput:
    """Data class for model prediction output in YOLO style."""

    bbox: List[List[int]]
    class_number: List[List[int]]
    class_name: List[List[str]]
    confidence: List[List[float]]


class BaseObjectDetector(nn.Module, ABC):
    """Base ObjectDetector class which returns predictions with logits to explain.

    Code based on https://github.com/pooya-mohammadi/yolov5-gradcam.
    """

    @abstractmethod
    def get_model(self) -> nn.Module:
        ...

    @abstractmethod
    def get_names(self) -> List[str]:
        ...

    @abstractmethod
    def get_number_of_classes(self) -> int:
        ...

    @abstractmethod
    def get_img_size(self) -> Tuple[int, int]:
        ...

    @abstractmethod
    def get_confidence(self) -> float:
        ...

    @abstractmethod
    def get_iou_thresh(self) -> float:
        ...

    @abstractmethod
    def get_agnostic(self) -> bool:
        ...

    @staticmethod
    def non_max_suppression(
        prediction: torch.Tensor,
        logits: torch.Tensor,
        number_of_classes: int,
        confidence_threshold: float = 0.6,
        iou_threshold: float = 0.45,
        classes: Optional[List[str]] = None,
        agnostic: bool = False,
        multi_label: bool = False,
        labels: Tuple[str] = (),
        max_det: int = 300,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Runs Non-Maximum Suppression (NMS) on inference and logits results.

        Code based on https://github.com/pooya-mohammadi/yolov5-gradcam.

        Args:
            prediction: Predictions from model forward pass.
            logits: Logits from model forward pass.
            number_of_classes: Number of classes that model can predict.
            confidence_threshold: Confidence threshold. Defaults to 0.6.
            iou_threshold: IoU threshold. Defaults to 0.45.
            classes: List of class names. Defaults to None.
            agnostic: If True it restricts max width of detection. Defaults to False.
            multi_label: If True function will assign multiple classes to single
                detection. Defaults to False.
            labels: If not empty function will process only detection of objects from
                this list of classes. Defaults to ().
            max_det: Maximum detections. Defaults to 300.

        Returns:
           Tuple of list of detections, on (n, 6) tensor per image [xyxy, conf, cls]
            and list of pruned input logits tensors (n, number-classes).
        """

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > confidence_threshold  # candidates

        # Checks
        assert (
            0 <= confidence_threshold <= 1
        ), f"Invalid Confidence threshold {confidence_threshold}, valid values are between 0.0 and 1.0"
        assert (
            0 <= iou_threshold <= 1
        ), f"Invalid IoU {iou_threshold}, valid values are between 0.0 and 1.0"

        # Settings
        max_wh = 4096  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        logits_output = [
            torch.zeros((0, number_of_classes), device=logits.device)
        ] * logits.shape[0]
        for xi, (x, log_) in enumerate(
            zip(prediction, logits)
        ):  # image index, image inference
            x = x[xc[xi]]  # confidence
            log_ = log_[xc[xi]]
            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > confidence_threshold).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[
                    conf.view(-1) > confidence_threshold
                ]
                log_ = log_[conf.view(-1) > confidence_threshold]
            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_threshold  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                    1, keepdim=True
                )  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            logits_output[xi] = log_[i]
            assert log_[i].shape[0] == x[i].shape[0]
            if (time.time() - t) > time_limit:
                print(f"WARNING: NMS time limit {time_limit}s exceeded")
                break  # time limit exceeded

        return output, logits_output

    def forward(
        self,
        img: torch.Tensor,
    ) -> Tuple[List[PredictionOutput], List[torch.Tensor],]:
        """Forward pass of the network.

        Args:
            img: Image to process.

        Returns:
            Tuple of 2 values, first is tuple of predictions containing bounding-boxes,
            class number, class name and confidence; second value is list of tensor
            with logits per each detection.
        """
        prediction, logits = self.get_model()(img)
        prediction, logits = self.non_max_suppression(
            prediction=prediction,
            logits=logits,
            number_of_classes=self.get_number_of_classes(),
            confidence_threshold=self.get_confidence(),
            iou_threshold=self.get_iou_thresh(),
            agnostic=self.get_agnostic(),
        )
        boxes, class_names, classes, confidences = [
            [[] for _ in range(img.shape[0])] for _ in range(4)
        ]
        names = self.get_names()
        for i, det in enumerate(prediction):  # detections per image
            if len(det):
                for *xyxy, conf, cls in det:
                    bbox = Box.box2box(
                        xyxy,
                        in_source=Box.BoxSource.Torch,
                        to_source=Box.BoxSource.Numpy,
                        return_int=True,
                    )
                    # TODO: line below is used to test detectio on resized image when
                    # width:height ratio was changed bboxes have negative coordinates
                    bbox = [np.abs(b) for b in bbox]
                    boxes[i].append(bbox)
                    confidences[i].append(round(conf.item(), 2))
                    cls = int(cls.item())
                    classes[i].append(cls)
                    if names is not None:
                        class_names[i].append(names[cls])
                    else:
                        class_names[i].append(cls)
        return (
            [
                PredictionOutput(
                    bbox=bbox,
                    class_number=class_no,
                    class_name=class_name,
                    confidence=confidence,
                )
                for bbox, class_no, class_name, confidence in zip(
                    boxes[0], classes[0], class_names[0], confidences[0]
                )
            ],
            logits,
        )

    @staticmethod
    def preprocessing(
        img: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        change_original_ratio: bool = False,
        scaleup: bool = True,
    ) -> torch.Tensor:
        """Preprocess image before prediction.

        Preprocessing is a process consisting of steps:
        * adding batch dimension
        * resizing images to desired shapes
        * adjusting image channels to (B x C x H x W)
        * convertion to float

        Args:
            img: Image to preprocess.
            new_shape: Desired shape of image. Defaults to (640, 640).
            change_original_ratio: If resized image should have different height to
                width ratio than original image. Defaults to False.
            scaleup: If scale up image. Defaults to True.

        Returns:
            Tensor containing preprocessed image.
        """
        if len(img.shape) != 4:
            # add batch dimension
            img = np.expand_dims(img, axis=0)

        # resize all images from batch
        img = np.array(
            [
                resize_image(
                    image=im,
                    new_shape=new_shape,
                    change_original_ratio=change_original_ratio,
                    scaleup=scaleup,
                )
                for im in img
            ]
        )
        # convert array from (B x H x W x C) to (B x C x H x W)
        img = img.transpose((0, 3, 1, 2))
        img = torch.from_numpy(img)  # .to(self.device)

        # convert from uint8 to float
        img = img / 255.0
        return img


class YOLOv5ObjectDetector(BaseObjectDetector):
    """Custom YOLOv5 ObjectDetector class which returns predictions with logits to explain."""

    def __init__(
        self,
        model: WrapperYOLOv5ObjectDetectionModel,
        device: torch.DeviceObjType,
        img_size: Tuple[int, int],
        names: Optional[List[str]] = None,
        mode: str = "eval",
        confidence: float = 0.4,
        iou_thresh: float = 0.45,
        agnostic_nms: bool = False,
    ):
        super().__init__()
        self.device = device

        # in this case model on __call__ or on forward function has to return tuple of 3 variables:
        # 1st would be prediction tensor of shape [bs, x, number_of_classes + 5]
        # 2nd would be logits tensor of shape [bs, x, number_of_classes]
        # 3rd is discarded in this class
        # where x is number of hidden sizes of target layer
        self.model = None

        self.img_size = img_size
        self.mode = mode
        self.confidence = confidence
        self.iou_thresh = iou_thresh
        self.agnostic = agnostic_nms
        self.model = model
        self.model.requires_grad_(True)
        self.model.to(device)
        if self.mode == "train":
            self.model.train()
        else:
            self.model.eval()
        # fetch the names
        if names is None:
            self.names = [
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "couch",
                "potted plant",
                "bed",
                "dining table",
                "toilet",
                "tv",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush",
            ]
        else:
            self.names = names

        # preventing cold start
        img = torch.zeros((1, 3, *self.img_size), device=device)
        self.model(img)

    def get_model(self) -> nn.Module:
        return self.model

    def get_names(self) -> List[str]:
        return self.names

    def get_number_of_classes(self) -> int:
        return len(self.names)

    def get_img_size(self) -> Tuple[int, int]:
        return self.img_size

    def get_confidence(self) -> float:
        return self.confidence

    def get_iou_thresh(self) -> float:
        return self.iou_thresh

    def get_agnostic(self) -> bool:
        return self.agnostic


def get_yolo_layer(model: BaseObjectDetector, layer_name: str) -> torch.nn.Module:
    """Obtain YOLOv5 layer from model by name.

    Args:
        model: YOLOv5 model.
        layer_name: The name of the layer with its hierarchical information.

    Return:
        Model's layer.
    """
    hierarchy = layer_name.split("_")
    target_layer = model.model.model._modules[  # pylint: disable = (protected-access)
        hierarchy[0]
    ]

    for h in hierarchy[1:]:
        target_layer = target_layer._modules[h]  # pylint: disable = (protected-access)
    return target_layer
