"""
Based on code: https://github.com/pooya-mohammadi/yolov5-gradcam.
"""

import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision
from deep_utils.utils.box_utils.boxes import Box
from torch import nn
from yolo_models.model import WrapperYOLOv5ObjectDetectionModel
from yolo_models.utils import box_iou, letterbox, xywh2xyxy


class BaseObjectDetector(nn.Module, ABC):
    """Base Custom ObjectDetector class which returns predictions with logits to explain."""

    @abstractmethod
    def get_model(self) -> nn.Module:
        ...

    @abstractmethod
    def get_names(self) -> List[str]:
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
        conf_thres: float = 0.6,
        iou_thres: float = 0.45,
        classes: Optional[List[str]] = None,
        agnostic: bool = False,
        multi_label: bool = False,
        labels: Tuple[str] = (),
        max_det: int = 300,
    ):
        """Runs Non-Maximum Suppression (NMS) on inference and logits results

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls] and pruned input logits (n, number-classes)
        """

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert (
            0 <= conf_thres <= 1
        ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert (
            0 <= iou_thres <= 1
        ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

        # Settings
        max_wh = 4096  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        logits_output = [torch.zeros((0, 80), device=logits.device)] * logits.shape[0]
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
            # log_ *= x[:, 4:5]
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                # log_ = x[:, 5:]
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
                log_ = log_[conf.view(-1) > conf_thres]
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
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
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

    @staticmethod
    def yolo_resize(
        img: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
        auto: bool = True,
        scale_fill: bool = False,
        scaleup: bool = True,
    ):
        return letterbox(
            img,
            new_shape=new_shape,
            color=color,
            auto=auto,
            scale_fill=scale_fill,
            scaleup=scaleup,
        )

    def forward(
        self,
        img: torch.Tensor,
    ) -> Tuple[
        Tuple[
            List[List[int]],
            List[List[int]],
            List[List[str]],
            List[List[float]],
        ],
        List[torch.Tensor],
    ]:
        """Forward pass of the network.

        Args:
            img: Image to process.

        Returns:
            Tuple of 2 values, first is tuple of predictions containing bounding-boxes,
            class number, class name and confidence; second value is list of tensor
            with logits per each detection.
        """
        prediction, logits, _ = self.get_model()(img)
        prediction, logits = self.non_max_suppression(
            prediction,
            logits,
            self.get_confidence(),
            self.get_iou_thresh(),
            classes=None,
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
                    boxes[i].append(bbox)
                    confidences[i].append(round(conf.item(), 2))
                    cls = int(cls.item())
                    classes[i].append(cls)
                    if names is not None:
                        class_names[i].append(names[cls])
                    else:
                        class_names[i].append(cls)
        return [boxes, classes, class_names, confidences], logits

    def preprocessing(self, img: np.ndarray) -> torch.Tensor:
        if len(img.shape) != 4:
            img = np.expand_dims(img, axis=0)
        im0 = img.astype(np.uint8)
        img = np.array(
            [self.yolo_resize(im, new_shape=self.get_img_size())[0] for im in im0]
        )
        img = img.transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
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

    def get_img_size(self) -> Tuple[int, int]:
        return self.img_size

    def get_confidence(self) -> float:
        return self.confidence

    def get_iou_thresh(self) -> float:
        return self.iou_thresh

    def get_agnostic(self) -> bool:
        return self.agnostic
