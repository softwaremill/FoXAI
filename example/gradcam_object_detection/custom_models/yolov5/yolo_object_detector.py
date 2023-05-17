"""File contains YOLOv5 ObjectDetector class."""
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision
from torchvision.ops.boxes import box_convert, box_iou

from foxai.explainer.computer_vision.object_detection.base_object_detector import (
    BaseObjectDetector,
)
from foxai.explainer.computer_vision.object_detection.types import PredictionOutput

from .model import WrapperYOLOv5ObjectDetectionModel


class YOLOv5ObjectDetector(BaseObjectDetector):
    """Custom YOLOv5 ObjectDetector class which returns predictions with logits to explain.

    Based on code: https://github.com/pooya-mohammadi/yolov5-gradcam.
    """

    def __init__(
        self,
        model: WrapperYOLOv5ObjectDetectionModel,
        img_size: Tuple[int, int],
        names: List[str],
        mode: str = "eval",
        confidence: float = 0.4,
        iou_thresh: float = 0.45,
        agnostic_nms: bool = False,
    ):
        super().__init__()
        self.device = model.device

        # in this case model on __call__ or on forward function has to return tuple of 3 variables:
        # 1st would be prediction tensor of shape [bs, x, number_of_classes + 5]
        # 2nd would be logits tensor of shape [bs, x, number_of_classes]
        # 3rd is discarded in this class
        # where x is number of hidden sizes of target layer
        self.img_size = img_size
        self.mode = mode
        self.confidence = confidence
        self.iou_thresh = iou_thresh
        self.agnostic = agnostic_nms
        self.names = names
        self.model = model
        self.model.requires_grad_(True)
        self.model.to(self.device)
        if self.mode == "train":
            self.model.train()
        else:
            self.model.eval()

        # preventing cold start
        img = torch.zeros((1, 3, *self.img_size), device=self.device)
        self.model(img)

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
            max_det: Maximum detections. Defaults to 300.

        Returns:
           Tuple of list of detections, on (n, 6) tensor per image [xyxy, conf, cls]
            and list of pruned input logits tensor (n, number-classes).
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

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            box = box_convert(x[:, :4], in_fmt="cxcywh", out_fmt="xyxy")

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
        image: torch.Tensor,
    ) -> Tuple[List[PredictionOutput], List[torch.Tensor]]:
        """Forward pass of the network.

        Args:
            image: Image to process.

        Returns:
            Tuple of 2 values, first is tuple of predictions containing bounding-boxes,
            class number, class name and confidence; second value is list of tensors
            with logits per each detection.
        """
        prediction, logits = self.model(image)
        prediction, logits = self.non_max_suppression(
            prediction=prediction,
            logits=logits,
            number_of_classes=len(self.names),
            confidence_threshold=self.confidence,
            iou_threshold=self.iou_thresh,
            agnostic=self.agnostic,
        )
        boxes: List[List[np.ndarray]] = []
        class_names: List[List[str]] = []
        classes: List[List[int]] = []
        confidences: List[List[float]] = []
        boxes, class_names, classes, confidences = [  # type: ignore
            [[] for _ in range(image.shape[0])] for _ in range(4)
        ]
        for i, det in enumerate(prediction):  # detections per image
            if len(det):
                for *xyxy, conf, cls in det:
                    bbox_raw: List[torch.Tensor] = [
                        xyxy[1],
                        xyxy[0],
                        xyxy[3],
                        xyxy[2],
                    ]
                    # TODO: line below is used to test detectio on resized image when
                    # width:height ratio was changed bboxes have negative coordinates
                    bbox: np.ndarray = np.array(
                        [np.abs(int(b.item())) for b in bbox_raw]
                    )
                    boxes[i].append(bbox)
                    confidences[i].append(round(conf.item(), 2))
                    cls = int(cls.item())
                    classes[i].append(cls)
                    if self.names is not None:
                        class_names[i].append(self.names[cls])
                    else:
                        class_names[i].append(cls)
        return (
            [
                PredictionOutput(
                    bbox=list(bbox),
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


def get_yolo_layer(model: BaseObjectDetector, layer_name: str) -> torch.nn.Module:
    """Obtain YOLOv5 layer from model by name.

    Args:
        model: YOLOv5 model.
        layer_name: The name of the layer with its hierarchical information.

    Return:
        Model's layer.
    """
    if not isinstance(model.model, torch.nn.Module) or not isinstance(
        model.model.model, torch.nn.Module
    ):
        raise RuntimeError("Model is not valid YOLOv5 model.")

    hierarchy = layer_name.split("_")
    target_layer: Optional[torch.nn.Module] = None
    target_layer = model.model.model._modules[  # pylint: disable = (protected-access)
        hierarchy[0]
    ]

    for h in hierarchy[1:]:
        if not isinstance(target_layer, torch.nn.Module):
            raise RuntimeError("Selected layer is not present in network.")

        if target_layer._modules is not None:  # pylint: disable = (protected-access)
            target_layer = target_layer._modules[h]  # pylint: disable = W0212
        else:
            raise RuntimeError("Selected layer is not present in network.")

    if target_layer is not None:
        return target_layer
    else:
        raise RuntimeError("Selected layer is not present in network.")
