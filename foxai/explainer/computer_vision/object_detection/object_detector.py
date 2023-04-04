import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from deep_utils.utils.box_utils.boxes import Box
from torch import nn
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.ssd import SSD
from torchvision.ops import boxes as box_ops

from foxai.explainer.computer_vision.object_detection.model import (
    WrapperYOLOv5ObjectDetectionModel,
)
from foxai.explainer.computer_vision.object_detection.types import PredictionOutput
from foxai.explainer.computer_vision.object_detection.utils import (
    box_iou,
    resize_image,
    xywh2xyxy,
)


class BaseObjectDetector(nn.Module, ABC):
    """Base ObjectDetector class which returns predictions with logits to explain.

    Code based on https://github.com/pooya-mohammadi/yolov5-gradcam.
    """

    @abstractmethod
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
        img_tensor = torch.from_numpy(img)

        # convert from uint8 to float
        img_tensor = img_tensor / 255.0
        return img_tensor


class YOLOv5ObjectDetector(BaseObjectDetector):
    """Custom YOLOv5 ObjectDetector class which returns predictions with logits to explain.

    Based on code: https://github.com/pooya-mohammadi/yolov5-gradcam.
    """

    def __init__(
        self,
        model: WrapperYOLOv5ObjectDetectionModel,
        device: torch.device,
        img_size: Tuple[int, int],
        names: List[str],
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
        self.img_size = img_size
        self.mode = mode
        self.confidence = confidence
        self.iou_thresh = iou_thresh
        self.agnostic = agnostic_nms
        self.names = names
        self.model = model
        self.model.requires_grad_(True)
        self.model.to(device)
        if self.mode == "train":
            self.model.train()
        else:
            self.model.eval()

        # preventing cold start
        img = torch.zeros((1, 3, *self.img_size), device=device)
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
                    if self.names is not None:
                        class_names[i].append(self.names[cls])
                    else:
                        class_names[i].append(cls)
        return (
            [
                PredictionOutput(
                    bbox=bbox.tolist(),
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


class SSDObjectDetector(BaseObjectDetector):
    """Custom SSD ObjectDetector class which returns predictions with logits to explain.

    Code based on https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py.
    """

    def __init__(
        self,
        model: SSD,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.model = model
        self.class_names = class_names

    def forward(
        self,
        image: torch.Tensor,
    ) -> Tuple[List[PredictionOutput], List[torch.Tensor]]:
        """Forward pass of the network.

        Args:
            image: Image to process.

        Returns:
            Tuple of 2 values, first is tuple of predictions containing bounding-boxes,
            class number, class name and confidence; second value is tensor with logits
            per each detection.
        """
        # get the original image sizes
        images = [image[0]]
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert (
                len(val) == 2
            ), f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}"
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        image_list: ImageList
        image_list, targets = self.model.transform(images, None)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    assert False, (
                        "All bounding boxes should have positive height and width. "
                        + f"Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        # get the features from the backbone
        features = self.model.backbone(image_list.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.model.head(features)

        # create the set of anchors
        anchors = self.model.anchor_generator(image_list, features)

        detections: List[Dict[str, torch.Tensor]] = []
        detections, logits = self.postprocess_detections(
            head_outputs=head_outputs,
            image_anchors=anchors,
            image_shapes=image_list.image_sizes,
        )
        detections = self.model.transform.postprocess(
            detections, image_list.image_sizes, original_image_sizes
        )

        detection_class_names = [str(val.item()) for val in detections[0]["labels"]]
        if self.class_names:
            detection_class_names = [
                self.class_names[val.item()] for val in detections[0]["labels"]
            ]

        # change order of bounding boxes
        # at the moment they are [x2, y2, x1, y1] and we need them in
        # [x1, y1, x2, y2]
        detections[0]["boxes"] = detections[0]["boxes"].detach().cpu()
        for detection in detections[0]["boxes"]:
            tmp1 = detection[0].item()
            tmp2 = detection[2].item()
            detection[0] = detection[1]
            detection[2] = detection[3]
            detection[1] = tmp1
            detection[3] = tmp2

        predictions = [
            PredictionOutput(
                bbox=bbox.tolist(),
                class_number=class_no.item(),
                class_name=class_name,
                confidence=confidence.item(),
            )
            for bbox, class_no, class_name, confidence in zip(
                detections[0]["boxes"],
                detections[0]["labels"],
                detection_class_names,
                detections[0]["scores"],
            )
        ]

        return predictions, logits

    def postprocess_detections(
        self,
        head_outputs: Dict[str, torch.Tensor],
        image_anchors: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor]]:
        bbox_regression = head_outputs["bbox_regression"]
        logits = head_outputs["cls_logits"]
        pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)
        pred_class = torch.argmax(pred_scores[0], dim=1)
        pred_class = pred_class[None, :, None]

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections: List[Dict[str, torch.Tensor]] = []

        for boxes, scores, anchors, image_shape in zip(
            bbox_regression, pred_scores, image_anchors, image_shapes
        ):
            boxes = self.model.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_boxes: List[torch.Tensor] = []
            image_scores: List[torch.Tensor] = []
            image_labels: List[torch.Tensor] = []
            for label in range(1, num_classes):
                score = scores[:, label]

                keep_idxs = score > self.model.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = det_utils._topk_min(  # pylint: disable = (protected-access)
                    score, self.model.topk_candidates, 0
                )
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(
                    torch.full_like(
                        score, fill_value=label, dtype=torch.int64, device=device
                    )
                )

            image_box: torch.Tensor = torch.cat(image_boxes, dim=0)
            image_score: torch.Tensor = torch.cat(image_scores, dim=0)
            image_label: torch.Tensor = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(
                boxes=image_box,
                scores=image_score,
                idxs=image_label,
                iou_threshold=self.model.nms_thresh,
            )
            keep = keep[: self.model.detections_per_img]

            detections.append(
                {
                    "boxes": image_box[keep],
                    "scores": image_score[keep],
                    "labels": image_label[keep],
                }
            )
        # add batch dimension for further processing
        keep_logits = logits[0][keep][None, :]
        return detections, list(keep_logits)
