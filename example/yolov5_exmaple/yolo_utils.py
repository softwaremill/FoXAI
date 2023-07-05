import logging
import math
import time
from typing import Any, Iterator, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision
from typing_extensions import Final

from foxai.logger import create_logger

_LOGGER: Optional[logging.Logger] = None

MAXIMUM_BBOX_WIDTH_HEIGHT: Final[int] = 7680
"""Maximum width and height in pixels of the single bounding box."""

MAXIMUM_NUMBER_OF_BOXES_TO_NMS: Final[int] = 30000
"""Maximum number of bounding boxes that we can enter
into the torchvision non max suppression algorithm. torchvision.ops.nms()"""

MAXIMUM_NUMBER_OF_BOXES_TO_MERGE: Final[int] = 3e3
"""Maximum number of bounding boxes that can be merged into a single box."""


def log() -> logging.Logger:
    """Get or create logger."""
    # pylint: disable = global-statement
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = create_logger(__name__)
    return _LOGGER


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    x1y1x2y2: bool = True,
    GIoU: bool = False,  # pylint: disable = invalid-name
    DIoU: bool = False,  # pylint: disable = invalid-name
    CIoU: bool = False,  # pylint: disable = invalid-name
    eps: float = 1e-7,
):
    """Compute IoU of two bounding-boxes.

    Implementation based on eriklindernoren/PyTorch-YOLOv3/pytorchyolo/utils/utils.py

    Args:
        box1: bounding-box 1
        box2: bounding-box 2
        x1y1x2y2: True if top_left-bottom-right mode, False if xy_center-width_height mode
        GIoU: whether to use Generalized Intersection over Union:
        DIoU: whether to use Distance-IoU https://arxiv.org/abs/1911.08287v1
        CIoU: https://arxiv.org/abs/1911.08287v1
        eps: epsilon to be used to avoid dividing by 0.

    Returns:
        bounding-box IoU.
    """

    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def xywh2xyxy(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert nx4 boxes from [x, y, w, h] to
    [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.

    Original implementation in torch.hub.ultralytics_yolov5.utils.general.py

    Args:
        x: [x, y, w, h] tensor

    Returns:
        [x1, y1, x2, y2] tensor
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nm: int = 0,
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Original implementation in torch.hub.ultralytics_yolov5.utils.general.py

    Args:
        prediction: the model prediction
        conf_thres: confidence threshold, for counting the detection as valid
        iou_thres: intersection over union threshold for non max suppresion algorithm
        classes: classes to keep. Rest of the classes are rejected
        agnostic: if True, non max suppression algorithm is run on raw bboxes. However it may
            happen that different classes have bboxes in similar place. NMS would discard one of
            those bboxes and keep only the one with higher confidence. If we want to keep bboxes
            that are in similar place, but have different class label, we should set agnostic to False.
        multi_label: whether we want to keep multiple labels with its confidence for each bbox, or
            only pick the class label with highest confidence.
        labels: ground truth labels
        max_det: maximum number of detections
        nm: number of tensor elements not related to class prediction (xywh -> 4)

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections

    # bool, whether we want to keep overlapping bounding boxes with different class types
    # and a number of classes yolo, can detect is greater than 1.
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    # use merge-NMS. If true bboxes are merget with weighted average,
    # rather, than bboxes with low score ase discarded.
    merge = False

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    # pylint: disable = unnecessary-comprehension
    for xi, _ in enumerate([jj for jj in prediction]):
        x = prediction[xi]
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # Compute conf
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(
            x[:, :4]
        )  # (center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        if not torch.isfinite(x).all():
            x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[
            x[:, 4].argsort(descending=True)[:MAXIMUM_NUMBER_OF_BOXES_TO_NMS]
        ]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else MAXIMUM_BBOX_WIDTH_HEIGHT)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (
            1 < n < MAXIMUM_NUMBER_OF_BOXES_TO_MERGE
        ):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = bbox_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            log().warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def make_divisible(x: int, divisor: Union[int, torch.Tensor]) -> int:
    """Returns nearest x divisible by divisor.

    Original implementation in torch.hub.ultralytics_yolov5.utils.general.py

    Args:
        x: the number to be divided
        divisor: denominator

    Returns:
        the nearest x divisible by divisor
    """

    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def letterbox(
    im: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Resize and pad image while meeting stride-multiple constraints.

    Original implementation in torch.hub.ultralytics_yolov5.utils.general.py

    Args:
        im: the image to be resized and pad of shape (height,width,channels)
        new_shape: the destination shape
        color: padding color
        auto: whether to automaticlly compute the padding size from other parameters
        scale_fill: don't pad, just stratch the image
        scaleup: wheter to allow scaling up the image. If False, only scaling down is allowed (better mAP).
        stride: the yolo network stride

    Returns:
        resized and padded image
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r: float = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio: Tuple[float, float] = (r, r)  # width, height ratios
    new_unpad: Tuple[int, int] = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im: np.ndarray = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def scale_boxes(
    img1_shape: Tuple[int, ...],
    boxes: torch.Tensor,
    img0_shape: Tuple[int, ...],
    ratio_pad: Optional[Tuple[Tuple[float, float], Tuple[int, int]]] = None,
) -> torch.Tensor:
    """Rescale boxes (xyxy) from img1_shape to img0_shape.

    Original implementation in torch.hub.ultralytics_yolov5.utils.general.py

    Args:
        img1_shape: shape of resized image
        boxes: the bbox size
        img0_shape: shape of original image
        ratio_pad: new/old image ratio of letter box algorithm

    Returns:
        scaled bbox to the img0_shape
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes: Union[torch.Tensor, np.ndarray], shape: Tuple[int, int]) -> None:
    """Clip boxes (xyxy) to image shape (height, width)

    Original implementation in torch.hub.ultralytics_yolov5.utils.general.py

    Args:
        boxes: bboxes to be clipped
        shape: maximum x and y size

    Return:
        inplace clipped bbox
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def get_variables(
    model: torch.nn.Module,
    include: Tuple[str, ...] = (),
    exclude: Tuple[str, ...] = (),
) -> Iterator[Tuple[str, Any]]:
    """Copy attributes from b to a, options to only include [...] and to exclude [...]

    Based on ultralytics/yolov3/utils/torch_utils.py implementation of def copy_attr(a, b, include=(), exclude=())
    Args:
        model: to get model attributes from
        include: attributes to get
        exclude: attributes to exclude

    Returns:
        attribute names and values
    """
    for k, v in model.__dict__.items():
        if (include and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            yield k, v
