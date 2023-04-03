from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from src.types import PredictionOutput
from src.yolo import Detect
from torch import nn
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops

DetectionOutput = Tuple[torch.Tensor, torch.Tensor]


class WrapperDetect(nn.Module):
    """Wrapper over Detect class from YOLOv5 that returns predictions and logits.

    Code based on https://github.com/pooya-mohammadi/yolov5-gradcam.
    """

    def __init__(self, model: nn.Module, anchors=()):
        super().__init__()
        self.model = model
        self.register_buffer(
            "anchors", torch.tensor(anchors).float().view(self.model.nl, -1, 2)
        )  # shape(nl,na,2)

    @property
    def f(self):
        return self.model.f

    @property
    def m(self):
        return self.model.m

    @property
    def na(self):
        return self.model.na

    @property
    def stride(self):
        return self.model.stride

    @property
    def i(self):
        return self.model.i

    def forward(self, x: torch.Tensor) -> DetectionOutput:
        z: List[torch.Tensor] = []  # inference output
        logits_: List[torch.Tensor] = []
        for i in range(self.model.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.model.na, self.model.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.model.training:  # inference
                if self.model.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.model.grid[i], self.model.anchor_grid[i] = self._make_grid(
                        nx, ny, i
                    )
                logits = x[i][..., 5:]
                y = x[i].sigmoid()
                if self.model.inplace:
                    y[..., 0:2] = (
                        y[..., 0:2] * 2.0 - 0.5 + self.model.grid[i]
                    ) * self.model.stride[
                        i
                    ]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.model.anchor_grid[
                        i
                    ]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (
                        y[..., 0:2] * 2.0 - 0.5 + self.model.grid[i]
                    ) * self.model.stride[
                        i
                    ]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.model.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.model.no))
                logits_.append(logits.view(bs, -1, self.model.no - 5))
        return torch.cat(z, 1), torch.cat(logits_, 1)

    def _make_grid(
        self, nx: int = 20, ny: int = 20, i: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.model.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.model.na, ny, nx, 2)).float()
        anchor_grid = (
            (self.model.anchors[i].clone() * self.model.stride[i])
            .view((1, self.model.na, 1, 1, 2))
            .expand((1, self.model.na, ny, nx, 2))
            .float()
        )
        return grid, anchor_grid


class WrapperYOLOv5ObjectDetectionModel(nn.Module):
    """Wrapper for ObjectDetectionModel of YOLOv5 network.

    Code based on https://github.com/pooya-mohammadi/yolov5-gradcam.
    """

    def __init__(self, model: nn.Module, device: torch.DeviceObjType):
        super().__init__()
        self.save = model.save
        self.yaml = model.yaml
        self.names = model.names
        self.inplace = self.yaml
        self.stride = model.stride
        self.ch = self.yaml.get("ch", None)
        self.device = device

        m = model.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, self.ch, s, s))]
            )  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride

        # replace Detect class with DetectWrapper class in YOLOv5 network
        # to return predictions with logits
        custom_detect = WrapperDetect(m)
        model.model[-1] = custom_detect
        self.model = model
        self.model.eval().to(device)

    def forward(
        self,
        x: torch.Tensor,
    ) -> DetectionOutput:
        return self._forward_once(x)  # single-scale inference, train

    def _forward_once(
        self,
        x: torch.Tensor,
    ) -> DetectionOutput:
        y: List[torch.Tensor] = []  # outputs
        for m in self.model.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x


class WrapperSSD(nn.Module):
    """Wrapper for torchvision's SSD model.

    Code based on https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py.
    """

    def __init__(
        self,
        model: nn.Module,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.model = model
        self.class_names = class_names

    def forward(
        self,
        image: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
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
        images, targets = self.model.transform(images, targets)

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
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.model.head(features)

        # create the set of anchors
        anchors = self.model.anchor_generator(images, features)

        detections: List[Dict[str, torch.Tensor]] = []
        detections, logits = self.postprocess_detections(
            head_outputs=head_outputs,
            image_anchors=anchors,
            image_shapes=images.image_sizes,
        )
        detections = self.model.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
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
                class_number=[class_no.tolist()],
                class_name=class_name,
                confidence=[confidence.tolist()],
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
    ) -> List[Dict[str, torch.Tensor]]:
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

            image_boxes = []
            image_scores = []
            image_labels = []
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

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(
                boxes=image_boxes,
                scores=image_scores,
                idxs=image_labels,
                iou_threshold=self.model.nms_thresh,
            )
            keep = keep[: self.model.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )
        # add batch dimension for further processing
        keep_logits = logits[0][keep][None, :]
        return detections, keep_logits
