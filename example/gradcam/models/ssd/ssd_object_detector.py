"""File contains SSD ObjectDetector class."""
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.ssd import SSD
from torchvision.ops import boxes as box_ops

from foxai.explainer.computer_vision.object_detection.base_object_detector import (
    BaseObjectDetector,
)
from foxai.explainer.computer_vision.object_detection.types import PredictionOutput


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
