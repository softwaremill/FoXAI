"""Example of running XAI on YOLOv5.

Below two, same options. One directly using captum, and second using AutoXAI

# XAI inference:
# Option 1
integrated_gradients = IntegratedGradients(forward_func=yolo_model)
noise_tunnel = NoiseTunnel(integrated_gradients)
attributions = noise_tunnel.attribute(
    input_image, nt_samples=1, nt_type="smoothgrad_sq",stdevs=0.1, target=0
)

# Option 2
with AutoXaiExplainer(
    model=yolo_model,
    explainers=[
        ExplainerWithParams(
            explainer_name=Explainers.CV_NOISE_TUNNEL_EXPLAINER,
            nt_samples=1,
            nt_type="smoothgrad_sq",
            stdevs=0.1,
            target=0,
        )
    ],
) as xai_model:
    _, attributions = xai_model(input_image)

in both cases yolo_model is a wrapper on the original yolo model, of type XaiYoloWrapper
"""

from typing import Final, List, Tuple, TypeVar

import cv2
import numpy as np
import torch
import torchvision
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

from autoxai.context_manager import AutoXaiExplainer, Explainers, ExplainerWithParams
from autoxai.explainer.base_explainer import CVExplainer
from example.yolov5_exmaple.yolo_utils import (  # scale_boxes,
    get_variables,
    letterbox,
    make_divisible,
    non_max_suppression,
    xywh2xyxy,
)

TARGET: Final[int] = 0
""" The target class to be explained with XAI.

For yolo it takes all preditctions belonging to the given class.
It means that if 2 persons were detected, the XAI will be computed
for both of them. Instance specific XAI requires code modification.
"""

YoloModelT = TypeVar("YoloModelT", bound=torch.nn.Module)
"""Any subclass of pythorch module."""


class XaiYoloWrapper(torch.nn.Module):
    """The Xai wrapper for the yolo model.

    Most explainers except the model output to consist
    only classes. However many models have custom outputs.
    In case of YOLO_v5 the output is [N,6] tensor, where:
        - N: is the number of predicted objects
        - [x,y,w,h,confidence,class]

    In order to make the output usable by regular
    xai expaliners, we need to convert the output
    to the following shape:
        - [cls1, cls2, ....., clsM], where M is
        the total number of classes.

    We loose the information about number of predictions
    and thier locations. In order to get those data
    we need to run the regular inference separately.
    """

    def __init__(
        self, model: YoloModelT, conf: float = 0.25, iou: float = 0.45
    ) -> None:
        """
        Args:
            model: the yolo model to be used.
            conf: confidence threshold for predicted objects
            iou: iou threshold for preddicted bboxes for nms algorithm
        """
        super().__init__()
        self.model = model
        self.training = model.training

        params = dict(get_variables(model=model, include=("names")))
        self.number_of_classes: int = len(params["names"])
        self.conf: float = conf
        self.iou: float = iou

    def xai_non_max_suppression(
        self,
        prediction: torch.Tensor,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        agnostic: bool = False,
        max_det: int = 300,
    ) -> torch.Tensor:
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

        Args:
            prediction: the model prediction
            conf_thres: confidence threshold, for counting the detection as valid
            iou_thres: intersection over union threshold for non max suppresion algorithm
            agnostic: if True, non max suppression algorithm is run on raw bboxes. However it may
                happen that different classes have bboxes in similar place. NMX would discard one of
                those bboxes and keep only the one with higher confidence. If we want to keep bboxes
                that are in similar place, but have different class label, we should set agnostic to False.
            max_det: maximum number of detections

        Returns:
            batch of detections of shape (B,80), where 80 are classes confidence

        Example output:
                cls0_conf   cls1_conf   ....    cls79_conf
            0  0.005       0.00002     ...     0.87002
            1  0.535       0.20002     ...     0.08002
            .  ...         ...         ...     ...
            B  0.00008     0.10302     ...     0.0289
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

        # calculate batch size
        batch_size = prediction.shape[0]  # batch size

        # calculate True/False list for all predictions that meet confidence threshold
        # criteria for all samples in batch
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

        # set start mask index as first element after the last class index
        mi = 5 + self.number_of_classes  # mask start index

        # create empty tensor representing result - probability of each class
        out_predictions = torch.zeros(
            (batch_size, self.number_of_classes), device=prediction.device
        )

        # pylint: disable = unnecessary-comprehension
        for xi, _ in enumerate([jj for jj in prediction]):

            # get sample prediction
            x = prediction[xi]

            # get all anchors that meet confidence threshold criteria from a single sample from a batch
            x_high_conf = x.detach()[xc[xi]]

            # if none of the anchors meet threshold criteria
            if not x_high_conf.shape[0]:
                # get class outputs
                x = x[:, 5:]
                # set all class outputs to zero (no gradient)
                # does outputs does not contribute to the final prediction
                x *= 0
                # sum same classes together to get the shape [number_of_objectes,num_of_classes] -> [num_of_classes]
                out_predictions[xi] = x.sum(dim=0)
                continue

            # multiply class probability by confidence score
            x_high_conf[:, 5:] *= x_high_conf[:, 4:5]  # conf = obj_conf * cls_conf

            # get bounding box dimensions
            box = xywh2xyxy(x_high_conf[:, :4])

            # get confidence and argmax of classes for all anchors
            conf, j = x_high_conf[:, 5:mi].max(1, keepdim=True)

            # overwrite anchors that meet confidence criteria with
            # bounding box dimensions, confidence, probability and
            x_high_conf = torch.cat((box, conf, j.float()), dim=1)[
                conf.view(-1) > conf_thres
            ]

            # get number of anchors
            number_of_anchors = x_high_conf.shape[0]  # number of boxes

            # if no anchors are present
            if not number_of_anchors:  # no boxes
                # get only class confidence
                x = x[:, 5:]
                # set all class outputs to zero (no gradient)
                # does outputs does not contribute to the final prediction
                x *= 0
                # sum same classes together to get the shape [number_of_objectes,num_of_classes] -> [num_of_classes]
                out_predictions[xi] = x.sum(dim=0)
                continue

            # get indices of predictions by confidence score in descending order
            x_indexs = x_high_conf[:, 4].argsort(descending=True)[:max_nms]

            # get predictions by confidence in descending order
            x_high_conf = x_high_conf[x_indexs]

            # get class
            c = x_high_conf[:, 5:6] * (0 if agnostic else max_wh)  # classes
            # get boxes (with offset by class) and scores
            boxes, scores = (
                x_high_conf[:, :4] + c,
                x_high_conf[:, 4],
            )

            # get bounding boxes indices to keep from NMS
            selected_indices = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

            # limit detections to specified number
            selected_indices = selected_indices[:max_det]  # limit detections

            # get indexes of x_high_conf tensor, with high confidence and non-overlaping bboxes
            x_indexs = x_indexs[selected_indices]

            # grab indexes x  tensor, with high confidence and non-overlaping bboxes
            pick_indices = xc[xi].nonzero()[x_indexs]

            # get classes confidence
            # in place opeartions not supported for gradient computation
            # we need to clone the tensor and keep track of gradient
            class_confidence = x[:, 5:].clone()
            if class_confidence.requires_grad:
                class_confidence.retain_grad()
            # get object confidence
            object_confidence = x[:, 4:5].clone()
            if object_confidence.requires_grad:
                object_confidence.retain_grad()

            # multiply class confidence by object confidence
            x[:, 5:] = class_confidence * object_confidence

            # retain only classes predictions
            x = x[:, 5:]

            # create mask of anchors and mark selected
            mask = torch.zeros_like(x)
            mask[pick_indices] = 1

            # erase non-selected anchors
            x = x * mask

            # sum probabilities of classes over all anchors
            # instance confidence to semantic confidence
            out_predictions[xi] = x.sum(dim=0)

        return out_predictions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.xai_non_max_suppression(
            x,
            conf_thres=self.conf,
            iou_thres=self.iou,
            agnostic=False,
            max_det=1000,
        )  # NMS

        return x


def pre_process(
    image: np.ndarray, sample_model_parameter: torch.Tensor, stride: int
) -> Tuple[torch.Tensor, List[int], List[int]]:
    """Transform the input image to the yolo network.

    Args:
        image: the input image to the network
        sample_model_parameter: the model parameter is used to read
            the model type (fp32/fp16) and target device
        stride: the yolo network stride

    Retuns:
        tensor image ready to be feed into the network
    """
    size: Tuple[int, int] = (640, 640)
    if image.shape[0] < 5:  # image in CHW
        image = image.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
    image = (
        image[..., :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    )  # enforce 3ch input
    shape0 = image.shape[:2]  # HWC
    g = max(size) / max(shape0)  # gain
    shape1 = [int(y * g) for y in shape0]
    np_image = image if image.data.contiguous else np.ascontiguousarray(image)  # update
    shape1 = [make_divisible(x, stride) for x in np.array(shape1)]  # inf shape
    x = letterbox(np_image, shape1, auto=False)[0]  # pad
    x = np.ascontiguousarray(
        np.expand_dims(np.array(x), axis=0).transpose((0, 3, 1, 2))
    )  # stack and BHWC to BCHW
    x = (
        torch.from_numpy(x)
        .to(sample_model_parameter.device)
        .type_as(sample_model_parameter)
        / 255
    )  # uint8 to fp16/32

    return x, shape0, shape1


def main():
    """Run YOLO_v5 XAI and save results."""

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    image = Image.open("example/images/zidane.jpg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    params = dict(get_variables(model=model, include=("names", "stride")))

    input_image, _, _ = pre_process(
        image=np.asarray(image),
        sample_model_parameter=next(model.parameters()),
        stride=params["stride"],
    )

    yolo_model = XaiYoloWrapper(model=model.model, conf=model.conf, iou=model.iou).to(
        device=device
    )
    input_image = input_image.to(device)

    with AutoXaiExplainer(
        model=yolo_model,
        explainers=[
            ExplainerWithParams(
                explainer_name=Explainers.CV_NOISE_TUNNEL_EXPLAINER,
                nt_samples=1,
                nt_type="smoothgrad_sq",
                stdevs=0.1,
                target=0,
            )
        ],
    ) as xai_model:
        _, attributions = xai_model(input_image)

    attributions = attributions["CV_NOISE_TUNNEL_EXPLAINER"]

    # standard inference
    y = model.model(input_image)
    y = non_max_suppression(
        y,
        conf_thres=model.conf,
        iou_thres=model.iou,
        classes=None,
        agnostic=False,
        multi_label=False,
        max_det=1000,
    )  # NMS

    # uncomment to scale boxes to the original image size
    # scale_boxes(shape1, y[0][:, :4], shape0)
    y = y[0].detach().cpu()
    bboxes = y[:, :4]
    normalized_image: torch.Tensor = (
        (
            (input_image - torch.min(input_image))
            / (torch.max(input_image) - torch.min(input_image))
            * 255
        )
        .type(torch.uint8)
        .squeeze()
    )
    labels: List[str] = [params["names"][label.item()] for label in y[:, -1]]
    pred_image = torchvision.utils.draw_bounding_boxes(
        image=normalized_image,
        boxes=bboxes,
        labels=labels,
    )
    figure = CVExplainer.visualize(
        attributions=attributions.squeeze(), transformed_img=pred_image
    )
    canvas = FigureCanvas(figure)
    canvas.draw()
    buf = canvas.buffer_rgba()
    image = np.asarray(buf)
    cv2.putText(
        img=image,
        text=params["names"][TARGET],
        org=(image.shape[0] // 2, image.shape[1] // 8),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=image.shape[0] / 500,
        color=(0, 0, 0),
    )
    cv2.imwrite(
        "./example/yolov5_exmaple/xai.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )


if __name__ == "__main__":
    main()
