import torch
import torchvision
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, List
from captum.attr import IntegratedGradients, NoiseTunnel
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from autoxai.context_manager import AutoXaiExplainer, ExplainerWithParams, Explainers

from example.yolov5_exmaple.yolo_utils import (
    get_variables,
    xywh2xyxy,
    make_divisible,
    letterbox,
    non_max_suppression,
    scale_boxes,
)
from autoxai.explainer.base_explainer import CVExplainer

CONF = 0.25
IOU = 0.45

class XaiYoloWrapper(torch.nn.Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.model = model
        self.training = model.training
    
        params = dict(get_variables(model=model,include=("names")))
        self.number_of_classes:int = len(params['names'])

    def xai_non_max_suppression(
        self,
        prediction,
        conf_thres:float=0.25,
        iou_thres:float=0.45,
        agnostic=False,
        multi_label=False,
        max_det=300,
        nm=0,  # number of masks
    ):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        device = prediction.device
        mps = 'mps' in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        
        mi = 5 + nc  # mask start index
        m_out = torch.zeros((bs, self.number_of_classes), device=prediction.device)
        for xi, _ in enumerate([jj for jj in prediction]):
            x = prediction[xi]

            # Compute conf
            x_high_conf = x.detach()[xc[xi]]  # confidence
            if not x_high_conf.shape[0]:
                x = x[:,5:]
                x *= 0
                m_out[xi] = x.sum(dim=0)
                continue
            # Compute conf
            x_high_conf[:, 5:] *= x_high_conf[:, 4:5]  # conf = obj_conf * cls_conf
            box = xywh2xyxy(x_high_conf[:, :4])
            mask = x_high_conf[:, mi:]
            conf, j = x_high_conf[:, 5:mi].max(1, keepdim=True)
            x_high_conf = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
            n = x_high_conf.shape[0]  # number of boxes
            if not n:  # no boxes
                x = x[:,5:]
                x *= 0
                m_out[xi] = x.sum(dim=0)
                continue
            x_indexs = x_high_conf[:, 4].argsort(descending=True)[:max_nms]
            x_high_conf = x_high_conf[x_indexs]
            c = x_high_conf[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x_high_conf[:, :4] + c, x_high_conf[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections
            x_indexs = x_indexs[i]
            
            pick_indices = xc[xi].nonzero()[x_indexs]
            mask = torch.zeros(size = (xc[xi].shape[0],),device=device)
            mask[pick_indices] = 1

            class_confidence = x[:, 5:].clone()
            if class_confidence.requires_grad:
                class_confidence.retain_grad()
            object_confidence = x[:, 4:5].clone()
            if object_confidence.requires_grad:
                object_confidence.retain_grad()
            x[:, 5:] = class_confidence * object_confidence
            x = x[:,5:]
            mask = torch.zeros_like(x)
            mask[pick_indices] = 1
            x = x * mask
            m_out[xi] = x.sum(dim=0)

        return m_out
    
    def forward(self,x):
        x = self.model(x)
        x = self.xai_non_max_suppression(x,
            conf_thres=CONF,
            iou_thres=IOU,
            agnostic=False,
            multi_label=False,
            max_det=1000)  # NMS

        return x

def pre_process(image:np.ndarray, sample_model_parameter:torch.Tensor, stride:int) -> Tuple[torch.Tensor, List[int], List[int]]:
    size:Tuple[int,int]=(640,640)
    if image.shape[0] < 5:  # image in CHW
        image = image.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
    image = image[..., :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
    shape0 = image.shape[:2]  # HWC
    g = max(size) / max(shape0)  # gain
    shape1 = [int(y * g) for y in shape0]
    np_image = image if image.data.contiguous else np.ascontiguousarray(image)  # update
    shape1 =[make_divisible(x, stride) for x in np.array(shape1)] # inf shape
    x = letterbox(np_image, shape1, auto=False)[0]  # pad
    x = np.ascontiguousarray(np.expand_dims(np.array(x),axis=0).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
    x = torch.from_numpy(x).to(sample_model_parameter.device).type_as(sample_model_parameter) / 255  # uint8 to fp16/32

    return x, shape0, shape1

if __name__ == "__main__":
    TARGET = 0
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    image = Image.open("example/images/zidane.jpg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    params = dict(get_variables(model=model,include=("names","stride")))
    
    input_image, shape0, shape1 = pre_process(image=np.asarray(image),sample_model_parameter = next(model.parameters()),stride=params['stride'])

    yolo_model = XaiYoloWrapper(model=model.model).to(device=device)
    input_image = input_image.to(device)

    # Inference
    
    '''integrated_gradients = IntegratedGradients(forward_func=yolo_model)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    attributions = noise_tunnel.attribute(
        input_image, nt_samples=1, nt_type="smoothgrad_sq",stdevs=0.1, target=0
    )
    '''

    with AutoXaiExplainer(
        model=yolo_model,
        explainers=[ExplainerWithParams(
            explainer_name=Explainers.CV_NOISE_TUNNEL_EXPLAINER,
            nt_samples=1, 
            nt_type="smoothgrad_sq",
            stdevs=0.1, 
            target=0,
        )],
    ) as xai_model:
        _, attributions = xai_model(input_image)
    
    attributions = attributions["CV_NOISE_TUNNEL_EXPLAINER"]


    # standard inference
    y = model.model(input_image)
    y = non_max_suppression(y,
        conf_thres=CONF,
        iou_thres=IOU,
        classes=None,
        agnostic=False,
        multi_label=False,
        max_det=1000)  # NMS
    #scale_boxes(shape1, y[0][:, :4], shape0)
    y = y[0].detach().cpu()
    bboxes = y[:,:4]
    normalized_image:torch.Tensor = (
        (input_image - torch.min(input_image)) / (torch.max(input_image) -torch.min(input_image))*255
    ).type(torch.uint8).squeeze()
    labels:List[str] = [params['names'][label.item()] for label in y[:,-1]]
    pred_image = torchvision.utils.draw_bounding_boxes(
        image=normalized_image, 
        boxes=bboxes, 
        labels=labels,
    )
    figure = CVExplainer.visualize(attributions=attributions,transformed_img=pred_image)
    canvas = FigureCanvas(figure)
    canvas.draw()
    buf = canvas.buffer_rgba()
    image = np.asarray(buf)
    cv2.putText(
        img=image,
        text=params['names'][TARGET],
        org = (image.shape[0]//2, image.shape[1]//8),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=image.shape[0]/500,
        color=(0,0,0),
    )
    cv2.imwrite("./example/yolov5_exmaple/xai.png",image)