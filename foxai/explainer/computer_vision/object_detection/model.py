# mypy: ignore-errors

from typing import List, Tuple

import torch
from torch import nn

from foxai.explainer.computer_vision.object_detection.yolo import Detect

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

    def __init__(self, model: nn.Module, device: torch.device):
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
        custom_detect = WrapperDetect(model=m)
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
