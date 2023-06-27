from torch import Tensor
from typing import Literal
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Sequential
from torchvision.models import segmentation


class DeepLabV3(Module):
    _pretrained_data = "COCO_WITH_VOC_LABELS_V1"
    _variants = {
        "mobilenet_v3_large": "DeepLabV3_MobileNet_V3_Large_Weights",
        "resnet50": "DeepLabV3_ResNet50_Weights",
        "resnet101": "DeepLabV3_ResNet101_Weights"
    }

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            backbone_variant: Literal[
                "mobilenet_v3_large", "resnet50", "resnet101"
            ],
            load_pretrained: bool = True
    ):
        super().__init__()
        if backbone_variant not in self._variants.keys():
            raise NotImplementedError(
                f"Unknown backbone varinat: {backbone_variant}"
            )
        net = getattr(segmentation, f"deeplabv3_{backbone_variant}")
        weight = getattr(
            getattr(segmentation, self._variants[backbone_variant]),
            self._pretrained_data
        ) if load_pretrained else None
        net = net(weight=weight)
        if in_channels != net.backbone.conv1.in_channels:
            init_layer = net.backbone.conv1
            net.backbone.conv1 = Conv2d(
                in_channels=in_channels,
                out_channels=init_layer.out_channels,
                kernel_size=init_layer.kernel_size,
                stride=init_layer.stride,
                padding=init_layer.padding,
                dilation=init_layer.dilation,
                groups=1,
                bias=init_layer.bias is not None,
                padding_mode=init_layer.padding_mode,
                device=init_layer.weight.device,
                dtype=init_layer.weight.dtype
            )
        classifier = list(net.classifier.children())
        last_layer = classifier[-1]
        if out_channels != last_layer.out_channels:
            classifier[-1] = Conv2d(
                in_channels=last_layer.in_channels,
                out_channels=out_channels,
                kernel_size=last_layer.kernel_size,
                stride=last_layer.stride,
                padding=last_layer.padding,
                dilation=last_layer.dilation,
                groups=1,
                bias=last_layer.bias is not None,
                padding_mode=last_layer.padding_mode,
                device=last_layer.weight.device,
                dtype=last_layer.weight.dtype
            )
            net.classifier = Sequential(*classifier)
        self._net = net
        self._in_channels = in_channels
        self._out_channels = out_channels

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def model(self):
        return self._net

    def forward(self, x: Tensor) -> Tensor:
        return self._net(x)
