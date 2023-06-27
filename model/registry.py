import cv2
from typing import Any
from typing import Dict
from torch.nn import Module
from typing import Optional
from .net import TransFuser
from types import SimpleNamespace
from torch.nn import CrossEntropyLoss
from .loss import OhemCrossEntropyLoss
from .loss import RecallCrossEntropyLoss
from segmentation_models_pytorch import DeepLabV3
from segmentation_models_pytorch import DeepLabV3Plus
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.losses import JaccardLoss


class WrapedDeepLabV3(Module):
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_channels: int = 256,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 8,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()
        self._net = DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            upsampling=upsampling,
            aux_params=aux_params
        )
        self._in_channels = in_channels
        self._out_channels = classes

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, data: Dict[str, Any]):
        x = data["image"]
        return self._net(x)


class WrapedDeepLabV3Plus(Module):
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            encoder_output_stride: int = 16,
            decoder_channels: int = 256,
            decoder_atrous_rates: tuple = (12, 24, 36),
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 4,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()
        self._net = DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            encoder_output_stride=encoder_output_stride,
            decoder_channels=decoder_channels,
            decoder_atrous_rates=decoder_atrous_rates,
            activation=activation,
            upsampling=upsampling,
            aux_params=aux_params
        )
        self._in_channels = in_channels
        self._out_channels = classes

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, data: Dict[str, Any]):
        x = data["image"]
        return self._net(x)


MODEL_REGISTRY = SimpleNamespace(
    DeepLabV3=DeepLabV3,
    TransFuser=TransFuser,
    WrapedDeepLabV3=WrapedDeepLabV3,
    WrapedDeepLabV3Plus=WrapedDeepLabV3Plus
)

LOSS_REGISTRY = SimpleNamespace(
    DiceLoss=DiceLoss,
    JaccardLoss=JaccardLoss,
    CrossEntropyLoss=CrossEntropyLoss,
    OhemCrossEntropyLoss=OhemCrossEntropyLoss,
    RecallCrossEntropyLoss=RecallCrossEntropyLoss
)

# noinspection PyUnresolvedReferences
FLAG_ENUMS = SimpleNamespace(
    BORDER_CONSTANT=cv2.BORDER_CONSTANT,
    BORDER_REPLICATE=cv2.BORDER_REPLICATE,
    BORDER_REFLECT=cv2.BORDER_REFLECT,
    BORDER_WRAP=cv2.BORDER_WRAP,
    BORDER_REFLECT_101=cv2.BORDER_REFLECT_101,
    INTER_LINEAR=cv2.INTER_LINEAR,
    INTER_CUBIC=cv2.INTER_CUBIC,
    INTER_AREA=cv2.INTER_AREA,
    INTER_LANCZOS4=cv2.INTER_LANCZOS4
)
