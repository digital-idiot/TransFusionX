from typing import Any
from typing import Dict
from torch.nn import GELU
from torch.nn import Linear
from torch.nn import Module
from torch.nn import Conv2d
from einops import rearrange
from parts import DeepLabV3
from parts import Transceiver
from torch.nn import Sequential


class LNN(Sequential):
    def __init__(
            self, in_channels: int, out_channels: int, bias: bool = True
    ):
        super().__init__(
            Linear(
                in_features=in_channels,
                out_features=out_channels,
                bias=bias
            ),
            GELU()
        )
        self._in_channels = in_channels
        self._out_channes = out_channels

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channes


class TransFuser(Module):
    def __init__(
            self,
            n_classes: int,
            image_channels: int,
            cloud_channels: int,
            query_channels: int,
            image_branch_spec: Dict[str, Any],
            cloud_branch_spec: Dict[str, Any],
    ):
        super(TransFuser, self).__init__()
        image_branch_spec["in_channels"] = image_channels
        self._image_branch = DeepLabV3(**image_branch_spec)

        self._pre_cloud = LNN(
            in_channels=cloud_channels,
            out_channels=cloud_branch_spec["encoder_spec"]["input_channels"]
        )
        self._cloud_branch = Transceiver.from_args(**cloud_branch_spec)

        self._pre_query = LNN(
            in_channels=query_channels,
            out_channels=cloud_branch_spec["decoder_spec"]["output_channels"]
        )

        self._post_cloud = Sequential(
            Conv2d(
                in_channels=self._pre_query.out_channels,
                out_channels=self._image_branch.out_channels,
                kernel_size=2,
                stride=1,
                dilation=1,
                padding="same",
                groups=1,
                bias=True,
                padding_mode="reflect"
            ),
            GELU()
        )
        hidden_channels = max(self._image_branch.out_channels//2, n_classes)
        self._classifier_stem = Sequential(
            Conv2d(
                in_channels=self._image_branch.out_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding="same",
                groups=1,
                bias=True,
                padding_mode="reflect"
            ),
            GELU(),
            Conv2d(
                in_channels=hidden_channels,
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                dilation=1,
                padding=0,
                groups=1,
                bias=True,
                padding_mode="zeros"
            )
        )
        self._classifier_skip = Conv2d(
            in_channels=self._image_branch.out_channels,
            out_channels=n_classes,
            kernel_size=1,
            stride=1,
            dilation=1,
            padding=0,
            groups=1,
            bias=True,
            padding_mode="zeros"
        )
        self._out_channels = n_classes

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, data: Any):
        image = data["image"]
        cloud = data["cloud"]
        cloud_mask = data["cloud_mask"]
        query = data["query"]

        image = self._image_branch(image)["out"]
        cloud = self._pre_cloud(cloud)
        query = self._pre_query(query)
        cloud = self._cloud_branch(x=cloud, mask=cloud_mask, query=query)
        cloud = rearrange(
            tensor=cloud,
            pattern="b (h w) c -> b c h w",
            h=image.shape[-2],
            w=image.shape[-1]
        )
        cloud = self._post_cloud(cloud)
        prediction = ((image * cloud.softmax(1)) + (image.softmax(1) * cloud))
        prediction = self._classifier_stem(
            prediction
        ) + self._classifier_skip(
            prediction
        )
        return prediction
