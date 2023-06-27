import torch
from typing import Sequence
from einops import rearrange
from data_io.containers import PointCloud


class PositionEncoderBase(object):
    def __init__(
            self,
            embed_dim: int,
            sampling_frequency: float,
            mode: str = 'freq',
            concat_loc: str = None,
    ):
        """

        :param embed_dim: Size of the embedding dimension
        :param sampling_frequency: Sampling frequency
        :param concat_loc: location of original position to concatenate
            'pre', 'post' or None (no concatenation)
        """
        # super(PositionEncoderBase, self).__init__()
        assert isinstance(embed_dim, int) and (embed_dim > 0), (
            f"Illegal value for 'embed_dim': {embed_dim}\n" +
            "Expected a positive integer"
        )
        assert isinstance(
            sampling_frequency, (int, float)
        ) and (sampling_frequency > 0), (
            f"Got invalid sampling_frequency: {sampling_frequency}\n"
            "'sampling_frequency' must be positive"
        )
        assert mode in {'freq', 'sin_cos'}, (
            f"Unknown 'mode': {mode}\n" +
            "Expected one of {'freq', 'sin_cos'}"
        )
        if mode == 'freq':
            n_channels = embed_dim
            self._proj = lambda x: x
        elif mode == 'sin_cos':
            n_channels = 2 * embed_dim
            self._proj = lambda x: rearrange(
                tensor=torch.stack(
                    tensors=[x.sin(), x.cos()], dim=-1
                ),
                pattern="... i j -> ... (i j)"
            )
        else:
            raise ValueError(
                f"Unknown 'mode': {mode}\n" +
                "Expected one of {'freq', 'sin_cos'}"
            )
        if concat_loc is None:
            self._cat = lambda x, emb_x: emb_x
        elif concat_loc == 'pre':
            n_channels += 1
            self._cat = lambda x, emb_x: torch.cat(
                tensors=[x.unsqueeze(-1), emb_x], dim=-1
            )
        elif concat_loc == 'post':
            n_channels += 1
            self._cat = lambda x, emb_x: torch.cat(
                tensors=[emb_x, x.unsqueeze(-1)], dim=-1
            )
        else:
            raise ValueError(
                f"Unknown 'concat_loc': {concat_loc}\n" +
                "Expected one of {'pre', 'post', None}"
            )

        self._n_channels = n_channels
        self._concat_loc = concat_loc
        self._sampling_frequency = sampling_frequency
        self._mode = mode
        nyquist_frequency = 0.5 * sampling_frequency
        self._coefficients = torch.pi * torch.linspace(
            start=1,
            end=nyquist_frequency,
            steps=embed_dim
        )

    @property
    def n_channels(self):
        return self._n_channels

    def _frequencies(self, x: torch.Tensor):
        return torch.einsum("i j, k -> ijk", x, self._coefficients.to(x))

    def encode(self, x: torch.Tensor):
        """
        Calculates frequency embeddings

        :param x: Tensor of shape B × N with value range [-1, +1]
        :return: Tuple of a single Tensor of shape B × N × C
        """

        return self._cat(x=x, emb_x=self._proj(x=self._frequencies(x=x)))


class PositionEncoder(object):
    @classmethod
    def construct(
            cls,
            embed_dims: Sequence[int],
            sampling_frequencies: Sequence[float],
            mode: str = 'freq',
            concat_loc: str = None,
    ):
        return cls(
            encoders=[
                PositionEncoderBase(
                    embed_dim=embed_dim,
                    sampling_frequency=sampling_frequency,
                    mode=mode,
                    concat_loc=concat_loc
                )
                for embed_dim, sampling_frequency in zip(
                    embed_dims, sampling_frequencies
                )
            ]
        )

    def __init__(
            self,
            encoders: Sequence[PositionEncoderBase]
    ):
        """

        :param encoders:
        """
        n_channels = 0
        for i, encoder in enumerate(encoders):
            if not isinstance(encoder, PositionEncoderBase):
                raise ValueError(
                    f"'encoders[{i}]' is a {type(encoder)} object!\n" +
                    "Expected a PositionEncoderBase object instead"
                )
            n_channels += encoder.n_channels
        # super(PositionEncoder, self).__init__()
        self._encoders = encoders
        self._n_dim = len(self._encoders)
        self._n_channels = n_channels

    @property
    def n_channels(self):
        return self._n_channels

    @property
    def n_dim(self):
        return self._n_dim

    def encode(self, x: torch.Tensor):
        """
        Calculates frequency embeddings

        :param x: Tensor of shape B × N × D with value range [-1, +1]
        :return: Tuple of a single Tensor of shape B × N × C
        """

        return torch.cat(
            tensors=[
                encoder.encode(t)
                for encoder, t in zip(self._encoders, x.unbind(dim=-1))
            ],
            dim=-1
        )

    def encode_cloud(self, cloud: PointCloud):
        assert self.n_dim == cloud.n_dim, (
                "Dimension mismatch.\n" +
                f"Encoder expected {self.n_dim}D point cloud " +
                f"but {cloud.n_dim}D point cloud"
        )
        coordinates = self.encode(x=cloud.coordinates.unsqueeze(0)).squeeze(0)
        return cloud.update_coordinates(coordinates=coordinates)
