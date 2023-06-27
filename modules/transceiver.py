import torch
import torch.nn as tnn
from typing import Any
from typing import Dict
from einops import repeat
from typing import Optional
from einops import rearrange
from parts.utils import Residual
from parts.utils import Sequential
from parts.utils import init_parameters


class MultiHeadAttention(tnn.Module):
    def __init__(
            self,
            num_heads: int,
            num_q_input_channels: int,
            num_kv_input_channels: int,
            num_qk_channels: Optional[int] = None,
            num_v_channels: Optional[int] = None,
            num_output_channels: Optional[int] = None,
            causal_attention: bool = False,
            dropout: float = 0.0,
            qkv_bias: bool = True,
            out_bias: bool = True
    ):
        """Multi-head attention as specified in
            https://arxiv.org/abs/2107.14795 Appendix E plus support for rotary
            position embeddings (https://arxiv.org/abs/2104.09864) and causal
            attention.

        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_qk_channels: Number of query and key channels. Default is
            number `num_q_input_channels`
        :param num_v_channels: Number of value channels. Default is
            `num_qk_channels`.
        :param num_output_channels: Number of output channels. Default is
            `num_q_input_channels`
        :param causal_attention: Whether to apply a causal attention mask.
            Default is `False`.
        :param dropout: Dropout probability for attention matrix values.
            Default is `0.0`
        :param qkv_bias: Whether to use a bias term for query, key and value
            projections. Default is `True`.
        :param qkv_bias: Whether to use a bias term for output projection.
            Default is `True`."""
        super().__init__()

        if num_qk_channels is None:
            num_qk_channels = num_q_input_channels

        if num_v_channels is None:
            num_v_channels = num_qk_channels

        if num_output_channels is None:
            num_output_channels = num_q_input_channels

        if num_qk_channels % num_heads != 0:
            raise ValueError(
                f"num_qk_channels ({num_qk_channels}) must be divisible by " +
                f"num_heads ({num_heads})"
            )

        if num_v_channels % num_heads != 0:
            raise ValueError(
                f"num_v_channels ({num_v_channels}) must be divisible by " +
                f"num_heads ({num_heads})"
            )

        num_qk_channels_per_head = num_qk_channels // num_heads

        self.dp_scale = num_qk_channels_per_head ** -0.5
        self.num_heads = num_heads
        self.causal_attention = causal_attention

        self.q_proj = tnn.Linear(
            num_q_input_channels, num_qk_channels, bias=qkv_bias
        )
        self.k_proj = tnn.Linear(
            num_kv_input_channels, num_qk_channels, bias=qkv_bias
        )
        self.v_proj = tnn.Linear(
            num_kv_input_channels, num_v_channels, bias=qkv_bias
        )
        self.o_proj = tnn.Linear(
            num_v_channels, num_output_channels, bias=out_bias
        )
        self.dropout = tnn.Dropout(dropout)

    def forward(
            self,
            x_q,
            x_kv,
            pad_mask=None,
            rot_pos_emb_q: Optional = None,
            rot_pos_emb_k: Optional = None
    ):
        """
        :param x_q: Query input of shape (B, N, D) where B is the batch size, N
            the query sequence length and D the number of query input channels
            (= `num_q_input_channels`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch
            size, L the key/value sequence length and C are the number of
            key/value input channels (= `num_kv_input_channels`)
        :param pad_mask: Boolean key padding mask. `True` values indicate
            padding tokens.
        :param rot_pos_emb_q: Applies a rotary position embedding to query i.e.
            if defined, rotates the query.
        :param rot_pos_emb_k: Applies a rotary position embedding to key i.e.
            if defined, rotates the key.
        :return: attention result of shape (B, N, F) where B is the batch size,
            N the query sequence length and F the number of output channels
            (= `num_output_channels`)"""

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        q, k, v = (
            rearrange(
                x, "b n (h c) -> b h n c", h=self.num_heads
            ) for x in [q, k, v]
        )
        q = q * self.dp_scale

        if rot_pos_emb_q is not None:
            q = rot_pos_emb_q.rotate(q)

        if rot_pos_emb_k is not None:
            k = rot_pos_emb_k.rotate(k)

        attn = torch.einsum("b h i c, b h j c -> b h i j", q, k)
        attn_max_neg = -torch.finfo(attn.dtype).max

        if pad_mask is not None:
            pad_mask = rearrange(pad_mask, "b j -> b 1 1 j")
            attn.masked_fill_(pad_mask, attn_max_neg)

        if self.causal_attention:
            i = q.shape[2]
            j = k.shape[2]

            causal_mask = torch.ones(
                (i, j), device=x_q.device, dtype=torch.bool
            ).triu(j - i + 1)
            attn.masked_fill_(causal_mask, attn_max_neg)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        o = torch.einsum("b h i j, b h j c -> b h i c", attn, v)
        o = rearrange(o, "b h n c -> b n (h c)", h=self.num_heads)

        return self.o_proj(o)


class CrossAttention(tnn.Module):
    def __init__(
            self,
            num_heads: int,
            num_q_input_channels: int,
            num_kv_input_channels: int,
            num_qk_channels: Optional[int] = None,
            num_v_channels: Optional[int] = None,
            causal_attention: bool = False,
            dropout: float = 0.0,
            qkv_bias: bool = True,
            out_bias: bool = True
    ):
        """Pre-layer norm cross-attention (see `MultiHeadAttention` for
            attention details)."""
        super().__init__()
        self.q_norm = tnn.LayerNorm(num_q_input_channels)
        self.kv_norm = tnn.LayerNorm(num_kv_input_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

    def forward(
            self,
            x_q,
            x_kv=None,
            x_kv_prefix=None,
            pad_mask=None,
            rot_pos_emb_q=None,
            rot_pos_emb_k=None
    ):
        """Pre-layer norm cross-attention of query input `x_q` to key/value
            input (`x_kv` or `x_kv_prefix`). If `x_kv_prefix` is defined, the
            entire key/value input is assumed to  be a concatenation of
            `x_kv_prefix` and `x_q` along the sequence dimension. In this case,
            the query attends to itself at the end of the key/value sequence
            (use case Perceiver AR). If `x_kv_prefix` is not defined, `x_kv` is
            assumed to be the entire key/value input."""
        x_q = self.q_norm(x_q)

        if x_kv is None:
            x_kv_prefix = self.kv_norm(x_kv_prefix)
            x_kv = torch.cat([x_kv_prefix, x_q], dim=1)
        else:
            x_kv = self.kv_norm(x_kv)

        return self.attention(
            x_q,
            x_kv,
            pad_mask=pad_mask,
            rot_pos_emb_q=rot_pos_emb_q,
            rot_pos_emb_k=rot_pos_emb_k
        )


class SelfAttention(tnn.Module):
    def __init__(
            self,
            num_heads: int,
            num_channels: int,
            num_qk_channels: Optional[int] = None,
            num_v_channels: Optional[int] = None,
            causal_attention: bool = False,
            dropout: float = 0.0,
            qkv_bias: bool = True,
            out_bias: bool = True
    ):
        """Pre-layer norm self-attention (see `MultiHeadAttention` and for
            attention details)."""
        super().__init__()
        self.norm = tnn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_channels,
            num_kv_input_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

    def forward(self, x, pad_mask=None, rot_pos_emb=None):
        """Pre-layer norm self-attention of input `x`."""
        x = self.norm(x)
        return self.attention(
            x,
            x,
            pad_mask=pad_mask,
            rot_pos_emb_q=rot_pos_emb,
            rot_pos_emb_k=rot_pos_emb
        )


class CrossAttentionLayer(Sequential):
    def __init__(
            self,
            num_heads: int,
            num_q_input_channels: int,
            num_kv_input_channels: int,
            num_qk_channels: Optional[int] = None,
            num_v_channels: Optional[int] = None,
            causal_attention: bool = False,
            widening_factor: int = 1,
            dropout: float = 0.0,
            attention_residual: bool = True,
            qkv_bias: bool = True,
            out_bias: bool = True,
            mlp_bias: bool = True
    ):
        cross_attn = CrossAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )
        super().__init__(
            Residual(cross_attn) if attention_residual else cross_attn,
            Residual(
                MLP(num_q_input_channels, widening_factor, bias=mlp_bias)
            ),
        )


class SelfAttentionLayer(Sequential):
    def __init__(
            self,
            num_heads: int,
            num_channels: int,
            num_qk_channels: Optional[int] = None,
            num_v_channels: Optional[int] = None,
            causal_attention: bool = False,
            widening_factor: int = 1,
            dropout: float = 0.0,
            qkv_bias: bool = True,
            out_bias: bool = True,
            mlp_bias: bool = True
    ):
        self_attn = SelfAttention(
            num_heads=num_heads,
            num_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )
        super().__init__(
            Residual(self_attn),
            Residual(MLP(num_channels, widening_factor, bias=mlp_bias)),
        )


class SelfAttentionBlock(Sequential):
    def __init__(
            self,
            num_layers: int,
            num_heads: int,
            num_channels: int,
            num_qk_channels: Optional[int] = None,
            num_v_channels: Optional[int] = None,
            causal_attention: bool = False,
            widening_factor: int = 1,
            dropout: float = 0.0,
            qkv_bias: bool = True,
            out_bias: bool = True,
            mlp_bias: bool = True
    ):
        layers = [
            SelfAttentionLayer(
                num_heads=num_heads,
                num_channels=num_channels,
                num_qk_channels=num_qk_channels,
                num_v_channels=num_v_channels,
                causal_attention=causal_attention,
                widening_factor=widening_factor,
                dropout=dropout,
                qkv_bias=qkv_bias,
                out_bias=out_bias,
                mlp_bias=mlp_bias,
            )
            for _ in range(num_layers)
        ]
        super().__init__(*layers)


class MLP(Sequential):
    def __init__(
            self, num_channels: int, widening_factor: int, bias: bool = True
    ):
        super().__init__(
            tnn.LayerNorm(num_channels),
            tnn.Linear(num_channels, widening_factor * num_channels, bias=bias),
            tnn.GELU(),
            tnn.Linear(widening_factor * num_channels, num_channels, bias=bias),
        )


class TransceiverEncoder(tnn.Module):
    def __init__(
            self,
            *,
            input_channels: int,
            num_latents: int,
            num_latent_channels: int,
            num_cross_attention_heads: int = 4,
            num_cross_attention_qk_channels: Optional[int] = None,
            num_cross_attention_v_channels: Optional[int] = None,
            num_cross_attention_layers: int = 1,
            first_cross_attention_layer_shared: bool = False,
            cross_attention_widening_factor: int = 1,
            num_self_attention_heads: int = 4,
            num_self_attention_qk_channels: Optional[int] = None,
            num_self_attention_v_channels: Optional[int] = None,
            num_self_attention_layers_per_block: int = 6,
            num_self_attention_blocks: int = 1,
            first_self_attention_block_shared: bool = True,
            self_attention_widening_factor: int = 1,
            dropout: float = 0.0,
            init_scale: float = 0.02
    ):
        """Generic Perceiver IO encoder.
        :param input_channels: channel dimension size (C) of input tensor of
            shape (B, M, C) where B is the batch size, M the input sequence
            length
        :param num_latents: Number of latent variables (N).
        :param num_latent_channels: Number of latent channels (D).
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key
            channels for cross-attention (see
            `MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for
            cross-attention (see `MultiHeadAttention.num_v_channels`
            for details).
        :param num_cross_attention_layers: Number of cross-attention layers
            (alternating with self-attention blocks).
        :param first_cross_attention_layer_shared: Whether the first
            cross-attention layer should share its weights with subsequent
            cross-attention layers (if any).
        :param num_self_attention_heads: Number of self-attention heads.
        :param num_self_attention_qk_channels: Number of query and key channels
            for self-attention (see `MultiHeadAttention.num_qk_channels` for
            details).
        :param num_self_attention_v_channels: Number of value channels for
            self-attention (see `MultiHeadAttention.num_v_channels` for
            details).
        :param num_self_attention_layers_per_block: Number of self-attention
            layers per self-attention block.
        :param num_self_attention_blocks: Number of self-attention blocks
            sharing weights between corresponding
            self-attention layers.
        :param first_self_attention_block_shared: Whether the first
            self-attention block should share its weights with subsequent
            self-attention blocks (if any).
        :param dropout: Dropout probability for self- and cross-attention
            layers and residuals.
        :param init_scale: Standard deviation for random normal initialization
            of parameters.

        Note: 'num_latent_channels' must be divisible by both
            'num_cross_attention_heads' and 'num_self_attention_heads'

        Example: PerceiverEncoder(
                    input_channels=3,
                    num_latents=7,
                    num_latent_channels=4,
                    num_cross_attention_heads=2,
                    num_self_attention_heads=2
                )
        """
        super().__init__()

        self._input_channels = input_channels
        self._output_channels = num_latent_channels
        self._num_latents = num_latents

        if num_cross_attention_layers <= 0:
            raise ValueError("num_cross_attention_layers must be > 0")

        if num_self_attention_blocks <= 0:
            raise ValueError("num_self_attention_blocks must be > 0")

        if num_cross_attention_layers > num_self_attention_blocks:
            raise ValueError(
                "num_cross_attention_layers " +
                "must be <= num_self_attention_blocks"
            )

        self._num_cross_attention_layers = num_cross_attention_layers
        self._num_self_attention_blocks = num_self_attention_blocks

        self._first_cross_attention_layer_shared = (
            first_cross_attention_layer_shared
        )
        self._first_self_attention_block_shared = (
            first_self_attention_block_shared
        )

        def cross_attn():
            layer = CrossAttentionLayer(
                num_heads=num_cross_attention_heads,
                num_q_input_channels=num_latent_channels,
                num_kv_input_channels=input_channels,
                num_qk_channels=num_cross_attention_qk_channels,
                num_v_channels=num_cross_attention_v_channels,
                widening_factor=cross_attention_widening_factor,
                dropout=dropout,
            )
            return layer

        def self_attn():
            return SelfAttentionBlock(
                num_layers=num_self_attention_layers_per_block,
                num_heads=num_self_attention_heads,
                num_channels=num_latent_channels,
                num_qk_channels=num_self_attention_qk_channels,
                num_v_channels=num_self_attention_v_channels,
                widening_factor=self_attention_widening_factor,
                dropout=dropout
            )

        self._cross_attn_1 = cross_attn()
        self._self_attn_1 = self_attn()

        if self.extra_cross_attention_layer:
            self._cross_attn_n = cross_attn()

        if self.extra_self_attention_block:
            self._self_attn_n = self_attn()

        # learnable initial latent vectors
        self._latent = tnn.Parameter(
            torch.empty(num_latents, num_latent_channels)
        )
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self._latent.normal_(0.0, init_scale)
            init_parameters(self, init_scale)

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def output_channels(self):
        return self._output_channels

    @property
    def num_latents(self):
        return self._num_latents

    @property
    def extra_cross_attention_layer(self):
        return (
            self._num_cross_attention_layers > 1
        ) and not self._first_cross_attention_layer_shared

    @property
    def extra_self_attention_block(self):
        return (
           self._num_self_attention_blocks > 1
        ) and not self._first_self_attention_block_shared

    def forward(self, x, *, pad_mask=None):
        b, *_ = x.shape

        # repeat initial latent vector along batch dimension
        x_latent = repeat(self._latent, "... -> b ...", b=b)

        x_latent = self._cross_attn_1(x_latent, x, pad_mask=pad_mask)
        # noinspection PyCallingNonCallable
        x_latent = self._self_attn_1(x_latent)

        cross_attn_n = (
            self._cross_attn_n
        ) if self.extra_cross_attention_layer else self._cross_attn_1
        self_attn_n = (
            self._self_attn_n
        ) if self.extra_self_attention_block else self._self_attn_1

        for i in range(1, self._num_self_attention_blocks):
            if i < self._num_cross_attention_layers:
                x_latent = cross_attn_n(x_latent, x, pad_mask=pad_mask)
            # noinspection PyCallingNonCallable
            x_latent = self_attn_n(x_latent)

        return x_latent


class TransceiverDecoder(tnn.Module):
    def __init__(
            self,
            *,
            output_channels: int,
            num_latent_channels: int,
            num_cross_attention_heads: int = 4,
            num_cross_attention_qk_channels: Optional[int] = None,
            num_cross_attention_v_channels: Optional[int] = None,
            cross_attention_widening_factor: int = 1,
            cross_attention_residual: bool = True,
            dropout: float = 0.0,
            init_scale: float = 0.02
    ):
        """Generic Perceiver IO decoder.

        :param output_channels: The number of cross-attention output channels.
            F of (B x N x F) output tensor.
        :param num_latent_channels: Number of latent channels (C_latent) as
            produced by a Perceiver IO encoder.
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key
            channels for cross-attention (see
            `MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for
            cross-attention (see
            `MultiHeadAttention.num_v_channels` for details).
        :param dropout: Dropout probability for cross-attention layers and
            residuals.
        :param init_scale: Standard deviation for random normal initialization
            of parameters.
        """
        super(TransceiverDecoder, self).__init__()

        cross_attn = CrossAttentionLayer(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=output_channels,
            num_kv_input_channels=num_latent_channels,
            num_qk_channels=num_cross_attention_qk_channels,
            num_v_channels=num_cross_attention_v_channels,
            widening_factor=cross_attention_widening_factor,
            attention_residual=cross_attention_residual,
            dropout=dropout,
        )

        self.cross_attn = cross_attn
        self._init_parameters(init_scale)

        self._input_channels = num_latent_channels
        self._output_channels = output_channels

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def output_channels(self):
        return self._output_channels

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            init_parameters(self, init_scale)

    def forward(self, x, output_query):
        # output_query = self.output_adapter.output_query(x)
        # noinspection PyCallingNonCallable
        return self.cross_attn(output_query, x)


class Transceiver(tnn.Module):
    def __init__(self, *, encoder: TransceiverEncoder, decoder: TransceiverDecoder):
        assert (encoder.output_channels == decoder.input_channels), (
            "Encoder and Decoder are not compatible with each other!\n" +
            f"Encoder yields tensor with {encoder.output_channels} but " +
            f"Decoder accepts tensor with {decoder.input_channels} channels!"
        )
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    def forward(self, x: torch.Tensor, mask: torch.Tensor, query: torch.Tensor):
        x = self._encoder(x=x, pad_mask=mask)
        x = self._decoder(x=x, output_query=query)
        return x

    @classmethod
    def from_args(
            cls,
            encoder_spec: Dict[str, Any],
            decoder_spec: Dict[str, Any]
    ):
        encoder = TransceiverEncoder(**encoder_spec)
        decoder = TransceiverDecoder(**decoder_spec)
        return cls(encoder=encoder, decoder=decoder)
