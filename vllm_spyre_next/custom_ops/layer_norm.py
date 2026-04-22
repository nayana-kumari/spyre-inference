# SPDX-License-Identifier: Apache-2.0

import torch
from functools import lru_cache

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import register_layer, get_layer, _fake_impl

logger = init_logger(__name__)

_SPYRE_MIN_BATCH_SIZE = 64


class SpyreLayerNorm:
    """
    Spyre LayerNorm implementation (Spyre backend only).
    CPU fallback removed as requested.
    """

    def __init__(
        self,
        dim: int = 0,
        eps: float = 1e-5,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ):
        self.dim = dim
        self.eps = eps
        self.weight = weight
        self.bias = bias

        self._layer_name = register_layer(self, "spyre_layernorm")

    @staticmethod
    def forward_spyre(
        x: torch.Tensor,
        eps: float,
        hidden_size: int,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
    ):
        """
        Pure LayerNorm implementation (Spyre / fallback math path).
        """
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        x_norm = (x - mean) * torch.rsqrt(variance + eps)

        if weight is not None:
            x_norm = x_norm * weight
        if bias is not None:
            x_norm = x_norm + bias

        return x_norm

    def _forward_spyre_impl(
        self,
        x: torch.Tensor,
        eps: float,
        hidden_size: int,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
    ):
        """
        Spyre execution path only.
        Includes batching safety padding for Spyre kernel.
        """

        orig_batch_size = x.shape[0]
        x_dtype = x.dtype
        x_device = x.device

        # batch padding for Spyre constraint
        if x.shape[0] < _SPYRE_MIN_BATCH_SIZE:
            pad = _SPYRE_MIN_BATCH_SIZE - x.shape[0]

            pad_tensor = torch.zeros(
                pad,
                x.shape[1],
                x.shape[2],
                dtype=x.dtype,
                device=x.device,
            )

            x = torch.cat([x, pad_tensor], dim=0)

        # forward compute (Spyre assumed available in runtime)
        out = self.forward_spyre(
            x,
            eps,
            hidden_size,
            weight,
            bias,
        )

        # restore original shape
        out = out[:orig_batch_size]

        return out


def _op_func(
    x: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """
    Torch custom op entry point for vLLM OOT integration.
    MUST return tensor (required by torch schema inference).
    """

    layer = get_layer(layer_name)

    result = layer._forward_spyre_impl(
        x,
        layer.eps,
        layer.dim,
        layer.weight,
        layer.bias,
    )

    output.copy_(result)

    return output


@lru_cache(maxsize=1)
def register():
    """
    Register Spyre LayerNorm custom op.
    """

    SpyreLayerNorm()

    direct_register_custom_op(
        op_name="spyre_layernorm",
        op_func=_op_func,
        mutates_args=["output"],
        fake_impl=_fake_impl,
    )

    logger.info("Registered custom op: SpyreLayerNorm")
