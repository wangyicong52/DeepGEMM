import torch
import types
import warnings
from typing import Tuple, Optional, Union
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


_SUPPORTED_ACTIVATIONS = ('swiglu', 'situ')


def _validate_activation(activation: str) -> None:
    assert activation in _SUPPORTED_ACTIVATIONS, \
        f'Activation must be one of {_SUPPORTED_ACTIVATIONS}, got `{activation}`'


def _is_sm90() -> bool:
    return torch.cuda.get_device_capability()[0] == 9


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu',
                 base: Optional['SymmBuffer'] = None):
        # Align token count
        sm90 = _is_sm90()
        alignment = (
            _C.get_token_alignment_for_sm90_mega_moe()
            if sm90 else _C.get_token_alignment_for_mega_moe()
        )
        num_max_tokens_per_rank = align(num_max_tokens_per_rank, alignment)

        # Init
        _validate_activation(activation)
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation

        # Allocate or reuse a symmetric buffer
        if sm90:
            assert num_shared_experts == 0, 'SM90 uses separately computed shared experts'
            assert mma_type in ('fp8xfp4', 'fp8xfp8'), 'SM90 requires FP8 dispatch'
            num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_sm90_mega_moe(
                group.size(), num_experts, num_max_tokens_per_rank, num_topk,
                hidden, intermediate_hidden, True, activation
            )
        else:
            num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
                group.size(), num_experts,
                num_max_tokens_per_rank, num_topk,
                hidden, intermediate_hidden,
                mma_type, activation,
                num_shared_experts
            )
        if base is None:
            allocator = torch if group.size() == 1 else symm_mem
            self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
            self.handle = (
                types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
                if group.size() == 1
                else symm_mem.rendezvous(self.buffer, group=group)
            )
            self.buffer.zero_()
            self.group.barrier()
            torch.cuda.synchronize()
        else:
            assert base.buffer is not None and base.handle is not None and base.group is group, \
                'Cannot reuse an invalid symmetric buffer'
            assert num_bytes <= base.buffer.nbytes, \
                (f'The reused Mega MoE config requires {num_bytes} bytes, '
                 f'but the symmetric buffer only has {base.buffer.nbytes} bytes')
            self.buffer = base.buffer
            self.handle = base.handle

        # Create input buffer views
        if sm90:
            (self.x, self.x_sf, self.topk_idx, self.topk_weights,
             self.l1_acts, self.l1_acts_sf,
             self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)
            self.shared_l1_acts = self.shared_l1_acts_sf = None
            self.shared_l2_acts = self.shared_l2_acts_sf = None
        else:
            (self.x, self.x_sf,
             self.topk_idx, self.topk_weights,
             self.shared_l1_acts, self.shared_l1_acts_sf,
             self.shared_l2_acts, self.shared_l2_acts_sf,
             self.l1_acts, self.l1_acts_sf,
             self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


# TODO: remove this function
def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 num_shared_experts: int = 0,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Backward compat: derive `mma_type` from `use_fp8_dispatch` if provided
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts,
        mma_type=mma_type, activation=activation
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    # Unsqueeze for 2D
    assert t.dim() in (2, 3)
    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)

    # Transpose
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    result = torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))
    return result.squeeze(0) if squeeze_group_dim else result


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    # Unsqueeze for 2D
    assert sf.dtype == torch.int and sf.dim() in (2, 3)
    squeeze_group_dim = sf.dim() == 2
    if squeeze_group_dim:
        sf = sf.unsqueeze(0)

    # Transpose
    num_groups, mn, packed_sf_k = sf.shape
    assert mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    result = torch.empty_like(sf).copy_(result)
    return result.squeeze(0) if squeeze_group_dim else result


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
           Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    _validate_activation(activation)
    if isinstance(l1_weights, tuple):
        # FP8/FP4: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP
        l1_w = _interleave_weights(l1_weights[0])
        l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = _interleave_weights(l1_weights)
        l2_transformed = l2_weights
return l1_transformed, l2_transformed


def transform_weights_for_mega_moe_sm90(l1_weights, l2_weights):
    return (_interleave_weights(l1_weights[0]), l1_weights[1]), l2_weights


def transform_weights_for_mega_moe_sm90_fp4(l1_weights, l2_weights):
    def pack_scales(sf):
        assert sf.dtype == torch.float32 and sf.dim() == 3
        experts, rows, groups = sf.shape
        assert groups % 4 == 0, 'FP4 K/32 scale groups must be divisible by four'
        ue8m0 = (sf.contiguous().view(torch.int32) >> 23).bitwise_and(0xff).to(torch.uint8)
        return ue8m0.contiguous().view(torch.int32).reshape(experts, rows, groups // 4)

    l1, l1_sf = l1_weights
    l2, l2_sf = l2_weights
    assert l1.dtype in (torch.int8, torch.uint8) and l2.dtype in (torch.int8, torch.uint8)
    l1 = _interleave_weights(l1.contiguous().view(torch.int8))
    l1_sf = _interleave_weights(l1_sf)
    return (l1, pack_scales(l1_sf)), (l2.contiguous().view(torch.int8), pack_scales(l2_sf))


def mega_moe_pre_dispatch_sm90(
    x, topk_idx, topk_weights, sym_buffer, num_tokens=None, group_size=128,
    routed_scaling_factor=1.0,
):
    if num_tokens is None:
        num_tokens = x.size(0)
    return _C.mega_moe_pre_dispatch_sm90(
        x, topk_idx, topk_weights, sym_buffer.x, sym_buffer.x_sf,
        sym_buffer.topk_idx, sym_buffer.topk_weights,
        num_tokens, group_size, routed_scaling_factor,
    )


def fp8_mega_moe(
    y, l1_weights, l2_weights, sym_buffer, cumulative_local_expert_recv_stats=None,
    recipe=(1, 1, 128), activation='swiglu', activation_clamp=None, fast_math=True,
):
    return _C.fp8_mega_moe(
        y, l1_weights, l2_weights, cumulative_local_expert_recv_stats,
        sym_buffer.buffer, sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank, sym_buffer.num_experts,
        sym_buffer.num_topk, recipe, activation, activation_clamp, fast_math,
    )


def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True,
                     activation_alpha: float = 1.0,
                     activation_beta: float = 0.0,
                     num_sms: int = 0):
    _validate_activation(activation)
    if _is_sm90():
        assert shared_l1_weights is None and shared_l2_weights is None, \
            'SM90 uses separately computed shared experts'
        assert sym_buffer.mma_type == 'fp8xfp4', 'SM90 FP4 requires FP8 dispatch'
        assert sym_buffer.x_sf.dtype == torch.float32
        assert sym_buffer.x_sf.shape[1] == sym_buffer.hidden // 128
        return _C.fp8_fp4_mega_moe_sm90(
            y, l1_weights, l2_weights, cumulative_local_expert_recv_stats,
            sym_buffer.buffer, sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
            sym_buffer.num_max_tokens_per_rank, sym_buffer.num_experts,
            sym_buffer.num_topk, recipe, activation, activation_clamp,
            activation_alpha, activation_beta, fast_math, num_sms,
        )
    assert num_sms == 0, 'Per-call SM budgets are only supported by the SM90 backend'
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math,
        activation_alpha, activation_beta
    )

def bf16_mega_moe(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  shared_l1_weights: Optional[torch.Tensor] = None,
                  shared_l2_weights: Optional[torch.Tensor] = None,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True,
                  activation_alpha: float = 1.0,
                  activation_beta: float = 0.0):
    assert activation == 'swiglu', \
        f'BF16 Mega MoE only supports `swiglu`, got `{activation}`'
    _C.bf16_mega_moe(
        y,
        l1_weights,
        l2_weights,
        shared_l1_weights,
        shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math,
        activation_alpha, activation_beta
    )
