#pragma once

#include "sm90_mega_moe.hpp"

namespace deep_gemm {

// Opt-in RS configuration; the original FP4 and FP8 policies remain separate.
static MegaMoESM90Config get_mega_moe_config_sm90_fp4_rs(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens,
    const int& num_sms_override = 0) {
    DG_HOST_ASSERT(hidden % 512 == 0 and intermediate_hidden % 256 == 0);
    const int block_m = 64;
    const int block_n = 256;
    const int block_k = 256;
    const int num_stages = 3;
    const int num_dispatch_threads = 64;
    const int num_non_epilogue_threads = 64;
    const int num_epilogue_threads = 512;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int num_experts_per_wave = get_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk, intermediate_hidden,
        block_m, block_n, num_sms_override ? num_sms_override : runtime->get_num_sms(), num_max_pool_tokens,
        num_max_tokens_per_rank, num_ranks);

    constexpr int kSmemAlignment = 1024;
    const int smem_dispatch = align(num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment) +
        align(static_cast<int>(layout::Buffer(layout::Data(hidden), 2, 1).get_num_bytes()), kSmemAlignment);
    const int smem_cd = align(block_m * (block_n / 2) * 5, kSmemAlignment);
    // Preserve the tuned pipeline's shared-memory base alignment.
    const int smem_pipeline_padding = kSmemAlignment;
    const int smem_sfa = align((block_k / kSM90FP4L2ActSFGranK) * block_m * 4, 128);
    const int smem_sfb = align(block_n * 4 * (block_k / 128), 128);
    const int smem_per_stage = block_m * block_k + block_n * (block_k / 2) + smem_sfa + smem_sfb + 16;
    const int smem_barriers_fixed = (2 + 2 * (num_epilogue_threads / 32)) * 8;
    const int smem_size = smem_dispatch + smem_cd + smem_pipeline_padding + smem_barriers_fixed +
        num_stages * smem_per_stage;
    DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);
    return MegaMoESM90Config {
        block_m, block_n, block_k, 1, num_max_pool_tokens, num_padded_sf_pool_tokens,
        128, 64, num_experts_per_wave, num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads
    };
}

} // namespace deep_gemm
