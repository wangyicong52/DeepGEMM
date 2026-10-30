#pragma once

#include "sm90_mega.hpp"
#include "../jit_kernels/impls/sm90_fp8_fp4_mega_moe_rs.hpp"

namespace deep_gemm::mega {

static void fp8_fp4_mega_moe_sm90_rs(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const std::optional<float>& activation_alpha_opt,
    const std::optional<float>& activation_linear_beta_opt,
    const bool& fast_math,
    const int& num_sms_override = 0
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    const auto arch_major = jit->device.get_arch_major();
    DG_HOST_ASSERT(arch_major == 9);

    // Combine writes contiguous BF16 rows directly through y.data_ptr().
    DG_HOST_ASSERT(y.is_cuda() and y.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(y.dim() == 2 and y.is_contiguous());
    DG_HOST_ASSERT(y.size(0) <= num_max_tokens_per_rank);
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu" or activation == "situ");
    const bool use_situ = activation == "situ";
    DG_HOST_ASSERT(not use_situ or not activation_clamp_opt.has_value());

    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);
    const auto activation_alpha = activation_alpha_opt.value_or(4.0f);
    const auto activation_linear_beta = activation_linear_beta_opt.value_or(25.0f);
    DG_HOST_ASSERT(not use_situ or
                   (activation_alpha > 0.0f and activation_linear_beta > 0.0f));

    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        check_grouped_ab_sm90_fp4_mega_moe(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        check_grouped_ab_sm90_fp4_mega_moe(l2_weights);
    DG_HOST_ASSERT(y.size(1) == hidden);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());
    // The FP32 activation-scale rows must be 16-byte aligned.
    DG_HOST_ASSERT(hidden % 512 == 0 and intermediate_hidden % 256 == 0);
    DG_HOST_ASSERT(intermediate_hidden / 64 <= 64);

    check_sm90_fp4_sfb_layout(l1_weights_sf, intermediate_hidden * 2, hidden,
                              num_experts_per_rank);
    check_sm90_fp4_sfb_layout(l2_weights_sf, hidden, intermediate_hidden,
                              num_experts_per_rank);

    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_sm90_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        true, activation);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);
    (void)x;
    (void)x_sf;
    (void)topk_idx;
    (void)topk_weights;

    DG_HOST_ASSERT(deep_jit::get_env<int>("DG_USE_FP4_ACTS") == 0);
    DG_HOST_ASSERT(deep_jit::get_env<int>("DG_USE_FP8_COMBINE") == 0);

    if (num_sms_override) {
        DG_HOST_ASSERT(num_sms_override > 1);
        DG_HOST_ASSERT(num_sms_override <= runtime->get_num_sms());
        DG_HOST_ASSERT(num_sms_override % 2 == 0);
    }

    sm90_fp8_fp4_mega_moe_rs(y,
                          l1_acts, l1_acts_sf,
                          l2_acts, l2_acts_sf,
                          l1_weights, l2_weights,
                          l1_weights_sf, l2_weights_sf,
                          cumulative_local_expert_recv_stats,
                          sym_buffer_ptrs,
                          rank_idx, num_max_tokens_per_rank,
                          num_experts_per_rank,
                          num_tokens, num_topk,
                          hidden, intermediate_hidden,
                          activation,
                          activation_alpha,
                          activation_linear_beta,
                          activation_clamp, fast_math, num_sms_override);

    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

} // namespace deep_gemm::mega
