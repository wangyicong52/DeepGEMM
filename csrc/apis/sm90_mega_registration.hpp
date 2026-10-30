#pragma once

#include <torch/library.h>
#include "sm90_mega.hpp"

namespace deep_gemm::torch_registration::sm90 {

static int64_t get_token_alignment_for_sm90_mega_moe() {
    return mega::get_token_alignment_for_sm90_mega_moe();
}

// Serialize geometry instead of returning the old pybind slicing closure.
static std::tuple<int64_t, std::vector<int64_t>> get_symm_buffer_size_for_sm90_mega_moe(
    int64_t num_ranks, int64_t num_experts, int64_t capacity, int64_t num_topk,
    int64_t hidden, int64_t intermediate_hidden, bool use_fp8_dispatch,
    const std::string& activation) {
    DG_HOST_ASSERT(num_ranks > 0 and num_experts > 0 and capacity > 0);
    DG_HOST_ASSERT(num_topk > 0 and num_topk <= num_experts);
    DG_HOST_ASSERT(hidden > 0 and intermediate_hidden > 0);
    const auto [num_bytes, _slice] = mega::get_symm_buffer_size_for_sm90_mega_moe(
        mega::checked_int(num_ranks), mega::checked_int(num_experts),
        mega::checked_int(capacity), mega::checked_int(num_topk),
        mega::checked_int(hidden), mega::checked_int(intermediate_hidden),
        use_fp8_dispatch, activation);
    return {num_bytes, {num_ranks, num_experts, capacity, num_topk, hidden,
                        intermediate_hidden, use_fp8_dispatch, activation == "situ"}};
}

using SM90SymmBufferSlice = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

static SM90SymmBufferSlice slice_symm_buffer(
    const torch::Tensor& buffer, const std::vector<int64_t>& geometry) {
    DG_HOST_ASSERT(geometry.size() == 8);
    DG_HOST_ASSERT(geometry[6] == 1 and (geometry[7] == 0 or geometry[7] == 1));
    DG_HOST_ASSERT(buffer.is_cuda() and buffer.is_contiguous());
    DG_HOST_ASSERT(buffer.dim() == 1 and buffer.scalar_type() == torch::kInt8);
    const auto activation = geometry[7] ? "situ" : "swiglu";
    const auto [num_bytes, _geometry] = get_symm_buffer_size_for_sm90_mega_moe(
        geometry[0], geometry[1], geometry[2], geometry[3], geometry[4],
        geometry[5], true, activation);
    DG_HOST_ASSERT(buffer.nbytes() >= static_cast<size_t>(num_bytes));
    const auto [_num_bytes, slice] = mega::get_symm_buffer_size_for_sm90_mega_moe(
        mega::checked_int(geometry[0]), mega::checked_int(geometry[1]),
        mega::checked_int(geometry[2]), mega::checked_int(geometry[3]),
        mega::checked_int(geometry[4]), mega::checked_int(geometry[5]),
        true, activation);
    return slice(buffer);
}

static void fp8_fp4_mega_moe_sm90(
    const torch::Tensor& y, const torch::Tensor& l1, const torch::Tensor& l1_sf,
    const torch::Tensor& l2, const torch::Tensor& l2_sf,
    const std::optional<torch::Tensor>& stats, const torch::Tensor& buffer,
    const std::vector<int64_t>& pointers, int64_t rank, int64_t capacity,
    int64_t num_experts, int64_t num_topk, const std::vector<int64_t>& recipe,
    const std::string& activation, const std::optional<double>& clamp,
    const std::optional<double>& alpha, const std::optional<double>& beta,
    bool fast_math, int64_t num_sms) {
    DG_HOST_ASSERT(y.is_cuda() and y.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(y.dim() == 2 and y.is_contiguous());
    DG_HOST_ASSERT(y.size(1) == l1.size(2) * 2);
    DG_HOST_ASSERT(not pointers.empty() and rank >= 0 and rank < static_cast<int64_t>(pointers.size()));
    DG_HOST_ASSERT(buffer.is_cuda() and buffer.is_contiguous());
    DG_HOST_ASSERT(buffer.scalar_type() == torch::kInt8);
    DG_HOST_ASSERT(l1.device() == y.device() and l1_sf.device() == y.device());
    DG_HOST_ASSERT(l2.device() == y.device() and l2_sf.device() == y.device());
    DG_HOST_ASSERT(buffer.device() == y.device());
    mega::fp8_fp4_mega_moe_sm90(
        y, {l1, l1_sf}, {l2, l2_sf}, stats, buffer, pointers,
        mega::checked_int(rank), mega::checked_int(capacity),
        mega::checked_int(num_experts), mega::checked_int(num_topk),
        checked_recipe(recipe), activation, clamp, alpha, beta, fast_math,
        mega::checked_int(num_sms));
}

static void fp8_mega_moe(
    const torch::Tensor& y, const torch::Tensor& l1, const torch::Tensor& l1_sf,
    const torch::Tensor& l2, const torch::Tensor& l2_sf,
    const std::optional<torch::Tensor>& stats, const torch::Tensor& buffer,
    const std::vector<int64_t>& pointers, int64_t rank, int64_t capacity,
    int64_t num_experts, int64_t num_topk, const std::vector<int64_t>& recipe,
    const std::string& activation, const std::optional<double>& clamp, bool fast_math) {
    mega::fp8_mega_moe(
        y, {l1, l1_sf}, {l2, l2_sf}, stats, buffer, pointers,
        mega::checked_int(rank), mega::checked_int(capacity),
        mega::checked_int(num_experts), mega::checked_int(num_topk),
        checked_recipe(recipe), activation, clamp, fast_math);
}

static void mega_moe_pre_dispatch_sm90(
    const torch::Tensor& x, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
    const torch::Tensor& buf_x, const torch::Tensor& buf_x_sf,
    const torch::Tensor& buf_topk_idx, const torch::Tensor& buf_topk_weights,
    int64_t num_tokens, int64_t group_size, double routed_scaling_factor) {
    mega::mega_moe_pre_dispatch_sm90(
        x, topk_idx, topk_weights, buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
        mega::checked_int(num_tokens), mega::checked_int(group_size),
        static_cast<float>(routed_scaling_factor));
}

} // namespace deep_gemm::torch_registration::sm90

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    m.def("get_token_alignment_for_sm90_mega_moe() -> int");
    m.def("get_symm_buffer_size_for_sm90_mega_moe(int num_ranks, int num_experts, int capacity, int num_topk, int hidden, int intermediate_hidden, bool use_fp8_dispatch, str activation) -> (int, int[])");
    m.def("_slice_symm_buffer_for_sm90_mega_moe(Tensor(a) buffer, int[] geometry) -> (Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a))");
    m.def("fp8_fp4_mega_moe_sm90(Tensor(a!) y, Tensor l1, Tensor l1_sf, Tensor l2, Tensor l2_sf, Tensor(b!)? stats, Tensor(c!) buffer, int[] pointers, int rank, int capacity, int num_experts, int num_topk, int[3] recipe, str activation, float? clamp, float? alpha, float? beta, bool fast_math, int num_sms) -> ()");
    m.def("fp8_mega_moe(Tensor(a!) y, Tensor l1, Tensor l1_sf, Tensor l2, Tensor l2_sf, Tensor(b!)? stats, Tensor(c!) buffer, int[] pointers, int rank, int capacity, int num_experts, int num_topk, int[3] recipe, str activation, float? clamp, bool fast_math) -> ()");
    m.def("mega_moe_pre_dispatch_sm90(Tensor x, Tensor topk_idx, Tensor topk_weights, Tensor(a!) buf_x, Tensor(b!) buf_x_sf, Tensor(c!) buf_topk_idx, Tensor(d!) buf_topk_weights, int num_tokens, int group_size, float routed_scaling_factor) -> ()");
}

TORCH_LIBRARY_IMPL(deep_gemm, CatchAll, m) {
    m.impl("get_token_alignment_for_sm90_mega_moe", TORCH_FN(deep_gemm::torch_registration::sm90::get_token_alignment_for_sm90_mega_moe));
    m.impl("get_symm_buffer_size_for_sm90_mega_moe", TORCH_FN(deep_gemm::torch_registration::sm90::get_symm_buffer_size_for_sm90_mega_moe));
}

TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    m.impl("_slice_symm_buffer_for_sm90_mega_moe", TORCH_FN(deep_gemm::torch_registration::sm90::slice_symm_buffer));
    m.impl("fp8_fp4_mega_moe_sm90", TORCH_FN(deep_gemm::torch_registration::sm90::fp8_fp4_mega_moe_sm90));
    m.impl("fp8_mega_moe", TORCH_FN(deep_gemm::torch_registration::sm90::fp8_mega_moe));
    m.impl("mega_moe_pre_dispatch_sm90", TORCH_FN(deep_gemm::torch_registration::sm90::mega_moe_pre_dispatch_sm90));
}
