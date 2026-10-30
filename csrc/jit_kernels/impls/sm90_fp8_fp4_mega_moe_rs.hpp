#pragma once

#include "../../runtime/runtime.hpp"

#include <torch/torch.h>
#include "../../utils/exception.hpp"
#include <format>
#include "runtime_utils.hpp"
#include "../../runtime/launch.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/sm90_fp4_rs_mega_moe.hpp"

namespace deep_gemm {

// Single-node SM90 FP8 x FP4 MegaMoE uses swapAB with register weight operands.
// Weights use the lossless RS byte/sign layout produced by the Python packer;
// per-32 weight SF is packed UE8M0 and activation SF remains FP32 (128/64).
class SM90FP8FP4MegaMoERSRuntime final {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        int l1_act_group, l2_act_group;
        bool use_situ;
        float situ_beta;
        float situ_linear_beta;
        float activation_clamp;
        bool fast_math;
        MegaMoESM90Config config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormaps for activations (FP8) and packed FP4 weights.
        // Weight UE8M0 SFB are passed as raw uint32* (no TMA descriptor).
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        const uint32_t* l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        const uint32_t* l2_weights_sf;

        // Launch configs
        deep_jit::cuda::LaunchOptions launch_args;
    };

    static std::string generate(const Args& args) {
        std::string prefix;
        {
            const float tokens_per_expert =
                static_cast<float>(args.num_tokens) * args.num_ranks * args.num_topk / args.num_experts;
            // Sparse wide RS tiles use a shared decode LUT and four token
            // buckets to reduce the decode/promote specialization footprint.
            // The FP4 config fixes BN=BK=256. Reserve the 2 KiB LUT plus
            // conservative alignment headroom in addition to dynamic shared memory.
            if (tokens_per_expert <= 8.0f and args.config.smem_size + 4096 <= SM90ArchSpec::smem_capacity)
                prefix += "#define DG_SM90_FP4_RS_SHARED_LUT 1\n";
            // Dense wide RS tiles use four token buckets to avoid padding
            // 33..40 tokens to 64 while bounding the specialization footprint.
            // This dispatch also supports unclamped SwiGLU and SiTU.
            // SiTU keeps finer sparse buckets through 24 tokens/expert;
            // replacing N16/N24 below this density adds more padding work.
            const float dense_rs_min_tokens = args.use_situ ? 24.0f : 20.0f;
            if (tokens_per_expert > dense_rs_min_tokens) {
                prefix += "#define DG_SM90_FP4_RS_DENSE 1\n";
                // Moderately filled experts benefit from a 48-token bucket.
                // Keep the 40-token bucket for lower density and multi-tile experts.
                if (tokens_per_expert > 40.0f and tokens_per_expert <= 56.0f)
                    prefix += "#define DG_SM90_FP4_RS_DENSE_48 1\n";
            }
        }
        return prefix + std::format(R"(
#include <deep_gemm/impls/sm90_fp8_fp4_mega_moe_rs.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_fp4_mega_moe_rs_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}, {},
        {},
        {},
        {}, {}
    >);
}};
)",
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim->x, args.num_ranks,
    args.use_situ ? "true" : "false",
    to_string(args.situ_beta), to_string(args.situ_linear_beta),
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.l1_act_group, args.l2_act_group);
    }

    static void launch(const std::shared_ptr<deep_jit::cuda::Kernel>& kernel, const Args& args) {
        jit->launch(kernel, args.launch_args,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.l2_weights_sf
        );
    }
};

static void sm90_fp8_fp4_mega_moe_rs(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& activation,
    const float& activation_alpha,
    const float& activation_linear_beta,
    const float& activation_clamp,
    const bool& fast_math,
    const int& num_sms_override = 0
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));
    DG_HOST_ASSERT(activation == "swiglu" or activation == "situ");
    const bool use_situ = activation == "situ";
    DG_HOST_ASSERT(not use_situ or
                   (activation_alpha > 0.0f and activation_linear_beta > 0.0f));

    // Sanity: SFB tensors must be uint32 (UE8M0 packed) and weight tensors
    // must use byte-addressable packed FP4 storage (1 byte = 2 nibbles).
    DG_HOST_ASSERT(l1_weights_sf.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(l2_weights_sf.scalar_type() == torch::kInt);

    const int l1_act_group = hidden / l1_acts_sf.size(1);
    const int l2_act_group = intermediate_hidden / l2_acts_sf.size(1);

    DG_HOST_ASSERT(l1_act_group == 128 and l2_act_group == 64);
    const auto config = get_mega_moe_config_sm90_fp4_rs(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens, num_sms_override);

    // Report the final configuration after selecting the weight layout.
    if (deep_jit::get_env<int>("DG_JIT_DEBUG") or deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = std::format(
            "MegaMoESM90FP4Config(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={}, use_situ={}, weight_layout=rs)",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk,
            use_situ);
        static std::unordered_set<std::string> printed;
        if (printed.insert(key).second)
            std::cout << key << ": " << config << std::endl;
    }

    // Tensormap construction
    // Acts: FP8 e4m3, identical to FP8 path
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_padded_sf_pool_tokens, hidden,
                                                        config.block_m, l1_act_group,
                                                        1, 0, 0, false,
                                                        config.block_k / l1_act_group);

    // Packed FP4 weight tile: each byte = 2 nibbles. SM90 loads these as raw
    // bytes and software-decodes them before WGMMA, so the TensorMap must be a
    // UINT8 view with a packed K axis rather than a native FP4 TensorMap.
    const auto l1_weights_bytes = l1_weights.scalar_type() == torch::kByte
        ? l1_weights : l1_weights.view(torch::kByte);
    const auto l2_weights_bytes = l2_weights.scalar_type() == torch::kByte
        ? l2_weights : l2_weights.view(torch::kByte);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights_bytes,
                                                        hidden / 2, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k / 2, config.block_n,
                                                        static_cast<int>(l1_weights_bytes.stride(-2)),
                                                        config.swizzle_weights_mode, /*swizzle_base=*/0,
                                                        /*allow_tf32=*/false);

    // All math warpgroups stage one combined post-activation tile.
    const int tma_l1_out_box_n = config.block_n / 2;
    const int tma_l1_out_box_m = config.block_m;
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_max_pool_tokens,
                                                       tma_l1_out_box_n, tma_l1_out_box_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       0);

    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_padded_sf_pool_tokens, intermediate_hidden,
                                                        config.block_m, l2_act_group,
                                                        1, 0, 0, false,
                                                        config.block_k / l2_act_group);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights_bytes,
                                                        intermediate_hidden / 2, num_experts_per_rank * hidden,
                                                        config.block_k / 2, config.block_n,
                                                        static_cast<int>(l2_weights_bytes.stride(-2)),
                                                        config.swizzle_weights_mode, /*swizzle_base=*/0,
                                                        /*allow_tf32=*/false);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
    // Launch
    const auto num_sms = num_sms_override ? num_sms_override : runtime->get_num_sms();
    const SM90FP8FP4MegaMoERSRuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .l1_act_group = l1_act_group, .l2_act_group = l2_act_group,
        .use_situ = use_situ,
        .situ_beta = activation_alpha,
        .situ_linear_beta = activation_linear_beta,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .l1_weights_sf = reinterpret_cast<const uint32_t*>(l1_weights_sf.data_ptr()),
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .l2_weights_sf = reinterpret_cast<const uint32_t*>(l2_weights_sf.data_ptr()),
        .launch_args = make_launch_options(num_sms, config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, config.cluster_size)
    };
    const auto code = SM90FP8FP4MegaMoERSRuntime::generate(args);
    const auto runtime_name = "sm90_fp8_fp4_mega_moe_rs";
    const auto kernel = jit->compile(runtime_name, code);
    SM90FP8FP4MegaMoERSRuntime::launch(kernel, args);
}

} // namespace deep_gemm
