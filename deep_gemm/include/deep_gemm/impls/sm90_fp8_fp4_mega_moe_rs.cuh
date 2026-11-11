#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/fp4_decode_detail.cuh>
#include <deep_gemm/common/fp4_decode_rs.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

// Continuous FP32 activation scale, matching the FP8 path. SM90 WGMMA has no hardware
// block-scale operand (the SF is a plain FFMA in the epilogue), so the previous UE8M0
// (power-of-two) scale bought nothing on SM90 and only cost precision; the L2 SF pool is
// already fp32 (`l2_acts_sf` is kFloat32), so this is byte/layout neutral.
// This is the *activation* scale only -- the FP4 weight SFB stays UE8M0 (DSV4 external
// format) and is still decoded through the LUT path, untouched by this.
// Clamp amax before the reciprocal: padded rows have amax==0, and 448/0=inf -> 0*inf=NaN.
__forceinline__ __device__ void sm90_fp8_fp4_mega_moe_rs_get_e4m3_sf_and_sf_inv(
    const float2& amax, float2& sf, float2& sf_inv) {
    constexpr float kScale = 1.0f / 448.0f;
    const auto ax = fmaxf(amax.x, 1e-10f);
    const auto ay = fmaxf(amax.y, 1e-10f);
    sf.x = __fmul_rn(ax, kScale), sf_inv.x = 1.0f / sf.x;
    sf.y = __fmul_rn(ay, kScale), sf_inv.y = 1.0f / sf.y;
}

struct SM90FP8FP4MegaMoERSData {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    CUTLASS_HOST_DEVICE
    constexpr explicit SM90FP8FP4MegaMoERSData(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
#if defined(__CUDA_ARCH__)
        DG_TRAP_ONLY_DEVICE_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
#else
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
#endif
    }

    template <typename dtype_t = uint32_t>
    CUTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

struct SM90FP8FP4MegaMoERSBuffer {
    SM90FP8FP4MegaMoERSData data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;
    void* base;

    CUTLASS_HOST_DEVICE
    SM90FP8FP4MegaMoERSBuffer(const SM90FP8FP4MegaMoERSData& data_layout,
                            const uint32_t& num_ranks,
                            const uint32_t& max_num_tokens_per_rank,
                            void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(max_num_tokens_per_rank),
        base(base) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    CUTLASS_HOST_DEVICE
    SM90FP8FP4MegaMoERSBuffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    CUTLASS_HOST_DEVICE
    SM90FP8FP4MegaMoERSData get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
#if defined(__CUDA_ARCH__)
        DG_TRAP_ONLY_DEVICE_ASSERT(num_ranks == 1 or global);
#else
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
#endif
        return SM90FP8FP4MegaMoERSData(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

template <uint32_t kNumExpertsPerRank, uint32_t kNumExpertsPerLane, typename Scheduler>
CUTLASS_DEVICE void sm90_fp8_fp4_mega_moe_rs_fetch_cached_expert_recv_count(
    Scheduler& scheduler,
    const uint32_t* cached_recv_counts) {
    #pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
        const auto expert_idx = i * 32 + ptx::get_lane_idx();
        uint32_t value = 0;
        if (expert_idx < kNumExpertsPerRank)
            value = cached_recv_counts[expert_idx];
        scheduler.stored_num_tokens_per_expert[i] = value;
    }
    __syncwarp();
}

template <
    uint32_t kNumExpertsPerRank,
    uint32_t kNumExpertsPerLane,
    uint32_t kNumL1BlockKs,
    uint32_t kNumL2BlockKs,
    typename Scheduler,
    typename Func>
CUTLASS_DEVICE void sm90_fp8_fp4_mega_moe_rs_for_each_cached_block(
    Scheduler& scheduler,
    Func&& func,
    const uint32_t* cached_recv_counts) {
    sm90_fp8_fp4_mega_moe_rs_fetch_cached_expert_recv_count<
        kNumExpertsPerRank, kNumExpertsPerLane>(scheduler, cached_recv_counts);
    scheduler.set_expert_idx(0);

    while (true) {
        CUTE_TIE_DECL(scheduler.get_next_block(), block_phase, current_local_expert_idx, m_block_idx, n_block_idx);
        if (block_phase == sched::BlockPhase::None)
            break;

        func(block_phase, current_local_expert_idx,
             block_phase == sched::BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,
             m_block_idx, n_block_idx);
    }
}

// SM90 FP8 activations x FP4 weights, with swapAB register operands.
// Producers load packed weights and per-32 UE8M0 scales; math warpgroups
// decode directly into two K64 register buffers and overlap decode with WGMMA.
// Activation scales remain continuous FP32 with L1/L2 groups 128/64.

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    bool kUseSiTU,
    float kSiTUBeta,
    float kSiTULinearBeta,
    float kActivationClamp,
    bool kFastMath,
    uint32_t kL1ActsSFGranK = 128,
    uint32_t kL2ActsSFGranK = 64,
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumProducerWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_fp8_fp4_mega_moe_rs_impl(void* y,
                           int* cumulative_local_expert_recv_stats,
                           const uint32_t num_tokens,
                           const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                           const uint32_t* __restrict__ l1_weights_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                           const uint32_t* __restrict__ l2_weights_sf) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
#ifdef DG_SM90_FP4_RS_SHARED_LUT
    constexpr bool kSmallSharedLut = true;
#else
    constexpr bool kSmallSharedLut = false;
#endif
#ifdef DG_SM90_FP4_RS_DENSE
    constexpr bool kDenseRS = true;
#else
    constexpr bool kDenseRS = false;
#endif
    // =====================================================================
    // Template checks
    // =====================================================================
    DG_STATIC_ASSERT(kNumDispatchThreads == 64 and kNumNonEpilogueThreads == 64,
                              "RS requires two dispatch warps and two TMA producer warps");
    DG_STATIC_ASSERT(BLOCK_M == 64 and BLOCK_N == 256,
                              "Unsupported RS M/N tile");
    DG_STATIC_ASSERT(kNumEpilogueThreads == BLOCK_N * 2,
                              "Each math warpgroup owns 64 weight rows");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid expert count");
    DG_STATIC_ASSERT(BLOCK_K == 256, "Unsupported RS K tile");
    static_assert(kL1ActsSFGranK == 128 and kL2ActsSFGranK == 64);

    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

#ifdef DG_SM90_FP4_RS_SHARED_LUT
    // Publish through the existing CTA initialization barrier below.
    __shared__ uint64_t rs_decode_lut[256];
    if (thread_idx < 256)
        rs_decode_lut[thread_idx] = fp4_decode_detail::pack_scaled_e4m3_lut_from_e8m0_const(thread_idx);
#endif

    // Prefetch all TMA descriptors at the very beginning
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
    }

    // =====================================================================
    // Workspaces and symmetric buffer slicing (mirror SM100 layout, except SF
    // for L2 activations uses per-64 K granularity)
    // =====================================================================
    const auto workspace = layout::SM90Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    constexpr auto fp8_token_layout              = SM90FP8FP4MegaMoERSData(kHidden);
    constexpr auto bf16_token_layout             = SM90FP8FP4MegaMoERSData(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = SM90FP8FP4MegaMoERSData(kIntermediateHidden);
    constexpr auto fp8_sf_layout =
        SM90FP8FP4MegaMoERSData(kHidden * sizeof(float) / kL1ActsSFGranK);
    constexpr auto fp8_intermediate_sf_layout =
        SM90FP8FP4MegaMoERSData(kIntermediateHidden * sizeof(float) / kL2ActsSFGranK);
    constexpr auto input_topk_idx_layout         = SM90FP8FP4MegaMoERSData(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = SM90FP8FP4MegaMoERSData(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = SM90FP8FP4MegaMoERSData(sizeof(float), false);

    // Registered input area
    const auto input_token_buffer        = SM90FP8FP4MegaMoERSBuffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer           = SM90FP8FP4MegaMoERSBuffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer     = SM90FP8FP4MegaMoERSBuffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = SM90FP8FP4MegaMoERSBuffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

    // L1 input area
    const auto l1_token_buffer        = SM90FP8FP4MegaMoERSBuffer(fp8_token_layout, 1, kNumMaxPoolTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer           = SM90FP8FP4MegaMoERSBuffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = SM90FP8FP4MegaMoERSBuffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens, l1_sf_buffer.get_end_ptr());

    // L2 input area
    const auto l2_token_buffer = SM90FP8FP4MegaMoERSBuffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer    = SM90FP8FP4MegaMoERSBuffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());

    // Combine input area
    const auto combine_token_buffer = SM90FP8FP4MegaMoERSBuffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    // =====================================================================
    // GEMM data types and shape constants
    // =====================================================================
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_packed_dtype_t = int8_t;
    // All math warpgroups share the token rows and partition the weight rows.
    constexpr uint32_t kWarpgroupSplitN = kNumEpilogueWarpgroups;
    constexpr uint32_t WG_BLOCK_M = BLOCK_M;
    constexpr uint32_t WG_BLOCK_N = BLOCK_N / kWarpgroupSplitN;
    constexpr uint32_t kSwapABTokenChunks = BLOCK_M / 8;
    // After swapAB, WGMMA-N is the token dimension for both L1 and L2.
    using MaxTokenWGMMA = typename mma::sm90::FP8MMARSSelector<WG_BLOCK_M>::type;
    static_assert(WG_BLOCK_N == 64 and WG_BLOCK_M == 64);

    // Cluster=1 -> no multicast, A/B are loaded full-sized
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    constexpr uint32_t L1_OUT_BLOCK_N  = BLOCK_N / 2;  // post-gated activation
    // Each warpgroup stages its post-activation columns; WG0 stores the
    // combined tile after quantization using complete 64-column SF groups.
    constexpr bool kPrecomputedScatterPtr = kDenseRS;
    DG_STATIC_ASSERT(kNumMaxTokensPerRank % 2 == 0, "Pointer table alignment");
    const auto precomputed_scatter_ptrs = reinterpret_cast<uint64_t*>(workspace.get_token_src_metadata_ptr());
    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBPackedMode = 64;

    constexpr uint32_t kNumL1SFAPerBlockK = BLOCK_K / kL1ActsSFGranK;
    constexpr uint32_t kNumL2SFAPerBlockK = BLOCK_K / kL2ActsSFGranK;
    // SFB granularity for FP4 weights: per-32 K (DSV4 standard, UE8M0).
    // BLOCK_K=256 has 8 SFB groups along K, exactly one per WGMMA::K tile.
    constexpr uint32_t kScaleBGranK     = 32;
    static_assert(MaxTokenWGMMA::K == kScaleBGranK,
                  "WGMMA::K must equal kScaleBGranK so that 1 wgmma == 1 SFB block");

    // =====================================================================
    // Shared memory layout
    // =====================================================================
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    // Packed FP4 source tile (TMA-loaded raw nibbles)
    constexpr uint32_t SMEM_B_PACKED_SIZE_PER_STAGE = LOAD_BLOCK_N * (BLOCK_K / 2) * sizeof(b_packed_dtype_t);
    // SFA per-stage must be sized for the larger of L1 and L2.
    constexpr uint32_t kNumSFAPerBlockK =
        kNumL1SFAPerBlockK > kNumL2SFAPerBlockK
            ? kNumL1SFAPerBlockK
            : kNumL2SFAPerBlockK;
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        math::constexpr_align<uint32_t>(
            kNumSFAPerBlockK * BLOCK_M * sizeof(float), 128u);
    // SFB UE8M0 per-32: the B producer stages one packed uint32 per
    // N row per K128 in SMEM (two words for RS K256). Each word contains
    // four K/32 scale bytes, avoiding a global reload for each K group.
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE =
        math::constexpr_align<uint32_t>(LOAD_BLOCK_N * sizeof(uint32_t) * (BLOCK_K / 128), 128u);

    // L1 FP32/FP8 quantization staging and L2 BF16 scatter reuse one region.
    constexpr uint32_t SMEM_CD_L2_SIZE = BLOCK_M * BLOCK_N * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_CD_SWAP_L1_FP32_SIZE =
        BLOCK_M * L1_OUT_BLOCK_N * sizeof(float);
    constexpr uint32_t SMEM_CD_SWAP_L1_FP8_SIZE =
        BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t SMEM_CD_SWAP_L1_SIZE =
        SMEM_CD_SWAP_L1_FP32_SIZE + SMEM_CD_SWAP_L1_FP8_SIZE;
    constexpr uint32_t SMEM_CD_SIZE    = math::constexpr_align(
        SMEM_CD_L2_SIZE > SMEM_CD_SWAP_L1_SIZE ? SMEM_CD_L2_SIZE : SMEM_CD_SWAP_L1_SIZE,
        kSharedMemoryAlignment);

    // Retain the tuned stage-base padding while removing the unused amax scratch.
    constexpr uint32_t SMEM_PIPELINE_PADDING_SIZE = kSharedMemoryAlignment;

    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE +
        SMEM_PIPELINE_PADDING_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE +
                      SMEM_B_PACKED_SIZE_PER_STAGE +
                      SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE);

    // SMEM pointers
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = SM90FP8FP4MegaMoERSBuffer(
        fp8_token_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE);

    // CD output is shared by L1 (FP8) and L2 (BF16); reinterpret-cast as needed.

    auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_gemm_base);
    auto smem_cd_swap_l1_fp32 = reinterpret_cast<float*>(smem_gemm_base);
    auto smem_cd_swap_l1_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(
        math::advance_ptr(smem_gemm_base, SMEM_CD_SWAP_L1_FP32_SIZE));

    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(
            smem_gemm_base, SMEM_CD_SIZE + SMEM_PIPELINE_PADDING_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    // Packed FP4 source tile (TMA-loaded; consumed only by the math warpgroup
    // during the FP4-to-E4M3 dequant pass).
    auto smem_b_packed = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_packed_dtype_t>(smem_gemm_base,
            SMEM_CD_SIZE + SMEM_PIPELINE_PADDING_SIZE
            + kNumStages * (SMEM_A_SIZE_PER_STAGE)
            + i * SMEM_B_PACKED_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + SMEM_PIPELINE_PADDING_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_PACKED_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });

    auto sfb_start_ptr = sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE;
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sfb_start_ptr + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // Barriers live after SFA and staged SFB.
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sfb_start_ptr + kNumStages * SMEM_SFB_SIZE_PER_STAGE);
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + kNumDispatchWarps + kNumStages + i;
    });
    auto combine_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + kNumDispatchWarps + kNumStages + kNumStages + i;
    });

    // =====================================================================
    // Initialization
    // =====================================================================
    if (warp_idx == 0) {
        // Clean expert-count shared memory
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumExperts; i += 32)
            ptx::st_shared(smem_expert_count + i, 0u);
    } else if (warp_idx == 1) {
        // Init dispatch m-barriers
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM full/empty barriers and combine barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // A/SFA and packed B/SFB each contribute one producer arrival.
                full_barriers[i]->init(2);

                // Each math warp arrives once per stage release.
                empty_barriers[i]->init(kNumEpilogueWarps);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                combine_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // =====================================================================
    // Scheduler
    // =====================================================================
    constexpr uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u);
    constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
    constexpr uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N;
    constexpr uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K;
    constexpr uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K;
    auto scheduler = sched::LegacyMegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumExpertsPerWave,
        kNumSMs, kNumRanks,
        kNumExpertsPerLane, kNumL1BlockNs, kNumL2BlockNs,
        kNumL1BlockKs, kNumL2BlockKs,
        layout::SM90Workspace>(workspace);
    // Pipeline state shared by TMA loaders and math warpgroups
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM barrier indices (mirroring SM100)
    constexpr uint32_t kDispatchBarrierIdx              = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx  = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx          = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx       = 3;
    constexpr uint32_t kSchedulerCountCacheBarrierIdx   = 14;
    DG_STATIC_ASSERT(kEpilogueWGBarrierStartIdx + kNumEpilogueWarpgroups <= kSchedulerCountCacheBarrierIdx,
                     "Epilogue WG barriers overlap scheduler-count cache barrier");
    const uint32_t* cached_recv_counts = smem_expert_count;
    auto cache_expert_recv_counts = [&]() {
        if (thread_idx < kNumExpertsPerRank) {
            uint64_t value = 0;
            do {
                value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(thread_idx));
            } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            smem_expert_count[thread_idx] = static_cast<uint32_t>(value);
        }
        ptx::sync_unaligned(kNumThreads, kSchedulerCountCacheBarrierIdx);
    };

    // Cross-rank NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag    = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag   = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag   = 3;

    // One warpgroup contains two dispatch and two producer warps;
    // four math warpgroups decode weights, run WGMMA and perform epilogues.
    constexpr uint32_t kNumDispatchRegisters = 64;
    constexpr uint32_t kNumEpilogueRegisters = 104;
    // setmaxnreg redistributes the CTA's allocated pool, not all registers on
    // the SM. Account for the eight-register allocation quantum per thread;
    // a 640-thread CTA can own at most 96 * 640, rather than all 65,536.
    constexpr uint32_t kCTARegisterBudget =
        (65536 / (8 * kNumThreads)) * (8 * kNumThreads);
    DG_STATIC_ASSERT(
        (kNumDispatchThreads + kNumNonEpilogueThreads) * kNumDispatchRegisters +
            kNumEpilogueThreads * kNumEpilogueRegisters <= kCTARegisterBudget,
        "RS register quotas exceed the CTA pool");
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // SFB UE8M0 layouts (one uint32 per (n_row, 4 K32 groups = K128)):
    //   L1: shape [E, 2*IH, H/128] uint32, gran_mn=1 along N.
    //   L2: shape [E, H, IH/128] uint32.
    constexpr uint32_t kL1SFBKWords     = kHidden / 128;
    constexpr uint32_t kL2SFBKWords     = kIntermediateHidden / 128;
    constexpr uint32_t kL1SFBPerExpert  = (kIntermediateHidden * 2) * kL1SFBKWords;
    constexpr uint32_t kL2SFBPerExpert  = kHidden * kL2SFBKWords;
    // =====================================================================
    // ROLE 1: DISPATCH WARPS
    //   Mirrors SM100 dispatch with two changes:
    //     * SF is per-128 channel float (no UTCCP transpose). We store the
    //       remote per-token SF directly into the local L1 SF buffer in
    //       MN-major layout: `local_sf[k_chunk * num_padded_sf_pool_tokens + token_idx]`.
    //     * The "token_idx_in_expert" to SF token index is now the simple
    //       per-block linear mapping (no 4x32 transpose).
    // =====================================================================
    // Keep the whole RS producer warpgroup inside the deallocation branch.
    // Reconverging after setmaxnreg.dec and then splitting roles again makes
    // ptxas spill math state and serialize RS WGMMA (C7512). The math branch
    // below must remain disjoint until all producer work is finished.
    if (warp_idx < kNumDispatchWarps + kNumProducerWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();
        if (warp_idx < kNumDispatchWarps) {

            DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
            constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
            const auto read_topk_idx = [&](const auto& process) {
                #pragma unroll
                for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                     i < num_tokens;
                     i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                    int expert_idx = -1;
                    if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                        expert_idx = static_cast<int>(
                            __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                        if (expert_idx >= 0)
                            process(i * kNumTopk + lane_idx, expert_idx);
                    }
                    __syncwarp();
                }
            };

            // Count tokens per expert
            read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
                atomicAdd_block(smem_expert_count + expert_idx, 1);
            });
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            // Stake out per-expert SM offsets via global atomic
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
                smem_expert_count[i] = static_cast<uint32_t>(
                    ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
            }
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            // Write source token-topk indices to remote ranks
            read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
                const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
                const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
                const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                    expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
                *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
            });

            comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
                workspace, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
            );

            if (sm_idx == 0) {
                #pragma unroll
                for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                    const auto dst_rank_idx = i / kNumExpertsPerRank;
                    const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                    const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                    *sym_buffer.map(
                        workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                        dst_rank_idx) = expert_status & 0xffffffff;
                    ptx::atomic_add_sys(
                        sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                        expert_status);
                }
            }
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                                 kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                false, true);

            // Cache finalized expert counts before the dispatch/epilogue rendezvous
            // so loader warps can leave the all-CTA count barrier and start waiting
            // on L1 arrivals while dispatch and epilogue complete their handshake.
            cache_expert_recv_counts();

            // Sync with epilogue warps before pulling tokens
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

            // Token / SF pull loop
            uint32_t pull_mbarrier_phase = 0;
            const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
            const auto pull_mbarrier = dispatch_barriers[warp_idx];

            sm90_fp8_fp4_mega_moe_rs_fetch_cached_expert_recv_count<
                kNumExpertsPerRank, kNumExpertsPerLane>(scheduler, cached_recv_counts);

            constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
            int      current_expert_idx = -1;
            uint32_t stored_rank_count[kNumRanksPerLane] = {};
            uint32_t expert_start_idx = 0, expert_end_idx = 0;
            uint32_t expert_pool_block_offset = 0;

            constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
            for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
                int old_expert_idx = current_expert_idx;
                while (token_idx >= expert_end_idx) {
                    if (++ current_expert_idx >= kNumExpertsPerRank)
                        break;
                    expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                    expert_start_idx = expert_end_idx;
                    expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
                }
                if (current_expert_idx >= kNumExpertsPerRank)
                    break;

                if (old_expert_idx != current_expert_idx) {
                    old_expert_idx = current_expert_idx;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t j = i * 32 + lane_idx;
                        stored_rank_count[i] = j < kNumRanks ?
                            static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                    }
                }

                // Round-robin rank selection (identical to SM100)
                uint32_t current_rank_in_expert_idx;
                uint32_t remaining[kNumRanksPerLane];
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] = stored_rank_count[i];
                uint32_t offset = 0;
                uint32_t token_idx_in_expert = token_idx - expert_start_idx;
                uint32_t slot_idx = token_idx_in_expert;
                uint32_t token_idx_in_rank;
                while (true) {
                    uint32_t num_actives_in_lane = 0;
                    uint32_t min_in_lane = 0xffffffff;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        num_actives_in_lane += remaining[i] > 0;
                        if (remaining[i] > 0)
                            min_in_lane = cute::min(min_in_lane, remaining[i]);
                    }
                    const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                    const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                    const uint32_t num_round_tokens = length * num_active_ranks;
                    if (slot_idx < num_round_tokens) {
                        const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                        uint32_t num_seen_ranks = 0;
                        current_rank_in_expert_idx = 0;
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                            const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                            const uint32_t num_active_lanes = __popc(mask);
                            if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                                current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                            num_seen_ranks += num_active_lanes;
                        }
                        token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                        break;
                    }
                    slot_idx -= num_round_tokens;
                    offset += length;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                        remaining[i] -= cute::min(remaining[i], length);
                }

                const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                    current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
                const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
                const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

                // TMA pull token data into SMEM
                if (cute::elect_one_sync()) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                       current_rank_in_expert_idx),
                        pull_mbarrier, kHidden);
                }
                __syncwarp();

                // Copy FP32 activation SF, written linearly (no UTCCP transpose).
                constexpr uint32_t kNumSFFloats =
                    kHidden / kL1ActsSFGranK;
                DG_STATIC_ASSERT(
                    kNumSFFloats > 0 and kHidden % kL1ActsSFGranK == 0,
                    "Invalid SF");
                const auto remote_sf_ptr = sym_buffer.map(
                    input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>(),
                    current_rank_in_expert_idx);
                const auto local_sf_ptr  = l1_sf_buffer.get_base_ptr<float>();
                const uint32_t sf_pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
                #pragma unroll
                for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFFloats)
                        local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];
                }
                __syncwarp();

                const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
                if (cute::elect_one_sync()) {
                    const auto weight = *sym_buffer.map(
                        input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                        current_rank_in_expert_idx);
                    *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;

                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                    ptx::tma_store_1d(
                        l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                        pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                    if constexpr (kPrecomputedScatterPtr) {
                        const auto dst = combine_token_buffer.get_rank_buffer(src_topk_idx).get_data_buffer(src_token_idx);
                        precomputed_scatter_ptrs[pool_token_idx] = reinterpret_cast<uint64_t>(
                            sym_buffer.map(dst.get_base_ptr(), current_rank_in_expert_idx));
                    } else {
                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};
                    }

                    cute::tma_store_arrive();
                    ptx::tma_store_wait<0>();
                    ptx::red_add_rel(
                        workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
                }
                __syncwarp();
            }

            // Cleanup workspace, overlapping with combine.
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

            DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
            if (sm_idx == 0) {
                #pragma unroll
                for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                    *workspace.get_expert_send_count_ptr(i) = 0;
            } else {
                for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                    const auto num_recv_tokens = static_cast<uint32_t>(
                        *workspace.get_expert_recv_count_sum_ptr(i));
                    const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                    expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                    ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                    DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                    if (warp_idx == 0) {
                        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                    } else if (warp_idx == 1) {
                        if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                            ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                        __syncwarp();
                    }

                    for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                        *workspace.get_expert_recv_count_ptr(j, i) = 0;
                    __syncwarp();

                    for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                        *workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + j) = 0;
                        *workspace.get_l2_arrival_mask_ptr(expert_pool_block_offset + j) = 0;
                    }
                    __syncwarp();
                }
            }

            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                                 kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                true, false);

        // =====================================================================
        // ROLE 2: GEMM TMA LOAD warps (load A+SFA, B+SFB)
        //   Warps inside `kNumNonEpilogueThreads`: warp 0 loads A + SFA,
        //   warp 1 loads packed B + SFB; math warpgroups own register decode.
        // =====================================================================
        } else if (warp_idx == kNumDispatchWarps) {

            cache_expert_recv_counts();

            sm90_fp8_fp4_mega_moe_rs_for_each_cached_block<
                kNumExpertsPerRank, kNumExpertsPerLane, L1_SHAPE_K / BLOCK_K, L2_SHAPE_K / BLOCK_K>(
                scheduler, [&](const sched::BlockPhase& block_phase,
                               const uint32_t& local_expert_idx,
                               const uint32_t& num_k_blocks,
                               const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                const auto tensor_map_a_ptr = block_phase == sched::BlockPhase::Linear2
                    ? &tensor_map_l2_acts : &tensor_map_l1_acts;
                const auto tensor_map_sfa_ptr = block_phase == sched::BlockPhase::Linear2
                    ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

                const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;

                // Wait for the pool to be ready
                if (block_phase == sched::BlockPhase::Linear1) {
                    const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                    const auto expected = scheduler.template get_valid_m<false>();
                    while (ptx::ld_acq(ptr) != expected);
                } else {
                    constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
                    const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                    // Each L1 N block sets one bit; total bits = L1_SHAPE_N / BLOCK_N.
                    const uint64_t expected = (kNumL1BlockNs >= 64)
                        ? ~0ull : ((1ull << kNumL1BlockNs) - 1ull);
                    while (ptx::ld_acq_gpu(ptr) != expected);
                }
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    if (cute::elect_one_sync()) {
                        const uint32_t m_idx = pool_block_idx * BLOCK_M;
                        const uint32_t k_idx = k_block_idx * BLOCK_K;

                        // TMA load A
                        tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                            tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                            k_idx, m_idx, 1);

                        // TMA load SFA
                        if (block_phase == sched::BlockPhase::Linear1) {
                            tma::copy<BLOCK_M, kNumL1SFAPerBlockK, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx],
                                smem_sfa[stage_idx],
                                m_idx,
                                k_block_idx * kNumL1SFAPerBlockK,
                                1);
                            full_barriers[stage_idx]->arrive_and_expect_tx(
                                SMEM_A_SIZE_PER_STAGE +
                                kNumL1SFAPerBlockK * BLOCK_M * sizeof(float));
                        } else {
                            tma::copy<BLOCK_M, kNumL2SFAPerBlockK, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx],
                                smem_sfa[stage_idx],
                                m_idx,
                                k_block_idx * kNumL2SFAPerBlockK,
                                1);
                            full_barriers[stage_idx]->arrive_and_expect_tx(
                                SMEM_A_SIZE_PER_STAGE + kNumL2SFAPerBlockK * BLOCK_M * sizeof(float));
                        }
                    }
                    __syncwarp();

                }
            }, cached_recv_counts);

        } else if (warp_idx == kNumDispatchWarps + 1) {

            cache_expert_recv_counts();

            sm90_fp8_fp4_mega_moe_rs_for_each_cached_block<
                kNumExpertsPerRank, kNumExpertsPerLane, L1_SHAPE_K / BLOCK_K, L2_SHAPE_K / BLOCK_K>(
                scheduler, [&](const sched::BlockPhase& block_phase,
                               const uint32_t& local_expert_idx,
                               const uint32_t& num_k_blocks,
                               const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                const auto tensor_map_b_ptr =
                    block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights : &tensor_map_l1_weights;

                const uint32_t shape_n = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_N : L1_SHAPE_N;

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    if (cute::elect_one_sync()) {
                        const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                        // Packed FP4: each byte covers two K elements, so the
                        // logical K index advances by BLOCK_K (matching the FP8
                        // path), but the TMA box is BLOCK_K/2 bytes along the K
                        // dimension and the underlying tensor descriptor was built
                        // with `(N, K_packed = K/2)` shape on the host side. We
                        // pass `k_idx_packed = k_block_idx * (BLOCK_K / 2)` so the
                        // descriptor coordinate matches the packed K axis.
                        const uint32_t k_idx_packed = k_block_idx * (BLOCK_K / 2);

                        // TMA load packed FP4 weight tile into smem_b_packed.
                        auto b_full_barrier = full_barriers[stage_idx];
                        tma::copy<BLOCK_K / 2, LOAD_BLOCK_N, kSwizzleBPackedMode, b_packed_dtype_t>(
                            tensor_map_b_ptr, b_full_barrier, smem_b_packed[stage_idx],
                            k_idx_packed, n_idx, 1);
                    }
                    __syncwarp();

                    {
                        const bool is_l1 = block_phase == sched::BlockPhase::Linear1;
                        const uint32_t* sfb_base = is_l1 ? l1_weights_sf : l2_weights_sf;
                        const uint32_t sfb_per_expert = is_l1 ? kL1SFBPerExpert : kL2SFBPerExpert;
                        const uint32_t sfb_k_words = is_l1 ? kL1SFBKWords : kL2SFBKWords;
                        {
                            #pragma unroll
                            for (uint32_t row = lane_idx; row < LOAD_BLOCK_N; row += 32) {
                                const auto ptr = sfb_base + local_expert_idx * sfb_per_expert
                                    + (n_block_idx * BLOCK_N + row) * sfb_k_words + k_block_idx * 2;
                                reinterpret_cast<uint2*>(smem_sfb[stage_idx])[row] = __ldg(reinterpret_cast<const uint2*>(ptr));
                            }
                        }
                    }
                    __syncwarp();

                    if (cute::elect_one_sync()) {
                        {
                            full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_PACKED_SIZE_PER_STAGE);
                        }
                    }
                    __syncwarp();

                }
            }, cached_recv_counts);

        }
    } else {
        // =====================================================================
        // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
        // =====================================================================
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();
        // Epilogues exchange ordinary C++ shared-memory loads/stores as well
        // as PTX operations, so the rendezvous is also a compiler memory barrier.
        auto sync_epilogue = [](uint32_t count, uint32_t index) {
            asm volatile("bar.sync %0, %1;" :: "r"(index), "r"(count) : "memory");
        };

        const uint32_t epilogue_warp_idx  = warp_idx - (kNumDispatchWarps + kNumProducerWarps);
        const uint32_t epilogue_wg_idx    = epilogue_warp_idx / 4;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t warp_idx_in_wg     = epilogue_warp_idx % 4;

        // WGMMA-output register layout helpers
        const uint32_t row_idx = lane_idx / 4;
        const uint32_t col_idx = lane_idx % 4;
        const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
        const uint32_t r_1 = r_0 + 8;

        const uint32_t epilogue_wg_n_idx = epilogue_wg_idx;
        const uint32_t wg_n_offset = epilogue_wg_n_idx * WG_BLOCK_N;
        const uint32_t wg_l1_out_n_offset = epilogue_wg_n_idx * (WG_BLOCK_N / 2);
        const uint32_t smem_cd_l2_wg_offset = wg_n_offset;

        cache_expert_recv_counts();

        // Sync with dispatch
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        sm90_fp8_fp4_mega_moe_rs_for_each_cached_block<
            kNumExpertsPerRank, kNumExpertsPerLane, L1_SHAPE_K / BLOCK_K, L2_SHAPE_K / BLOCK_K>(
            scheduler, [&](const sched::BlockPhase& block_phase,
                           const uint32_t& local_expert_idx,
                           const uint32_t& num_k_blocks,
                           const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block_idx * BLOCK_N;

            // ---------------- GEMM ----------------
            constexpr uint32_t kAccumPerThread = MaxTokenWGMMA::kNumAccum;  // 32 for M=64,N=64
            float final_accum[kAccumPerThread] = {};
            // The K-stage loop is instantiated per (phase, swapAB-N): tile
            // invariant decisions (the Linear1/Linear2 branch and the swapAB
            // n_swap dispatch chain) used to run on every K-stage of the math
            // warp's critical path, which sets the whole pipeline's steady
            // beat; hoisting them to one dispatch per tile also lets the
            // compiler specialize each loop body.
            auto run_k_stages = [&]<bool kIsL1, uint32_t kNSwap>() {
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);

                    {
                        using RSWGMMA = typename mma::sm90::FP8MMARSSelector<kNSwap>::type;
                        constexpr uint32_t kRSAccum = RSWGMMA::kNumAccum;
                        constexpr uint32_t kSFGran = kIsL1 ? kL1ActsSFGranK : kL2ActsSFGranK;
                        constexpr uint32_t kMmasPerSF = kSFGran / 32;
                        float accum[kRSAccum] = {};
                        // Two K64 fragments: 2 buffers x 2 K32 MMA operands x
                        // 4 registers = 16 weight registers per math thread.
                        static_assert(kMmasPerSF >= 2);
                        uint32_t weight[2][2][4];
                        auto decode = [&](uint32_t half, uint32_t (&fragment)[2][4]) {
                            decode_fp4_rs_k64<BLOCK_K, LOAD_BLOCK_N>(smem_b_packed[stage_idx], smem_sfb[stage_idx],
                                wg_n_offset, half, lane_idx, warp_idx_in_wg, fragment
#ifdef DG_SM90_FP4_RS_SHARED_LUT
                                , rs_decode_lut
#endif
                            );
                        };
                        decode(0, weight[0]);
                        #pragma unroll
                        for (uint32_t half = 0; half < BLOCK_K / 64; ++half) {
                            #pragma unroll
                            for (uint32_t j = 0; j < 2; ++j)
                                fence_fp4_rs_fragment(weight[half % 2][j]);
                            #pragma unroll
                            for (uint32_t j = 0; j < kRSAccum; ++j)
                                ptx::warpgroup_fence_operand(accum[j]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t j = 0; j < 2; ++j) {
                                const uint32_t q = half * 2 + j;
                                auto desc_b = mma::sm90::make_smem_desc(smem_a[stage_idx] + (q / 4) * BLOCK_M * 128 + (q % 4) * 32, 1);
                                RSWGMMA::wgmma(weight[half % 2][j], desc_b, accum, q % kMmasPerSF != 0);
                            }
                            ptx::warpgroup_commit_batch();
                            // Decode the next K64 half while both current K32
                            // WGMMAs are outstanding; keep their RS operands live.
                            if (half + 1 < BLOCK_K / 64)
                                decode(half + 1, weight[(half + 1) % 2]);
                            ptx::warpgroup_wait<0>();
                            #pragma unroll
                            for (uint32_t j = 0; j < 2; ++j)
                                fence_fp4_rs_fragment(weight[half % 2][j]);
                            #pragma unroll
                            for (uint32_t j = 0; j < kRSAccum; ++j)
                                ptx::warpgroup_fence_operand(accum[j]);
                            if (((half + 1) * 2) % kMmasPerSF == 0) {
                                const uint32_t sf_group = (half * 2) / kMmasPerSF;
                                #pragma unroll
                                for (uint32_t j = 0; j < kRSAccum / 4; ++j) {
                                    const uint32_t t0 = j * 8 + col_idx * 2;
                                    const uint32_t t1 = t0 + 1;
                                    // Dense and shared-LUT RS promote the allocated token tile.
                                    // Padded columns remain private accumulators and are
                                    // discarded by the valid-token epilogue below.
                                    if (kDenseRS or kSmallSharedLut or t0 < valid_m) {
                                        const float sf = ptx::ld_shared(smem_sfa[stage_idx] + sf_group * BLOCK_M + t0);
                                        final_accum[j*4+0] += sf * accum[j*4+0];
                                        final_accum[j*4+2] += sf * accum[j*4+2];
                                    }
                                    if (kDenseRS or kSmallSharedLut or t1 < valid_m) {
                                        const float sf = ptx::ld_shared(smem_sfa[stage_idx] + sf_group * BLOCK_M + t1);
                                        final_accum[j*4+1] += sf * accum[j*4+1];
                                        final_accum[j*4+3] += sf * accum[j*4+3];
                                    }
                                }
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (lane_idx == 0)
                            empty_barriers[stage_idx]->arrive();
                    }
                }
            };

            // Per-tile dispatch of the specialized K-stage loop. `n_swap` and
            // the phase are tile constants, so decide them once here instead
            // of on every K-stage inside the loop.
            // Wide RS uses a bounded set of token tiles. Extra large-N
            // specializations increase live coordinates and spill at 104 regs.
            {
                if constexpr (kDenseRS) {
                    // Keep four specializations within the 104-register budget.
#ifdef DG_SM90_FP4_RS_DENSE_48
                    constexpr uint32_t kMiddleTokenBucket = 48;
#else
                    constexpr uint32_t kMiddleTokenBucket = 40;
#endif
                    const uint32_t n_swap = kUseSiTU ? valid_m : ((valid_m + 7u) / 8u) * 8u;
                    if (block_phase == sched::BlockPhase::Linear1) {
                        if (n_swap <= 8) run_k_stages.template operator()<true, 8>();
                        else if (n_swap <= 32) run_k_stages.template operator()<true, 32>();
                        else if (n_swap <= kMiddleTokenBucket) run_k_stages.template operator()<true, kMiddleTokenBucket>();
                        else run_k_stages.template operator()<true, 64>();
                    } else {
                        if (n_swap <= 8) run_k_stages.template operator()<false, 8>();
                        else if (n_swap <= 32) run_k_stages.template operator()<false, 32>();
                        else if (n_swap <= kMiddleTokenBucket) run_k_stages.template operator()<false, kMiddleTokenBucket>();
                        else run_k_stages.template operator()<false, 64>();
                    }
                } else {
                    // Separate SwiGLU bucket predicates from per-token validity.
                    // SiTU has a different epilogue lifetime and keeps the direct count.
                    // Both expressions select identical 8/16/24/32/64 tile sizes.
                    const uint32_t n_swap = kUseSiTU ? valid_m : ((valid_m + 7u) / 8u) * 8u;
                    if (block_phase == sched::BlockPhase::Linear1) {
                        if (n_swap <= 8) run_k_stages.template operator()<true, 8>();
                        else if (n_swap <= 16) run_k_stages.template operator()<true, 16>();
                        // The shared-LUT specialization omits N24 to reduce dispatch specializations.
#ifndef DG_SM90_FP4_RS_SHARED_LUT
                        else if (n_swap <= 24) run_k_stages.template operator()<true, 24>();
#endif
                        else if (n_swap <= 32) run_k_stages.template operator()<true, 32>();
                        else run_k_stages.template operator()<true, 64>();
                    } else {
                        if (n_swap <= 8) run_k_stages.template operator()<false, 8>();
                        else if (n_swap <= 16) run_k_stages.template operator()<false, 16>();
#ifndef DG_SM90_FP4_RS_SHARED_LUT
                        else if (n_swap <= 24) run_k_stages.template operator()<false, 24>();
#endif
                        else if (n_swap <= 32) run_k_stages.template operator()<false, 32>();
                        else run_k_stages.template operator()<false, 64>();
                    }
                }
            }

            // All warpgroups share the token rows. Empty tiles still participate
            // in the epilogue rendezvous after releasing their pipeline stages.
            if (valid_m == 0) {
                sync_epilogue(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                return;
            }

            // L1 supports SwiGLU (clamp + silu * up) and SiTU.
            auto gated_activation = [](float gate, float up) -> float {
                if constexpr (kUseSiTU) {
                    const float sigmoid = kFastMath
                        ? 0.5f * (1.0f + __tanhf(0.5f * gate))
                        : 1.0f / (1.0f + expf(-gate));
                    const float gate_tanh = kFastMath
                        ? __tanhf(gate / kSiTUBeta)
                        : tanhf(gate / kSiTUBeta);
                    const float up_tanh = kFastMath
                        ? __tanhf(up / kSiTULinearBeta)
                        : tanhf(up / kSiTULinearBeta);
                    const float capped_gate = kSiTUBeta * gate_tanh;
                    const float capped_up = kSiTULinearBeta * up_tanh;
                    return capped_gate * sigmoid * capped_up;
                } else {
                    if constexpr (kActivationClamp !=
                                  cute::numeric_limits<float>::infinity()) {
                        gate = cute::min(gate, kActivationClamp);
                        up = cute::min(cute::max(up, -kActivationClamp),
                                       kActivationClamp);
                    }
                    float e;
                    if constexpr (kFastMath and kActivationClamp <= 80.0f) {
                        const float exponent = __fmul_rn(gate, -1.4426950408889634f);
                        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(exponent));
                    } else {
                        e = kFastMath ? __expf(-gate) : expf(-gate);
                    }
                    const float sigmoid = kFastMath
                        ? math::fast_rcp(1.0f + e)
                        : 1.0f / (1.0f + e);
                    return gate * sigmoid * up;
                }
            };

            if (block_phase == sched::BlockPhase::Linear1) {
                {
                    // Each split-N WG writes its disjoint 32-column range into
                    // a shared token-major FP32 tile. Complete token rows are
                    // scanned for amax before quantization.
                    const uint32_t out_col_base =
                        wg_l1_out_n_offset + warp_idx_in_wg * 8 + row_idx;
                    auto store_l1_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        if (token_0 < valid_m) {
                            const float act0 = gated_activation(
                                final_accum[i * 4 + 0], final_accum[i * 4 + 2]);
                            smem_cd_swap_l1_fp32[
                                token_0 * L1_OUT_BLOCK_N + out_col_base] = act0;
                        }
                        if (token_1 < valid_m) {
                            const float act1 = gated_activation(
                                final_accum[i * 4 + 1], final_accum[i * 4 + 3]);
                            smem_cd_swap_l1_fp32[
                                token_1 * L1_OUT_BLOCK_N + out_col_base] = act1;
                        }
                    };

                    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                    store_l1_swap_chunk(0);
                    if (valid_m > 8) {
                        #pragma unroll
                        for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                            if (i < num_swap_token_chunks)
                                store_l1_swap_chunk(i);
                        }
                    }

                    sync_epilogue(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    if constexpr (kDenseRS) {
                        static_assert(L1_OUT_BLOCK_N == 128 and kL2ActsSFGranK == 64);
                        // Each half warp owns one complete 64-column SF group.
                        // The paired groups share the token and routing weight.
                        // Both disjoint 16-lane masks have every member participating;
                        // retain the original SF arithmetic and FP8 pair conversion.
                        for (uint32_t token = epilogue_warp_idx; token < valid_m; token += kNumEpilogueWarps) {
                            const uint32_t col = lane_idx * 4;
                            const float4 values = *reinterpret_cast<const float4*>(
                                smem_cd_swap_l1_fp32 + token * L1_OUT_BLOCK_N + col);
                            float amax = cute::max(0.0f, cute::abs(values.x));
                            amax = cute::max(amax, cute::abs(values.y));
                            amax = cute::max(amax, cute::abs(values.z));
                            amax = cute::max(amax, cute::abs(values.w));
                            const uint32_t group_mask = lane_idx < 16 ? 0x0000ffffu : 0xffff0000u;
                            amax = __uint_as_float(__reduce_max_sync(group_mask, __float_as_uint(amax)));
                            const float wtok = *l1_topk_weights_buffer.get_data_buffer(m_idx + token).get_base_ptr<float>();
                            amax *= cute::abs(wtok);
                            float2 amax_pair = {amax, amax}, sf_pair, sf_inv_pair;
                            sm90_fp8_fp4_mega_moe_rs_get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                            const float sf_inv = wtok * sf_inv_pair.x;
                            if (lane_idx % 16 == 0) {
                                const uint32_t sf_idx = n_block_idx * 2 + lane_idx / 16;
                                l2_sf_buffer.get_base_ptr<float>()[sf_idx * kNumPaddedSFPoolTokens + m_idx + token] = sf_pair.x;
                            }
                            const __nv_fp8x2_e4m3 lo(make_float2(values.x * sf_inv, values.y * sf_inv));
                            const __nv_fp8x2_e4m3 hi(make_float2(values.z * sf_inv, values.w * sf_inv));
                            *reinterpret_cast<uint32_t*>(smem_cd_swap_l1_fp8 + token * L1_OUT_BLOCK_N + col) =
                                static_cast<uint32_t>(lo.__x) | (static_cast<uint32_t>(hi.__x) << 16);
                        }
                    } else {
                        static_assert(kL2ActsSFGranK == 64);
                        // One warp reduces and quantizes one 64-column group.
                        // Read the completed FP32 tile after the same WG barrier;
                        // retain the original SF and pairwise FP8 conversion.
                        for (uint32_t token = epilogue_warp_idx; token < valid_m; token += kNumEpilogueWarps) {
                            #pragma unroll
                            for (uint32_t sf_group = 0; sf_group < L1_OUT_BLOCK_N / 64; ++sf_group) {
                                float4 values = {0, 0, 0, 0};
                                const uint32_t col = sf_group * 64 + lane_idx * 4;
                                if (lane_idx < 16)
                                    values = *reinterpret_cast<const float4*>(
                                        smem_cd_swap_l1_fp32 + token * L1_OUT_BLOCK_N + col);
                                float amax = cute::max(0.0f, cute::abs(values.x));
                                amax = cute::max(amax, cute::abs(values.y));
                                amax = cute::max(amax, cute::abs(values.z));
                                amax = cute::max(amax, cute::abs(values.w));
                                amax = __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(amax)));
                                const float wtok = *l1_topk_weights_buffer
                                    .get_data_buffer(m_idx + token).get_base_ptr<float>();
                                amax *= cute::abs(wtok);
                                float2 amax_pair = {amax, amax}, sf_pair, sf_inv_pair;
                                sm90_fp8_fp4_mega_moe_rs_get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                                const float sf_inv = wtok * sf_inv_pair.x;
                                if (lane_idx == 0) {
                                    const uint32_t sf_idx = n_block_idx * (L1_OUT_BLOCK_N / 64) + sf_group;
                                    l2_sf_buffer.get_base_ptr<float>()[sf_idx * kNumPaddedSFPoolTokens + m_idx + token] = sf_pair.x;
                                }
                                if (lane_idx < 16) {
                                    const __nv_fp8x2_e4m3 lo(make_float2(values.x * sf_inv, values.y * sf_inv));
                                    const __nv_fp8x2_e4m3 hi(make_float2(values.z * sf_inv, values.w * sf_inv));
                                    *reinterpret_cast<uint32_t*>(smem_cd_swap_l1_fp8 + token * L1_OUT_BLOCK_N + col) =
                                        static_cast<uint32_t>(lo.__x) | (static_cast<uint32_t>(hi.__x) << 16);
                                }
                            }
                        }
                    }

                    sync_epilogue(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_swap_l1_fp8,
                            n_block_idx * L1_OUT_BLOCK_N,
                            m_idx);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    ptx::tma_store_wait<0>();

                    sync_epilogue(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        ptx::red_or_rel_gpu(
                            workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                            1ull << n_block_idx);
                    }
                    __syncwarp();
                }
            } else {
                // ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------
                {
                    constexpr uint32_t kNumRowsPerWarp = WG_BLOCK_M / 8;

                    {
                        auto store_bf16 = [&](const uint32_t& token, const uint32_t& col, float value) {
                            smem_cd_l2[smem_cd_l2_wg_offset + token * BLOCK_N + col] =
                                __float2bfloat16_rn(value);
                        };

                        auto store_l2_swap_chunk = [&](const uint32_t& i) {
                            const uint32_t token_0 = i * 8 + col_idx * 2;
                            const uint32_t token_1 = token_0 + 1;
                            if (token_0 < valid_m) {
                                store_bf16(token_0, r_0, final_accum[i * 4 + 0]);
                                store_bf16(token_0, r_1, final_accum[i * 4 + 2]);
                            }
                            if (token_1 < valid_m) {
                                store_bf16(token_1, r_0, final_accum[i * 4 + 1]);
                                store_bf16(token_1, r_1, final_accum[i * 4 + 3]);
                            }
                        };

                        const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                        store_l2_swap_chunk(0);
                        if (valid_m > 8) {
                            #pragma unroll
                            for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                                if (i < num_swap_token_chunks)
                                    store_l2_swap_chunk(i);
                            }
                        }
                    }

                    {
                        sync_epilogue(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        // Scatter to remote ranks via NVLink (one row per warp-pair)
                        // Each warpgroup-warp covers 8 unique rows x 2 (r_0 + r_1 doubled by warps)
                        // Lane group of 16 within a warp maps to one row.
                        const uint32_t row_in_warp_block = lane_idx / 16;  // 0 or 1
                        const uint32_t lane_in_row = lane_idx % 16;
                        // Each warpgroup owns 64 columns of every token row.
                        constexpr uint32_t kColsPerScatterLane = WG_BLOCK_N / 16;
                        static_assert(kColsPerScatterLane == 4);
                        using ScatterVector = uint2;

                        #pragma unroll
                        for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                            const uint32_t row_in_wg = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                            const uint32_t m_idx_in_block = row_in_wg;
                            if (m_idx_in_block >= valid_m) break;

                            // Each lane owns 4 BF16s (8B); a uint4 load would
                            // be misaligned for odd lanes.
                            auto smem_ptr = smem_cd_l2
                                + smem_cd_l2_wg_offset
                                + row_in_wg * BLOCK_N
                                + lane_in_row * kColsPerScatterLane;
                            if constexpr (kPrecomputedScatterPtr) {
                                const auto remote_row_base = reinterpret_cast<uint8_t*>(
                                    precomputed_scatter_ptrs[m_idx + m_idx_in_block]);
                                const auto packed = *reinterpret_cast<ScatterVector*>(smem_ptr);
                                auto row_dst = reinterpret_cast<ScatterVector*>(remote_row_base +
                                    (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + lane_in_row * sizeof(ScatterVector));
                                *row_dst = packed;
                            } else {
                                const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                                const auto dst_token = combine_token_buffer.get_rank_buffer(src_metadata.topk_idx)
                                                       .get_data_buffer(src_metadata.token_idx);
                                const auto packed = *reinterpret_cast<ScatterVector*>(smem_ptr);
                                auto dst_ptr = math::advance_ptr<ScatterVector>(
                                    dst_token.get_base_ptr(),
                                    (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + lane_in_row * sizeof(ScatterVector));
                                *sym_buffer.map(dst_ptr, src_metadata.rank_idx) = packed;
                            }
                        }
                    }
                }

                sync_epilogue(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
        }, cached_recv_counts);

        // ---------------- COMBINE ----------------
        // NVLink barrier first: signals remote ranks that this rank's GEMM
        // outputs (NVLink scatter targets) are fully written.
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                             kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() { sync_epilogue(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );

        // Sync with dispatch (paired with dispatch's pre-cleanup sync) so that
        // dispatch may now safely clean workspace state.
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;
        constexpr bool kOneCombineChunkFits =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE;
        constexpr bool kTwoCombineChunksFit =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / 2 <= SMEM_BEFORE_BARRIER_SIZE;
        constexpr uint32_t kNumChunks =
            (kOneCombineChunkFits and kHidden <= 32 * kNumMaxRegistersForBuffer) ? 1 :
            (kTwoCombineChunksFit ? 2 : 4);
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= SMEM_BEFORE_BARRIER_SIZE, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        DG_TRAP_ONLY_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= static_cast<uint32_t>(
            reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer));

        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer = math::advance_ptr<uint4>(
            smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumEpilogueWarps) {
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;
                        // Order all lanes' shared reads before the next TMA write.
                        cutlass::arch::fence_view_async_shared();
                        __syncwarp();
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                bool do_reduce = move_mask_and_load(load_stage_idx);

                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_TRAP_ONLY_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
