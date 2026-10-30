#pragma once

#include <deep_gemm/common/fp4_decode_detail.cuh>

namespace deep_gemm {

CUTLASS_DEVICE void fp4x8_rs_interleaved_to_e4m3(
        uint32_t packed, uint32_t lut_lo, uint32_t lut_hi,
        uint32_t& out_lo, uint32_t& out_hi) {
    const uint32_t indices = packed & 0x77777777u;
    asm volatile(
        "{ .reg .b32 ml, mh;\n"
        "prmt.b32 ml, %4, %5, %2;\n"
        "prmt.b32 mh, %4, %5, %3;\n"
        "lop3.b32 %0, %6, 0x80808080, ml, 0xea;\n"
        "lop3.b32 %1, %7, 0x80808080, mh, 0xea;\n}"
        : "=r"(out_lo), "=r"(out_hi)
        : "r"(indices), "r"(indices >> 16), "r"(lut_lo), "r"(lut_hi),
          "r"(packed << 4), "r"(packed));
}

// One collective load brings in both K32 operands for this K64 half.
template <uint32_t BlockK, uint32_t LoadN>
CUTLASS_DEVICE void decode_fp4_rs_k64(
        const int8_t* packed, const uint32_t* sf,
        uint32_t row_base, uint32_t half, uint32_t lane, uint32_t warp,
        uint32_t (&out)[2][4]
#ifdef DG_SM90_FP4_RS_SHARED_LUT
        , const uint64_t* shared_lut
#endif
        ) {
    const uint32_t matrix = lane / 8;
    const uint32_t row = row_base + warp * 16 + (matrix % 2) * 8 + lane % 8;
    const uint32_t offset = row * 64 + (half % 2) * 32 + (matrix / 2) * 16 + (half / 2) * LoadN * 64;
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(packed))
        + (offset ^ ((offset >> 3) & 0x30u));
    uint32_t q[4];
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(q[0]), "=r"(q[1]), "=r"(q[2]), "=r"(q[3]) : "r"(address));
    const uint32_t output_row = row_base + warp * 16 + lane / 4;
    const uint32_t scales[2] = {sf[output_row * (BlockK / 128) + half / 2], sf[(output_row + 8) * (BlockK / 128) + half / 2]};
    #pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
        #pragma unroll
        for (uint32_t r = 0; r < 2; ++r) {
            const uint32_t e = (scales[r] >> (((half % 2) * 2 + j) * 8)) & 0xffu;
#ifdef DG_SM90_FP4_RS_SHARED_LUT
            const uint64_t lut = shared_lut[e];
#else
            const uint64_t lut = fp4_decode_detail::pack_scaled_e4m3_lut_from_e8m0_const(e);
#endif
            fp4x8_rs_interleaved_to_e4m3(
                q[j * 2 + r], static_cast<uint32_t>(lut), static_cast<uint32_t>(lut >> 32),
                out[j][r], out[j][r + 2]);
        }
    }
}

// Keep the current fragment live until its asynchronous WGMMA has completed.
CUTLASS_DEVICE void fence_fp4_rs_fragment(uint32_t (&a)[4]) {
    asm volatile("" : "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3]) :: "memory");
}

} // namespace deep_gemm
