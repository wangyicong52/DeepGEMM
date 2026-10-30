// Compare the production RS word decoder with scalar byte selection from the LUT.
// Run on CUDA; include deep_gemm/include and third-party/cutlass/include.
#include <cstdio>
#include <cuda_runtime.h>
#include <deep_gemm/common/fp4_decode_rs.cuh>

__global__ void check_decode(unsigned int* errors) {
    using namespace deep_gemm;
    // Every 16-bit pattern and every scale byte, in both halves. Complement
    // the upper half to catch accidental cross-half selector/sign dependencies.
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int half = id & 0xffffu;
    const unsigned int packed = half | ((half ^ 0xffffu) << 16);
    const auto lut = fp4_decode_detail::pack_scaled_e4m3_lut_from_e8m0_const(id >> 16);
    const auto lut_lo = static_cast<uint32_t>(lut);
    const auto lut_hi = static_cast<uint32_t>(lut >> 32);
    uint32_t rs_packed = packed & 0x77777777u;
    uint32_t expected[2] = {};
    #pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        const uint32_t nibble = (packed >> (4 * i)) & 0xfu;
        // RS puts the first four signs at bits 3,11,19,27 and the
        // remaining four at bits 7,15,23,31.
        const uint32_t sign_position = (i % 4) * 8 + 3 + (i / 4) * 4;
        rs_packed |= ((nibble >> 3) & 1u) << sign_position;
        const uint32_t magnitude = (lut >> ((nibble & 7u) * 8)) & 0xffu;
        const uint32_t byte = magnitude | ((nibble & 8u) << 4);
        expected[i / 4] |= byte << ((i % 4) * 8);
    }
    uint32_t lo, hi;
    fp4x8_rs_interleaved_to_e4m3(rs_packed, lut_lo, lut_hi, lo, hi);
    if (lo != expected[0] || hi != expected[1])
        atomicAdd(errors, 1u);
}

int main() {
    unsigned int *errors, result = 0;
    auto check = [](cudaError_t code) {
        if (code != cudaSuccess) {
            std::fprintf(stderr, "%s\n", cudaGetErrorString(code));
            return false;
        }
        return true;
    };
    if (!check(cudaMalloc(&errors, sizeof(result)))) return 1;
    if (!check(cudaMemset(errors, 0, sizeof(result)))) return 1;
    check_decode<<<(1u << 24) / 256, 256>>>(errors);
    if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) return 1;
    if (!check(cudaMemcpy(&result, errors, sizeof(result), cudaMemcpyDeviceToHost))) return 1;
    if (!check(cudaFree(errors))) return 1;
    std::printf("patterns=16777216 decoded_values=134217728 mismatches=%u\n", result);
    return result != 0;
}
