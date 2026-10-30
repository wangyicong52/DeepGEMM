#pragma once

#include <cmath>
#include <format>
#include <string>

namespace deep_gemm::detail {

static std::string normalize_cuda_hex_float_magnitude(std::string magnitude) {
    if (magnitude.compare(0, 2, "0x") != 0)
        magnitude.insert(0, "0x");
    return magnitude;
}

static std::string to_cuda_hex_float_literal(const float& v) {
    auto magnitude = normalize_cuda_hex_float_magnitude(std::format("{:a}", std::abs(v)));
    return std::format("{}{}f", std::signbit(v) ? "-" : "", magnitude);
}

}  // namespace deep_gemm::detail
