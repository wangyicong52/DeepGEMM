import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class JitFloatLiteralTest(unittest.TestCase):
    def test_cuda_hex_float_literal_is_portable(self) -> None:
        compiler = os.environ.get("CXX") or shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler is unavailable")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source = tmp_path / "test_float_literal.cpp"
            binary = tmp_path / "test_float_literal"
            source.write_text(
                """
#include <cassert>
#include <iostream>
#include <limits>
#include <string>

#include "csrc/jit_kernels/impls/float_literal.hpp"

int main() {
    using deep_gemm::detail::normalize_cuda_hex_float_magnitude;
    using deep_gemm::detail::to_cuda_hex_float_literal;

    assert(normalize_cuda_hex_float_magnitude("1p+0") == "0x1p+0");
    assert(normalize_cuda_hex_float_magnitude("0x1p+0") == "0x1p+0");

    for (const float value : {
             0.0f,
             -0.0f,
             1.0f,
             -1.0f,
             10.0f,
             std::numeric_limits<float>::denorm_min(),
             std::numeric_limits<float>::max(),
         }) {
        const std::string literal = to_cuda_hex_float_literal(value);
        const auto prefix = literal.find("0x");
        assert(prefix == 0 || (prefix == 1 && literal.front() == '-'));
        assert(literal.find("0x", prefix + 2) == std::string::npos);
        assert(literal.back() == 'f');
        std::cout << literal << '\\n';
    }
}
""".lstrip()
            )

            result = subprocess.run(
                [
                    compiler,
                    "-std=c++20",
                    "-I",
                    str(Path(__file__).parents[1]),
                    str(source),
                    "-o",
                    str(binary),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 and (
                "'format' file not found" in result.stderr
                or "no member named 'format'" in result.stderr
            ):
                self.skipTest(f"{compiler} does not provide C++20 std::format")
            self.assertEqual(result.returncode, 0, result.stderr)
            literals = subprocess.run(
                [str(binary)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()

            compile_check = tmp_path / "compile_literals.cpp"
            compile_check.write_text(
                "constexpr float values[] = {\n"
                + "".join(f"    {literal},\n" for literal in literals)
                + "};\n"
            )
            subprocess.run(
                [
                    compiler,
                    "-std=c++20",
                    "-c",
                    str(compile_check),
                    "-o",
                    str(tmp_path / "compile_literals.o"),
                ],
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
