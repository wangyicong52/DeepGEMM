"""CPU regressions for the SM90 FP4 port (no built CUDA extension required).

Runs the real Python module against a recording native shim, and compiles the
actual host heuristics/JIT emitters against minimal runtime/layout stand-ins.
This does NOT compile device code or validate GPU numerical behavior.
"""
import importlib.util
import inspect
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_mega(setting='0'):
    """Load production mega code without importing the CUDA extension."""
    pkg = types.ModuleType('_sm90_port_test')
    pkg.__path__ = [str(ROOT / 'deep_gemm')]
    pkg._C = types.SimpleNamespace()
    utils = types.ModuleType('_sm90_port_test.utils')
    utils.__path__ = []
    math = types.ModuleType('_sm90_port_test.utils.math')
    math.align = lambda value, multiple: (value + multiple - 1) // multiple * multiple
    spec = importlib.util.spec_from_file_location(
        '_sm90_port_test.mega', ROOT / 'deep_gemm/mega/__init__.py')
    mod = importlib.util.module_from_spec(spec)
    modules = {pkg.__name__: pkg, utils.__name__: utils, math.__name__: math,
               spec.name: mod}
    previous = {name: sys.modules.get(name) for name in modules}
    # Restore only our shim entries. Restoring all of sys.modules would unload
    # freshly imported PyTorch extension modules and can crash the interpreter.
    sys.modules.update(modules)
    try:
        with patch.dict(os.environ, {'DG_SM90_FP4_USE_RS': setting}):
            spec.loader.exec_module(mod)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return mod


class PythonBackendTest(unittest.TestCase):
    def test_selector_packer_and_native_call_stay_paired(self):
        import torch
        q = torch.arange(2 * 64 * 128).to(torch.uint8).reshape(2, 64, 128)
        sf = torch.ones((2, 64, 8), dtype=torch.float32)
        for setting in ('0', '1'):
            with self.subTest(setting=setting):
                m = load_mega(setting)
                m._is_sm90 = lambda: True
                calls = []
                m._C = types.SimpleNamespace(
                    fp8_fp4_mega_moe_sm90=lambda *args: calls.append(('ss', args)),
                    fp8_fp4_mega_moe_sm90_rs=lambda *args: calls.append(('rs', args)),
                    fp8_fp4_mega_moe=lambda *args: calls.append(('sm100', args)),
                    fp8_mega_moe=lambda *args: calls.append(('fp8', args)),
                )
                buf = types.SimpleNamespace(
                    buffer=None, handle=types.SimpleNamespace(buffer_ptrs=[0]),
                    group=types.SimpleNamespace(rank=lambda: 0),
                    num_max_tokens_per_rank=1920, num_experts=8, num_topk=2,
                    mma_type='fp8xfp4', hidden=256,
                    x_sf=torch.empty((1920, 2), dtype=torch.float32),
                    activation='swiglu', activation_group_sizes=(128, 64))
                packer = (m.transform_weights_for_mega_moe_sm90_fp4_rs if setting == '1'
                          else m.transform_weights_for_mega_moe_sm90_fp4_ss)
                self.assertIs(m.transform_weights_for_mega_moe_sm90_fp4, packer)
                with patch.dict(os.environ, {'DG_SM90_FP4_USE_RS': '1' if setting == '0' else '0'}):
                    weights = m.transform_weights_for_mega_moe_sm90_fp4((q, sf), (q, sf))
                    m.fp8_fp4_mega_moe(None, *weights, buf, num_sms=40)
                self.assertEqual(calls[-1][0], 'rs' if setting == '1' else 'ss')
                self.assertEqual(calls[-1][1][-1], 40)
                m.fp8_fp4_mega_moe_ss(None, *weights, buf)
                self.assertEqual(calls[-1][0], 'ss')
                m.fp8_mega_moe(None, *weights, buf)
                self.assertEqual(calls[-1][0], 'fp8')
                m._is_sm90 = lambda: False
                m.fp8_fp4_mega_moe(None, *weights, buf)
                self.assertEqual(calls[-1][0], 'sm100')
                with self.assertRaisesRegex(ValueError, 'only supported on SM90'):
                    m.fp8_fp4_mega_moe_rs(None, *weights, buf)
                self.assertEqual(inspect.signature(m.fp8_fp4_mega_moe),
                                 inspect.signature(m.fp8_fp4_mega_moe_ss))
                self.assertEqual(inspect.signature(m.fp8_fp4_mega_moe),
                                 inspect.signature(m.fp8_fp4_mega_moe_rs))

    def test_invalid_selector_rejected(self):
        for value in ('', 'true', '2', '-1'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                load_mega(value)

    def test_lossless_rs_layout_and_scales(self):
        import torch
        m = load_mega()
        # All byte values, negative signs, non-contiguous input, and varied SFs.
        q = torch.arange(2 * 64 * 128).to(torch.uint8).reshape(2, 64, 128)
        sf = torch.pow(2.0, torch.arange(2 * 64 * 8) % 24 - 16).reshape(2, 64, 8)
        raw = ((q, sf), (q.transpose(1, 2).contiguous().transpose(1, 2), sf.flip(-1)))
        rs = m.transform_weights_for_mega_moe_sm90_fp4_rs(*raw)
        ss = m.transform_weights_for_mega_moe_sm90_fp4_ss(*raw)
        perm = torch.tensor([0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15])
        def interleave(t):
            e, n, k = t.shape
            return t.reshape(e, 2, n // 16, 8, k).transpose(1, 2).contiguous().reshape_as(t)
        reference = ((interleave(q), interleave(sf)), raw[1])
        for (original, original_sf), (packed, packed_sf), ss_pair in zip(reference, rs, ss):
            words = packed.view(torch.int32)
            restored = words & 0x77777777
            for old, new in enumerate((0, 2, 4, 6, 1, 3, 5, 7)):
                restored |= ((words >> (new * 4)) & 8) << (old * 4)
            restored = restored.view(torch.int8).reshape(*packed.shape[:-1], -1, 16)
            restored = restored.index_select(-1, perm.argsort()).reshape_as(original)
            self.assertTrue(torch.equal(original.contiguous().view(torch.int8), restored))
            expected_sf = ((original_sf.contiguous().view(torch.int32) >> 23) & 255).to(torch.uint8)
            self.assertTrue(torch.equal(expected_sf, packed_sf.view(torch.uint8)))
            self.assertTrue(torch.equal(packed_sf, ss_pair[1]))
            self.assertTrue(packed.is_contiguous())
        self.assertFalse(torch.equal(rs[0][0], ss[0][0]))

    def test_invalid_options_rejected_before_native(self):
        m = load_mega()
        m._is_sm90 = lambda: True
        for op in (m.fp8_fp4_mega_moe_ss, m.fp8_fp4_mega_moe_rs):
            for buf, kwargs, error in (
                (types.SimpleNamespace(activation_group_sizes=(32, 32)), {}, 'activation groups'),
                (types.SimpleNamespace(), {'shared_l1_weights': object()}, 'shared expert'),
                (types.SimpleNamespace(), {'activation': 'situ', 'activation_clamp': 10}, 'SiTU'),
            ):
                with self.subTest(op=op.__name__, error=error), self.assertRaisesRegex(ValueError, error):
                    op(None, None, None, buf, **kwargs)

    def test_legacy_default_buffer_can_run_situ(self):
        import torch
        m = load_mega()
        m._is_sm90 = lambda: True
        calls = []
        m._C = types.SimpleNamespace(
            fp8_fp4_mega_moe_sm90=lambda *args: calls.append('ss'),
            fp8_fp4_mega_moe_sm90_rs=lambda *args: calls.append('rs'),
        )
        buf = types.SimpleNamespace(
            activation='swiglu', activation_group_sizes=(128, 64),
            buffer=None, handle=types.SimpleNamespace(buffer_ptrs=[0]),
            group=types.SimpleNamespace(rank=lambda: 0),
            num_max_tokens_per_rank=1920, num_experts=8, num_topk=2,
            mma_type='fp8xfp4', hidden=256,
            x_sf=torch.empty((1920, 2), dtype=torch.float32),
        )
        kwargs = dict(activation='situ', activation_alpha=4., activation_beta=25.)
        m.fp8_fp4_mega_moe_ss(None, None, None, buf, **kwargs)
        m.fp8_fp4_mega_moe_rs(None, None, None, buf, **kwargs)
        self.assertEqual(calls, ['ss', 'rs'])


class HostConfigTest(unittest.TestCase):
    def test_actual_heuristics_and_jit_emitters(self):
        compiler = os.environ.get('CXX') or shutil.which('c++')
        if not compiler:
            self.skipTest('C++20 compiler unavailable')
        def body(path):
            source = (ROOT / path).read_text()
            return source[source.index('namespace deep_gemm {'):]
        config = body('csrc/jit_kernels/heuristics/sm90_mega_moe.hpp')
        rs_config = body('csrc/jit_kernels/heuristics/sm90_fp4_rs_mega_moe.hpp')
        api = (ROOT / 'csrc/apis/sm90_mega.hpp').read_text()
        defaults = api[api.index('struct FP4SM90APIDefaults'):api.index('static std::tuple<int64_t,')]
        emitters = []
        for name in ('sm90_fp8_fp4_mega_moe', 'sm90_fp8_fp4_mega_moe_rs'):
            src = body(f'csrc/jit_kernels/impls/{name}.hpp')
            emitters.append(src[:src.index('    static void launch(')] + '};\n}\n')
        prelude = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include "csrc/jit_kernels/impls/float_literal.hpp"
#define DG_HOST_ASSERT(x) assert(x)
using nv_bfloat16 = uint16_t;
using CUtensorMap = int;
namespace deep_jit {
template<class T> T get_env(const char* key, T d = 0) {
    auto s = std::getenv(key); return s ? static_cast<T>(std::atoi(s)) : d;
}
namespace cuda { struct Dim { int x; }; struct LaunchOptions { std::optional<Dim> grid_dim; }; }
}
namespace deep_gemm {
struct SM90ArchSpec { static constexpr int smem_capacity = 232448; };
struct Runtime { int get_num_sms() { return 78; } } runtime_obj;
auto runtime = &runtime_obj;
namespace math {
template<class T> T align(T a,T b) { return (a+b-1)/b*b; }
}
using math::align;
template<class T> T ceil_div(T a,T b) { return (a+b-1)/b; }
std::string to_string(float x) { return detail::to_cuda_hex_float_literal(x); }
namespace layout {
constexpr int kCandidateBlockM[] = {8,16,32,64,96,128,192,240};
constexpr int kNumCandidateBlockMs = 8;
struct Data { int n; explicit Data(int n):n(n) {} };
struct Buffer { int n; Buffer(Data d,int r,int t):n(d.n*r*t){} int get_num_bytes(){return n;} };
int get_num_max_pool_tokens(int r,int t,int k,int e) {
    return align(r*t*std::min(k,e)+e*239,1920);
}
template<int N=0> struct SymBuffer {};
}
}
'''
        main = r'''
int main() {
    using namespace deep_gemm;
    unsetenv("DG_SM90_FP4_I2304_SPLIT_N");
    unsetenv("DG_MEGA_MOE_FP4_PIPELINE_OPTIMIZED");
    unsetenv("DG_MEGA_MOE_FP4_PACKED_PREFETCH");
    unsetenv("DG_MEGA_MOE_FP4_PAIRED_PRMT");
    // Fine-bucket bounds, short-K guard, SiTU exclusion, and V4.1 kill-switch.
    assert(!use_sm90_fp4_fine_token_buckets(5120,2304,15.999f,false));
    assert(use_sm90_fp4_fine_token_buckets(5120,2304,16.f,false));
    assert(use_sm90_fp4_fine_token_buckets(5120,2304,32.f,false));
    assert(!use_sm90_fp4_fine_token_buckets(5120,2304,32.001f,false));
    assert(!use_sm90_fp4_fine_token_buckets(1024,2304,24.f,false));
    assert(!use_sm90_fp4_fine_token_buckets(5120,2304,24.f,true));
    assert(get_fp4_sm90_api_defaults(48,256,6,false,8,5120,2304).swap_ab);
    assert(!get_fp4_sm90_api_defaults(48,257,6,false,8,5120,2304).swap_ab);
    setenv("DG_SM90_FP4_I2304_SPLIT_N","0",1);
    assert(!get_fp4_sm90_api_defaults(48,256,6,false,8,5120,2304).swap_ab);
    unsetenv("DG_SM90_FP4_I2304_SPLIT_N");
    bool saw64=false,saw128=false;
    for (int t=640;t<=8192;t+=16) {
        bool p=is_fp4_sm90_prefill_band(float(t)/8,false);
        saw64 |= !p; saw128 |= p;
    }
    assert(saw64 && saw128);
    int cases=0;
    for (int r : {1,8}) for (bool situ : {false,true})
    for (int ih : {1024,2048,2304,3072,4096})
    for (int t : {0,1,2,3,48,128,159,160,192,256,257,640,768,1024,8192}) {
        int e=384, er=e/r, h=5120;
        auto f=get_fp4_sm90_api_defaults(er,t,6,situ,r,h,ih);
        auto c=get_mega_moe_config_sm90_fp4(r,e,er,9600,t,6,h,ih,1000000,
                    situ,40,f.early_b_decode,f.decode_done_mbarrier,f.swap_ab);
        assert(c.smem_size<=SM90ArchSpec::smem_capacity && c.num_stages>=2);
        assert(c.block_m==(f.early_b_decode?128:64));
        assert(f.early_b_decode==f.ss_nsplit);
        assert((c.num_dispatch_threads+c.num_non_epilogue_threads)%128==0);
        // Independent shared-memory formula for fixed activation groups 128/64.
        bool split=c.block_m==64 && c.num_epilogue_threads==256;
        int cd=align(std::max(c.block_m*128*2,f.swap_ab?c.block_m*64*5:0),1024);
        bool ready=use_sm90_fp4_split_decode_pipeline(c.block_m,128,c.num_epilogue_threads/128,
                                c.num_non_epilogue_threads/32,f.swap_ab,f.decode_done_mbarrier);
        int fixed=align(e*4,1024)+align(h*(c.num_dispatch_threads/32),1024)+cd+
                    (split?1024:0)+(c.num_dispatch_threads/32+2*(c.num_epilogue_threads/32))*8;
        int stage=c.block_m*128+128*128+128*64+align(2*c.block_m*4,128)+512+
                    16+(f.early_b_decode?8:0)+(f.decode_done_mbarrier?(ready?16:8):0);
        assert(c.smem_size==fixed+c.num_stages*stage);
        SM90FP8FP4MegaMoERuntime::Args a{};
        a.hidden=h; a.intermediate_hidden=ih; a.num_experts=e; a.num_ranks=r;
        a.num_topk=6; a.num_tokens=t; a.num_max_tokens_per_rank=9600;
        a.use_situ=situ; a.situ_beta=4; a.situ_linear_beta=25; a.activation_clamp=10;
        a.fast_math=true; a.config=c; a.use_swap_ab=f.swap_ab;
        a.use_wide_load_decode=f.wide_load_decode; a.use_early_b_decode=f.early_b_decode;
        a.use_decode_done_mbarrier=f.decode_done_mbarrier; a.use_ss_nsplit=f.ss_nsplit;
        a.launch_args.grid_dim=deep_jit::cuda::Dim{40};
        auto source=SM90FP8FP4MegaMoERuntime::generate(a);
        bool fine=f.swap_ab && c.block_m==64 && c.num_epilogue_threads==256 &&
                    use_sm90_fp4_fine_token_buckets(h,ih,float(t)*r*6/e,situ);
        assert((source.find("#define DG_MEGA_MOE_FP4_FINE_TOKEN_BUCKETS 1")!=std::string::npos)==fine);
        assert((source.find("#define DG_MEGA_MOE_FP4_N64_DECODE_READY 1")!=std::string::npos)==ready);
        assert(source.find("0x0x")==std::string::npos);
        auto rc=get_mega_moe_config_sm90_fp4_rs(r,e,er,9600,t,6,h,ih,1000000,40);
        assert(rc.block_m==64 && rc.block_n==256 && rc.block_k==256);
        assert(rc.num_stages==3 && rc.num_dispatch_threads==64 &&
               rc.num_non_epilogue_threads==64 && rc.num_epilogue_threads==512);
        int rs_smem=align(e*4,1024)+align(h*2,1024)+40960+1024+272+
                    3*(16384+32768+1024+2048+16);
        assert(rc.smem_size==rs_smem && rs_smem<=232448);
        SM90FP8FP4MegaMoERSRuntime::Args b{};
        b.hidden=h;b.intermediate_hidden=ih;b.num_experts=e;b.num_ranks=r;b.num_topk=6;
        b.num_tokens=t;b.config=rc;b.use_situ=situ;b.l1_act_group=128;b.l2_act_group=64;
        b.launch_args.grid_dim=deep_jit::cuda::Dim{40};
        auto rs_source=SM90FP8FP4MegaMoERSRuntime::generate(b);
        float density=float(t)*r*6/e;
        assert((rs_source.find("#define DG_SM90_FP4_RS_DENSE 1")!=std::string::npos)==
                    (density>(situ?24.f:20.f)));
        assert((rs_source.find("#define DG_SM90_FP4_RS_SHARED_LUT 1")!=std::string::npos)==
                    (density<=8.f && rc.smem_size+4096<=232448));
        ++cases;
    }
    std::cout << "host_config_jit_cases=" << cases << '\n';
}
'''
        source = prelude + config + rs_config + '\nnamespace deep_gemm {\n' + defaults + '\n}\n' + ''.join(emitters) + main
        with tempfile.TemporaryDirectory(prefix='sm90-host-test-') as td:
            cpp, exe = Path(td)/'test.cpp', Path(td)/'test'
            cpp.write_text(source)
            result = subprocess.run([compiler, '-std=c++20', '-I', str(ROOT), str(cpp), '-o', str(exe)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('host_config_jit_cases=300', result.stdout)

    def test_bm128_vector_scatter_mapping(self):
        # Emulate the production address expressions and compare against the
        # independent WGMMA accumulator (row, col) ownership, not a second copy
        # of the optimized scatter mapping.
        kernel = (ROOT/'deep_gemm/include/deep_gemm/impls/sm90_fp8_fp4_mega_moe.cuh').read_text()
        fragment = kernel[kernel.index('const uint32_t store_base ='):]
        def expr(name):
            text = re.search(r'const uint32_t ' + name + r'\s*=\s*(.*?);', fragment, re.S).group(1)
            text = re.sub(r'(\d+)u\b', r'\1', text)
            text = re.sub(r'\(\(tid & (\d+)\) \? (\d+) : 0\)',
                          r'(\2 if (tid & \1) else 0)', text)
            return compile(text.replace('\n', ' '), '<kernel-address>', 'eval')
        expressions = {key: expr(key) for key in ('store_base', 'load_base', 'row_start', 'col_start')}
        def at(key, tid):
            return eval(expressions[key], {'__builtins__': {}}, {'tid': tid})
        scratch = {}
        for tid in range(128):
            lane, warp = tid % 32, tid // 32
            for g in range(4):
                for h in range(2):
                    i = g * 4 + h * 16
                    indices = (i, i+1, i+2, i+3, i+32, i+33, i+34, i+35)
                    base = (at('store_base', tid) ^ (g * 32)) + h * 2048
                    self.assertEqual(base % 16, 0)
                    for j, reg in enumerate(indices):
                        row = warp * 16 + lane // 4 + (reg % 4 // 2) * 8
                        col = (reg // 4) * 8 + (lane % 4) * 2 + reg % 2
                        addr = base + j * 2
                        self.assertNotIn(addr, scratch)
                        self.assertLess(addr, 64 * 128 * 2)
                        scratch[addr] = (row, col)
        self.assertEqual(len(scratch), 64 * 128)
        seen = set()
        for tid in range(128):
            load = at('load_base', tid)
            row = at('row_start', tid)
            col = at('col_start', tid)
            for g in range(4):
                a = [scratch[load + g*512 + j*2] for j in range(8)]
                b = [scratch[(load ^ 8256) + g*512 + j*2] for j in range(8)]
                for r, values in ((row+g*16, a[:2]+b[:2]+a[4:6]+b[4:6]),
                                  (row+g*16+8, a[2:4]+b[2:4]+a[6:8]+b[6:8])):
                    expected = [(r, c) for c in (*range(col,col+4), *range(col+64,col+68))]
                    self.assertEqual(values, expected)
                    for cell in values:
                        self.assertNotIn(cell, seen)
                        seen.add(cell)
        self.assertEqual(len(seen), 8192)

    def test_rs_kernel_import_and_host_guards(self):
        kernel = (ROOT/'deep_gemm/include/deep_gemm/impls/sm90_fp8_fp4_mega_moe_rs.cuh').read_text()
        self.assertIn('sched::LegacyMegaMoEScheduler<', kernel)
        self.assertIn('layout::SM90Workspace', kernel)
        self.assertNotIn('iaas_kernels', kernel)
        host = (ROOT/'csrc/apis/sm90_mega_rs.hpp').read_text()
        for guard in ('y.is_cuda()', 'y.is_contiguous()', 'y.size(1) == hidden',
                      'hidden % 512 == 0', 'intermediate_hidden % 256 == 0',
                      'num_sms_override % 2 == 0'):
            self.assertIn(guard, host)


if __name__ == '__main__':
    unittest.main()
