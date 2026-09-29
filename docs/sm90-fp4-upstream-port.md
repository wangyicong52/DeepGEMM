# SM90 MegaMoE FP4 上游优化移植

## 基线与范围

- 参考本地 `iaas-kernels` HEAD：`3a49c7925cdde7f4c70b7c9cbfe63fd62ed348f7`。
- 修改前 DeepGEMM HEAD：`45a95bac4`，分支 `wyc/vllm-mega-moe`。
- 增量来源：`d16b553`（调度与 token buckets）、`8bf1197`（activation / scatter epilogue）、`3a49c79`（可选 RS 后端）。此前 decode pipeline 更新已在当前分支，未重复覆盖。
- 本轮仅修改 DeepGEMM，不推送；不改参考仓库及原有未跟踪 `third-party/fmt/`。

## 默认 SS 路径更新

1. SwiGLU 的 BM64 / BM128 选择改为 padding + decode 成本比较，保留 BM128 和收益裕量；SiTU 保留原有成本模型。
2. 对符合形状条件、长 K 循环且平均每 expert 为 16–32 tokens 的 SwiGLU 启用 8-token 细分桶，并延伸 swapAB 选择范围。
3. 对符合条件的超稀疏 EP8 布局使用 8 个 decode helper warps，使已有 N64-ready pipeline 生效。
4. 普通 FC1 activation/reduction 优化，以及受 clamp 条件约束的快速 exponent 路径。
5. 普通 L2 提前计算 scatter 指针；BM128 / WG64×128 使用向量化 epilogue。复用现有 metadata 空间，不扩大 workspace；swapAB 保留 metadata fallback。

现有 `DG_SM90_FP4_I2304_SPLIT_N`、`DG_MEGA_MOE_FP4_PIPELINE_OPTIMIZED`、`DG_MEGA_MOE_FP4_PAIRED_PRMT`、`DG_MEGA_MOE_FP4_PACKED_PREFETCH` 诊断开关保留，不随上游清理一起删除。现有 vector-store paired decoder 也保留。

## 可选 RS 后端

RS 使用独立内核、host runtime 和配置。核心 CUDA 内核、RS decoder 与上游在命名空间/宏前缀替换后逐字一致。配置为 BM64、BN256、BK256、3 stages、20 warps；权重在寄存器中解码后交给 WGMMA。

### 自动选择

在 Python 首次导入前设置：

```bash
export DG_SM90_FP4_USE_RS=1
```

不设置或设为 `0` 使用 SS。此选择在 `deep_gemm.mega` 导入时固定，后续修改环境变量不会切换已打包权重对应的执行后端。

原有调用方式保持：

```python
import deep_gemm as dg

# raw_l1 / raw_l2 是原始 checkpoint 的 (packed_fp4_bytes, fp32_scales)。
w1, w2 = dg.transform_weights_for_mega_moe_sm90_fp4(raw_l1, raw_l2)
dg.fp8_fp4_mega_moe(y, w1, w2, sym_buffer, num_sms=0)
```

### 显式选择（适合 SS / RS 对比）

```python
ss_w1, ss_w2 = dg.transform_weights_for_mega_moe_sm90_fp4_ss(raw_l1, raw_l2)
rs_w1, rs_w2 = dg.transform_weights_for_mega_moe_sm90_fp4_rs(raw_l1, raw_l2)

dg.fp8_fp4_mega_moe_ss(y_ss, ss_w1, ss_w2, sym_buffer)
dg.fp8_fp4_mega_moe_rs(y_rs, rs_w1, rs_w2, sym_buffer)
```

**SS 和 RS 权重形状相同，但 byte/sign 布局不同，不能混用，也不能把已转换的 SS 权重再当原始 checkpoint 传入 RS packer。切换后端时应重启进程并重新打包原始权重。** 显式入口允许在同一进程持有两套对应权重；原始张量无法仅凭 shape 判断其所属布局，调用方仍需正确配对。

### 兼容边界

- RS 仅支持 SM90、单节点/NVLink 通信路径；不接入 RDMA。
- 激活 scale groups 固定为 `(128, 64)`，recipe 为 `(1, 1, 32)`。
- `hidden % 512 == 0`，`intermediate_hidden % 256 == 0`，且 `intermediate_hidden <= 4096`；输出要求连续 CUDA BF16 矩阵，宽度匹配 hidden。
- 支持 SwiGLU 和 SiTU。SiTU 不支持 `activation_clamp`；沿用本分支参数名 `activation_alpha` / `activation_beta`，须显式传入正值，例如 `4.0 / 25.0`。本轮使 SM90 的 symmetric buffer 构造接受已有 native 支持的 SiTU，非 SM90 保持原限制。
- 保留 `num_sms` 参数；非零值须大于 1、不超过当前 runtime SM 数并且为偶数。
- 不支持 shared expert weights；此类参数现在在 SM90 Python 入口明确报错，而非忽略。
- 通用 `fp8_fp4_mega_moe` 只在 SM90 上按 selector 路由；非 SM90、FP8、BF16 路径不改。SM100 仍须使用原有 `transform_weights_for_mega_moe`。
- 未引入 iaas-kernels 独有的 group32 API 或 FP8 调优，避免扩展到本次 FP4 范围之外。

## 本地验证

在隔离的 CPU PyTorch 环境执行：

```bash
python -m unittest discover -s tests -p test_sm90_fp4_port.py -v
python -m unittest discover -s tests -p test_jit_float_literal.py -v
```

`test_sm90_fp4_port.py` 共 7 项：

- 真实 Python 模块的 backend / packer 配对、导入后环境变更、显式 SS 入口、非 SM90 fallback 和 `num_sms` 转发。
- 非法 selector 值，以及 shared experts、scale groups、activation 参数拒绝。
- CPU PyTorch 真实权重打包，逆变换验证 byte/sign 无损及 scale 不变。
- 提取实际 C++ host heuristics 和 JIT emitters 编译并运行；300 组参数分别检查 SS/RS 配置、共享内存公式与宏选择。
- 按生产源码地址表达式枚举 BM128 向量化 scatter，对照独立 WGMMA accumulator 坐标，覆盖一个 WG 的 8192 个 BF16 位置，检查无重叠、无越界、无遗漏。
- RS native guards / 导出符号检查。

另外运行 Python AST 检查、项目 include 存在性检查、`git diff --check`；核对 RS 内核/decoder、SS activation/scatter 片段及旧 scheduler/workspace 与上游一致性。

**这些是 CPU 与 host 层验证，不等价于 CUDA 编译、ptxas/SASS 检查、native extension 构建、GPU 数值、并发同步或性能测试。C++ 配置测试替代了 runtime/layout 外部依赖，不能据此宣称完整 CUDA 构建通过。** 本机无 nvcc 或 NVIDIA GPU；按约定 GPU 测试由用户后续执行，当前不报告提速比例。

## GPU 后续验证入口（尚未运行）

先在目标 Linux / SM90 环境重新构建当前源码的 C++ 扩展，确保新的 RS native 符号及 CUDA headers 进入安装包。`setup.py` 的现有 package_data 通配规则覆盖新增 headers。

### Decoder 位模式回归

```bash
nvcc -std=c++20 -arch=sm_90a \
  -I deep_gemm/include -I third-party/cutlass/include \
  tests/test_sm90_fp4_rs_decode.cu -o /tmp/deepgemm_sm90_rs_decode
/tmp/deepgemm_sm90_rs_decode
```

预期 `mismatches=0`；测试枚举 16,777,216 组 pattern/scale 组合。

### SS / RS 数值与 graph 重放

```bash
# 小规模单卡；内部显式调用 SS/RS，各用对应 packer。
torchrun --standalone --nproc_per_node=1 tests/test_sm90_fp4_backends.py

# EP8 / 超稀疏、边界、较密集 tokens。
torchrun --standalone --nproc_per_node=8 tests/test_sm90_fp4_backends.py \
  --experts 384 --topk 6 --hidden 5120 --intermediate 2304 \
  --batches 1,2,3,48,128,160,192,256,257,640,768,1024 --repeats 16

# 分别覆盖强偏斜、全部无效路由、SiTU。
torchrun --standalone --nproc_per_node=8 tests/test_sm90_fp4_backends.py --routing skewed
torchrun --standalone --nproc_per_node=8 tests/test_sm90_fp4_backends.py --routing invalid
torchrun --standalone --nproc_per_node=8 tests/test_sm90_fp4_backends.py --activation situ
```

该入口对每个后端做 FP32 reference 检查、eager bitwise repeat、SS/RS 共用 workspace 的 CUDA Graph 交替重放。reference 包含激活 FP8 重量化及 BF16 scatter；默认 relative RMSE 阈值为 0.05，是显式测试容差，不是已测误差。它不要求 SS 与 RS 相互 bitwise 相等，不是 full-model 精度验证，也不是性能 benchmark。

性能回归应使用目标模型的真实形状、路由分布、GPU 和固定输入，分别记录旧 SS、新 SS、RS 的 kernel-only latency，排除 JIT 和权重打包耗时。上游说明过平均路由密度策略对部分强偏斜 routes 仍可能回退；不能用均匀随机路由的通过结果代替实际服务场景验证。
