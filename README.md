# tilelang-riscv

`tilelang-riscv` 是一个 TileLang 分支，目标是在不影响现有 CUDA、HIP、Metal backend 的前提下，为 `target="riscv"` / `target="linalg_riscv"` 提供一条面向 CPU / RISC-V 的 structured MLIR lowering 路径。

当前主线是：

```text
TileLang DSL
  -> TVM TIR
  -> structured MLIR (func/arith/scf/memref/linalg/...)
  -> LLVM
  -> RISC-V
```

这个仓库当前应当按 `examples/riscv/` 和 `testing/python/riscv/` 来理解能力边界。原始 upstream `examples/` 目录仍然保留，但它们不是 `linalg_riscv` backend 的完成度声明。

## 当前状态

更新时间：`2026-03-31`

- Phase 0：完成
- Phase 1：完成
- Phase 2：部分完成
- Phase 3：部分完成
- Phase 4：部分完成
- Phase 5：部分完成
- Phase 6：未完成

更具体地说：

- `riscv` 已归一化到 `linalg_riscv`
- `tilelang/engine/phase.py` 和 `tilelang/engine/lower.py` 已接入独立的 structured RISC-V lowering 路径
- `src/target/codegen_linalg_riscv.cc` 已使用真实 MLIR C++ API 构造 `mlir::ModuleOp`
- `src/target/rt_mod_linalg_riscv.cc` 已返回专用 `mlir` source runtime module
- vendored LLVM/MLIR toolchain 已有固定发现逻辑与构建脚本
- `tilelang.compile(..., target="riscv")` 已接到 `RiscvKernelAdapter`
- `.mlir` / `.ll` / `.s` / `.o` artifact 导出已可用
- 本地 x86 host 路径的共享库构建、加载和 NumPy / Torch 侧执行已可用
- qemu/spike runner 路径已接线，但是否能跑依赖机器环境

## 当前已完成能力

当前已经打通的 lowering / runtime 子集包括：

- `PrimFunc` buffer / scalar 参数到 `func.func` 参数
- `BlockRealize` / `Block`
- `For -> scf.for`
- `IfThenElse -> scf.if`
- `AllocBuffer` / `BufferRealize` / `DeclBuffer -> memref.alloca`
- `BufferLoad` / `BufferStore -> memref.load/store`
- 常量、cast、算术、比较、`Select`
- unit `thread_extent` / `T.Kernel(..., threads=1)` shell
- static compact row-major buffer parameters with explicit strides
- simple contiguous `match_buffer -> memref.subview`
- `tl.copy -> memref.copy` 或 `scf + memref.load/store` fallback
- `tl.clear` / fill 的 `scf + memref.store` fallback
- 简单 elementwise -> `linalg.generic`
- 简单 full-shape `sum/min/max` reduction 的 structured lowering 与 fallback
- `tl.gemm -> linalg.matmul`
- `tl.gemm(..., transpose_A=True)`
- `tl.gemm(..., transpose_B=True)`
- 双转置场景通过临时 transpose materialization 接到 matmul 路径
- rank-reduced static-1 subview lowering
- dynamic-shape buffer params 重新绑定
- host `.so` build/load
- `emit_mlir()` / `emit_llvm_ir()` / `emit_asm()` / `emit_object()`
- `tilelang.compile(..., target="riscv")`
- cache serialization / reload

当前默认 debug pipeline 为：

```text
canonicalize
cse
func.func(convert-linalg-to-loops)
canonicalize
cse
convert-scf-to-cf
expand-strided-metadata
lower-affine
finalize-memref-to-llvm
convert-math-to-llvm
convert-arith-to-llvm
convert-func-to-llvm
convert-cf-to-llvm
reconcile-unrealized-casts
```

## 当前支持的例子

下面这些例子位于 `examples/riscv/`，并由 `testing/python/riscv/test_riscv_examples.py` 覆盖：

- `example_vector_add.py`
- `example_copy.py`
- `example_reduce_sum.py`
- `example_reduce_max.py`
- `example_matmul.py`
- `example_gemv.py`
- `example_batched_gemm.py`
- `example_dynamic_batched_gemm.py`
- `example_grouped_gemm.py`
- `example_dynamic_grouped_gemm.py`
- `example_dynamic_shape.py`
- `example_rms_norm.py`
- `example_online_softmax.py`
- `example_topk.py`
- `example_convolution.py`

可以按功能大致理解为：

- 基础 elementwise / copy / reduction：`vector_add`、`copy`、`reduce_sum`、`reduce_max`
- 线性代数：`matmul`、`gemv`、`batched_gemm`、`dynamic_batched_gemm`、`grouped_gemm`、`dynamic_grouped_gemm`、`dynamic_shape`
- 算子型例子：`rms_norm`、`online_softmax`、`topk`、`convolution`

## 运行前准备

运行这些例子前，你需要：

- 一个能正常导入 `tilelang` 和 `tvm_ffi` 的 Python 环境
- 已构建的 TileLang 本地产物，至少包含 `build/lib`
- 可用的 LLVM/MLIR toolchain

当前仓库默认优先从这些位置找 LLVM/MLIR：

- `TILELANG_RISCV_LLVM_ROOT`
- `TILELANG_LLVM_INSTALL_DIR`
- `3rdparty/llvm-project/install`
- `3rdparty/llvm-project/build-host/install`
- `3rdparty/llvm-project/build/install`

如果还没构建 vendored LLVM/MLIR，可以先执行：

```bash
./maint/scripts/build_llvm_mlir.sh
```

如果你使用本机开发环境，最短准备方式通常是：

```bash
cd /path/to/tilelang-riscv
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tilelang-riscv
export REPO_ROOT="$(pwd)"
export LD_LIBRARY_PATH="$REPO_ROOT/build/lib:$REPO_ROOT/3rdparty/llvm-project/install/lib:${LD_LIBRARY_PATH}"
```

本文后面的命令都默认在仓库根目录执行，并统一使用 `python -m examples.riscv...` 的模块方式，这样通常不需要额外设置 `PYTHONPATH`。

## 如何运行例子

所有 `examples/riscv/*.py` 都共用同一套 CLI，支持：

- `--print-tir`
- `--emit-mlir`
- `--emit-llvm`
- `--emit-asm`
- `--emit-object`
- `--run-host`
- `--run-qemu`
- `--output-dir`

如果不传任何动作参数，默认行为等价于 `--run-host`。

最简单的 smoke test：

```bash
python -m examples.riscv.example_vector_add
python -m examples.riscv.example_matmul
python -m examples.riscv.example_convolution
```

更明确一点：

```bash
python -m examples.riscv.example_vector_add --run-host
python -m examples.riscv.example_matmul --run-host
python -m examples.riscv.example_dynamic_shape --run-host
```

## 如何查看生成的 TIR / MLIR / LLVM IR / 汇编

查看源 TIR：

```bash
python -m examples.riscv.example_vector_add --print-tir
```

直接把 MLIR 打到终端：

```bash
python -m examples.riscv.example_vector_add --emit-mlir
```

直接把 LLVM IR 打到终端：

```bash
python -m examples.riscv.example_vector_add --emit-llvm
```

直接把 RISC-V 汇编打到终端：

```bash
python -m examples.riscv.example_vector_add --emit-asm
```

如果你想把这些产物落盘，再自己慢慢看：

```bash
python -m examples.riscv.example_vector_add \
  --emit-mlir \
  --emit-llvm \
  --emit-asm \
  --emit-object \
  --output-dir /tmp/tilelang-riscv-out
```

这会生成：

- `/tmp/tilelang-riscv-out/vector_add.mlir`
- `/tmp/tilelang-riscv-out/vector_add.ll`
- `/tmp/tilelang-riscv-out/vector_add.s`
- `/tmp/tilelang-riscv-out/vector_add.o`

例如查看导出的 MLIR：

```bash
sed -n '1,120p' /tmp/tilelang-riscv-out/vector_add.mlir
```

其他例子也一样，只是文件名前缀会换成各自的 kernel 名，例如：

- `tile_matmul.mlir`
- `dynamic_matmul.mlir`
- `topk.mlir`

## Python API 查看生成源码

除了例子 CLI，也可以直接用 Python API。

### 方式 1：直接用 `tilelang.compile(..., target="riscv")`

```python
import tilelang
import tilelang.language as T


@T.prim_func
def tile_copy(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    for i in T.serial(4):
        with T.block("copy"):
            vi = T.axis.spatial(4, i)
            B[vi] = A[vi]


kernel = tilelang.compile(tile_copy, out_idx=[1], target="riscv")
print(kernel.get_kernel_source())  # 当前返回 MLIR 文本
kernel.close()
```

### 方式 2：使用 artifact export helper

```python
from tilelang.jit.adapter.riscv import emit_asm, emit_llvm_ir, emit_mlir, emit_object

mlir_text = emit_mlir(rt_mod)
llvm_ir = emit_llvm_ir(rt_mod)
asm_text = emit_asm(rt_mod)
obj_bytes = emit_object(rt_mod)
```

这里的 `rt_mod` 可以来自：

- `examples/riscv/common.py` 里的 `build_riscv_module(...)`
- 或者你自己通过 `target.build.tilelang_linalg_riscv` 构造出的 runtime module

## QEMU / Spike 运行方式

当前例子也支持：

```bash
python -m examples.riscv.example_vector_add --run-qemu --output-dir /tmp/vector_add-qemu
```

但这要求当前机器满足以下至少一项：

- `qemu-riscv64` 在 `PATH` 中
- 或者设置 `TILELANG_RISCV_RUNNER="spike pk"`

相关环境变量包括：

- `TILELANG_RISCV_RUNNER`
- `TILELANG_RISCV_RUNNER_FLAGS`
- `TILELANG_RISCV_LINKER`
- `TILELANG_RISCV_TRIPLE`
- `TILELANG_RISCV_MARCH`
- `TILELANG_RISCV_ABI`
- `TILELANG_RISCV_CPU`
- `TILELANG_RISCV_CLANG_FLAGS`

注意：qemu / spike 路径目前是可选验证面，不是所有机器都默认具备。

## 测试

建议至少跑下面两组：

```bash
pytest testing/python/riscv -q
pytest testing/python/riscv/test_riscv_examples.py -q
```

这些测试需要：

- Python 环境里 `tvm_ffi` 可导入
- `build/lib` 已准备好
- LLVM/MLIR toolchain 可被发现

## 当前明确未完成的部分

下面这些仍然是当前缺口，不应在 README 里被误表述成“已经完成”：

- 真正的 in-process native MLIR Python binding 还没有，当前 `tilelang.tladapter._native` 仍是工具驱动兼容层
- 更广泛的 region / subview / slice lowering 还没有全面覆盖
- 更广泛的 irregular broadcast / predication / generic elementwise 场景还没有全面覆盖
- reduction 识别目前仍主要是简单 full-shape 模式
- batched / mixed-shape `tl.gemm` 还没有完全泛化
- qemu / spike 验证依赖环境，不是每台机器都现成可用
- RVV-oriented vector pipeline 和性能优化属于后续 Phase 6 工作

## 关键目录

- `DESIGN.md`：架构目标、范围和原则
- `PLANS.md`：phase 划分、文件落点、MVP 边界
- `AGENTS.md`：当前实现快照和工程 guardrails
- `examples/riscv/`：当前 backend 的正式例子面
- `testing/python/riscv/`：当前 backend 的正式回归测试面
- `tilelang/tladapter/`：LLVM/MLIR toolchain discovery 与 pass-pipeline wrapper
- `tilelang/jit/adapter/riscv/`：artifact export、host wrapper、runner、adapter
- `src/target/codegen_linalg_riscv.cc`：TIR -> MLIR 主入口
- `src/target/rt_mod_linalg_riscv.cc`：runtime module 注册

## cc-connect

如果你通过 `cc-connect` 驱动这个仓库：

- 不要把密钥写进仓库
- 凭据只放在 `~/.cc-connect/config.toml`
- `work_dir` 指向本仓库根目录
- 推荐让该项目运行在独立的 `tilelang-riscv` conda 环境里

发送当前会话消息：

```bash
cc-connect --config /path/to/config.toml send -m "short message"
```

添加 cron：

```bash
cc-connect --config /path/to/config.toml \
  cron add -c "0 10 * * *" \
  --prompt "总结当前 RISC-V 后端进度，并给出下一步计划"
```
