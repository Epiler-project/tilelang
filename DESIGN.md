# TileLang-RISCV 设计实现文档

## 1. 目标

本文档给出一个面向实现的设计方案：在原版 `tilelang` 仓库中新增一条面向 RISC-V 的 MLIR backend，使 TileLang 生成的 TIR 不再直接走现有的 TVM/target-specific codegen，而是进入一条新的结构化 MLIR 路径：

```text
TileLang DSL
  -> TileLang / TVM TIR
  -> memref + tensor + linalg + scf + bufferization
  -> vector
  -> llvm
  -> RISC-V / RVV
```

这条路线的核心目标不是替代现有 CUDA/HIP backend，而是为 `tilelang-riscv` 提供一条适合 CPU/RISC-V 的、结构化的、可复用 MLIR 基础设施的 lowering 路径。

这里的“对接 MLIR”指的是：

- 在 `3rdparty/` 中真正引入可用的 MLIR/LLVM 基础设施
- 在 C++ 中使用真实 MLIR API 构建 IR
- 在 Python 侧导出最小必要的 binding / pass driver
- 不采用“直接拼接 MLIR 字符串”作为核心 codegen 方案


## 2. 结论先行

结论是：**可行，但必须作为一条新的 structured backend 来实现**。

不能采用的路线：

- 先让原版 TileLang 按现有流程完整 lower 成 CUDA/HIP 风格的 late TIR
- 再尝试从已经硬件化的 IR “恢复” 出 `linalg`

应该采用的路线：

- 在 TileLang 仍然保留结构信息时切出一条新的 backend
- 直接把 TIR 中的结构化语义映射到 MLIR 的 `linalg/tensor/memref/scf`
- 在 C++ 中构建真实的 `mlir::ModuleOp`
- 将不适合 `linalg` 的部分限制在 MVP 范围之外


## 3. 设计原则

### 3.1 新 backend，而不是修改现有 GPU backend

原版 `tilelang` 主干当前的 lowering 和 target optimization 明显偏向 GPU backend，包含：

- `LowerTileOp`
- layout / fragment / shared memory 相关 lowering
- pipeline / async copy / barrier
- vectorize / storage rewrite / unroll
- target-specific codegen

这些流程对于 `cuda/hip/metal` 是合理的，但对于 `linalg + memref` 路线并不合适。  
因此 `tilelang-riscv` 应当引入一个**独立 target 分支**。

### 3.2 从结构化 TIR 切入

`linalg` 适合表达：

- 规则的 elementwise
- 规则的 reduction
- matmul / batch matmul
- 规则 slice / copy / subview

`linalg` 不适合直接表达：

- 显式 barrier / async copy
- warp/thread binding
- shared/local.fragment 等 GPU 特有 memory hierarchy
- mma / wgmma / custom hardware intrinsic

因此切入点必须早于这些硬件化 pass。

### 3.3 backend 形态分两步

建议分为两个阶段：

- 阶段 A：TileLang TIR 到内存中的 MLIR Module，先打通真实 `linalg` IR 生成
- 阶段 B：接 MLIR pass pipeline，继续 lower 到 LLVM IR 和 RISC-V 目标代码

这样能先验证 IR 设计是否成立，再做后续性能优化。


## 4. 与当前 `npuir` 分支的关系

当前参考仓库中的 `npuir` 路线已经证明了一件重要的事：**TileLang TIR 直接对接 MLIR 基础设施是可落地的**。

可复用的经验主要有：

- 在 `tilelang/engine/lower.py` 中新增 target 分支
- 在 `tilelang/engine/phase.py` 中新增专用 lowering / optimize pipeline
- 在 `src/target/` 中增加一个基于 visitor 的 TIR -> MLIR codegen
- 在 C++ 中直接依赖 MLIR API，而不是只做文本导出
- 在 Python 侧通过 `tladapter` / pass pipeline 暴露 MLIR 能力
- 使用 `scf`、`memref`、`tensor`、`bufferization` 管理循环、region、copy、buffer ABI

但 `npuir` 不能直接复用为 `riscv` backend，原因是它的计算核心最终还是发射到 NPU 专用 dialect/op，例如：

- `hivm`
- `hfusion`
- NPU 专用 ABI / runtime attrs
- `tl.npuir_*` 风格的硬件化语义

因此 `tilelang-riscv` 应当**借鉴其工程组织方式**，但**重写计算 op 的映射目标**。


## 5. 总体架构

### 5.1 建议的 target 形式

有两种命名方案：

- 方案 A：新增 target kind `riscv`
- 方案 B：新增 target kind `linalg_riscv`

推荐方案 B。

原因：

- `riscv` 容易和后端 LLVM target triple 混淆
- 本项目本质上是一个“先发射 MLIR，再下沉到 RISC-V”的 backend
- `linalg_riscv` 能明确表达“它不是直接 LLVM codegen，而是 MLIR structured backend”

后续如需对外暴露简单接口，可在 Python 侧支持：

```python
@tilelang.jit(target="riscv")
```

然后在内部归一化为 `linalg_riscv`。

### 5.2 总体 lowering 流程

建议流程：

```text
Python DSL
  -> TVM TIR
  -> PreLowerSemanticCheck
  -> LowerAndLegalizeForRISCV
  -> OptimizeForRISCV
  -> CodeGenTileLangLinalgRISCV
  -> MLIR ModuleOp
  -> optional textual dump (.mlir)
  -> MLIR pass pipeline
  -> LLVM IR
  -> RISC-V object / asm / shared library
```

### 5.3 推荐的 MLIR dialect 组合

核心 dialect：

- `func`
- `arith`
- `scf`
- `tensor`
- `memref`
- `bufferization`
- `linalg`
- `vector`

后端 dialect：

- `cf`
- `llvm`

不建议一开始就引入自定义 RISC-V TileLang dialect。  
只有当某些 TileLang 语义确实无法被 `linalg/tensor/memref/scf` 表达时，才增加极少量 bridge dialect 或 custom op。

### 5.4 MLIR 集成形态

建议采用和 `npuir` 相近的工程组织方式，但目标改为通用 MLIR dialect：

- `3rdparty/llvm-project`
  - 作为仓库内 vendored submodule 管理
  - 当前固定到 `llvmorg-21.1.7`
  - 提供 LLVM + MLIR 头文件、库、工具链
- `maint/scripts/build_llvm_mlir.sh`
  - 负责在仓库内构建 host 侧 LLVM/MLIR 安装目录
- `tilelang/tladapter/toolchain.py`
  - 负责在 Python 侧定位 LLVM/MLIR 安装根目录、`mlir-opt`、`mlir-translate`、`llc`、`clang`
- `src/target/codegen_linalg_riscv.cc`
  - 在 C++ 中直接构建 `mlir::ModuleOp`
- `tilelang/tladapter/`
  - 暴露 pass pipeline、模块 parse/serialize、driver 封装
  - 当前先通过 vendored `mlir-opt` 驱动 pass pipeline
  - vendored MLIR Python package 路径发现能力已补齐，后续可切回进程内 binding
- `tilelang/jit/adapter/riscv/`
  - 负责 Python 侧编译、导出、运行入口

不建议第一版只做 “TIR -> MLIR string printer”，因为后续在：

- region/slice 组合
- `linalg` region/body 构造
- type / attribute 一致性
- pass pipeline 调试

这些环节上都会迅速放大维护成本。

如果为了先打通 target plumbing、runtime registration、Python dispatch，临时出现一个只用于 `inspect_source()` 的占位 source module，
它也只能被视为 **Phase 0 过渡件**，不能算作设计完成。真正满足本设计文档的实现，必须在 Phase 1 起切换到
“用 MLIR C++ API 构建内存中的 `mlir::ModuleOp`，再按需 dump/serialize”的路径。

### 5.5 Toolchain 约束

为了保证长期维护成本可控，`tilelang-riscv` 不应依赖“系统里碰巧装过一个 LLVM”。

- 首选来源是仓库内 `3rdparty/llvm-project`
- 构建输出默认放在：
  - `3rdparty/llvm-project/build-host`
  - `3rdparty/llvm-project/install`
- Python/CMake 两侧统一优先识别：
  - `TILELANG_RISCV_LLVM_ROOT`
  - `TILELANG_LLVM_INSTALL_DIR`

这意味着：

- 源码版本通过 submodule 固定
- 构建方式通过 `maint/scripts/build_llvm_mlir.sh` 固定
- 运行时发现逻辑通过 `tilelang.tladapter.toolchain` 固定

这样后续无论是接 C++ MLIR API、Python binding，还是 artifact export，都不会漂移到不可复现的外部环境。

当前实现状态补充：

- 当 `TILELANG_RISCV_MLIR_MODE=ON` 且 vendored LLVM/MLIR 已安装后，
  `src/target/codegen_linalg_riscv.cc` 已经会使用真实 MLIR C++ API 构造
  `mlir::ModuleOp`
- `src/target/rt_mod_linalg_riscv.cc` 已经返回专用 `mlir` source module，
  不再复用 `CSourceModuleCreate(..., "mlir", ...)`
- 当前已经打通的 structured lowering 子集包括：
  - `PrimFunc` buffer/scalar 参数到 `func.func` 参数
  - unit `thread_extent` / `T.Kernel(..., threads=1)` shell
  - `BlockRealize/Block`
  - `For -> scf.for`
  - `IfThenElse -> scf.if`
  - `AllocBuffer/BufferRealize/DeclBuffer -> memref.alloca`
  - static compact row-major buffer parameters with explicit strides
  - simple contiguous `match_buffer -> memref.subview`
  - `tl.tileop.copy -> memref.copy` or `scf + memref.load/store` fallback
  - `tl.tileop.fill -> scf + memref.store` fallback
  - `tl.tileop.gemm_py -> linalg.matmul` / `linalg.matmul_transpose_a` /
    `linalg.matmul_transpose_b` for static 2D single-transpose matmul
  - `BufferLoad/BufferStore -> memref.load/store`
  - simple reduction init block 通过 `LowerInitBlock` 降成 `scf.if` fallback
  - 常量、`Cast`、`Add/Sub/Mul/Div`、比较、`Select`
- 当前已经打通的 artifact/export + host 验证子集包括：
  - `emit_mlir()`
  - `emit_llvm_ir()`
  - `emit_asm()`
  - `emit_object()`
  - `build_host_shared_library()`
  - `load_host_module()`
  - `run_host()`
  - `tilelang.compile(..., target="riscv")`
  - 基于 `ctypes + NumPy` 的 flattened memref ABI host simulation
  - 基于 `RiscvKernelAdapter` 的轻量 CPU torch-facing runtime
  - `linalg_riscv` 的 cache serialization / reload
- 当前默认 debug pipeline 为：
  - `canonicalize`
  - `cse`
  - `func.func(convert-linalg-to-loops)`
  - `canonicalize`
  - `cse`
  - `convert-scf-to-cf`
  - `expand-strided-metadata`
  - `finalize-memref-to-llvm`
  - `convert-arith-to-llvm`
  - `convert-func-to-llvm`
  - `convert-cf-to-llvm`
  - `reconcile-unrealized-casts`
- 已有自动化样例覆盖：
  - simple copy
  - elementwise add
  - scalar-param saxpy
  - if-guarded store
  - local `alloc_buffer` staging
  - contiguous subview / `match_buffer`
  - reduce-sum fallback
  - TileLang `T.copy` kernel shell
  - TileLang `T.clear` / fill kernel shell
  - TileLang `T.gemm -> linalg.matmul` kernel shell
  - TileLang `T.gemm(..., transpose_A=True)`
  - TileLang `T.gemm(..., transpose_B=True)`
  - `.mlir/.ll/.s/.o` artifact export
  - x86 host shared-library build and copy-kernel correctness
  - `examples/riscv/example_vector_add.py`
  - `examples/riscv/example_copy.py`
  - `examples/riscv/example_reduce_sum.py`
  - `examples/riscv/example_matmul.py`
  - 上述 examples 的 `--run-host` 与 `--emit-asm/--emit-object`
  - direct `tilelang.compile(..., target="riscv")` host execution
- 仍然属于后续任务的部分主要是：
  - 更完整的 region / subview 组合与 rank-reduction 场景
  - 更直接的 reduction 识别，而不是只依赖 `LowerInitBlock + scf` fallback
  - `linalg.generic`、`linalg.reduce`
  - simultaneously-transposed / batched / mixed-shape `tl.gemm`
  - qemu / spike / rv64 smoke runner
  - `--run-qemu` 背后的真实执行器
  - 当前机器缺少 `qemu-riscv64` / `spike` / `pk`，因此真实 RISC-V runner 还缺运行环境验证


## 6. 切入点设计

### 6.1 为什么不能复用原版完整 pass pipeline

原版 `tilelang` 的完整 lowering 会逐步引入：

- target-specific layout
- GPU memory hierarchy
- async copy
- warp specialization
- TMA / WGMMA / MMA
- flatten / storage rewrite / unroll

这些 pass 会让 IR 越来越接近 GPU 硬件执行形态，而不是 structured tensor IR。

因此 `tilelang-riscv` 必须在这些 pass 之前切出。

### 6.2 推荐切点

推荐在原版 `LowerAndLegalize` 的前半段之后切出，不再继续进入 GPU 特化 pass。

推荐保留的前置 pass：

- `BindTarget`
- `AddWrapperForSingleBufStore`
- `LegalizeNegativeIndex`
- `VerifyParallelLoop`
- `InjectAssumes`
- `Simplify`

可选保留：

- `LegalizeSafeMemoryAccess`
- `HoistNonRestrictParams`

建议跳过：

- `LowerBlackwell2SM`
- `LayoutInference`
- `LowerTileOp`
- `LowerL2Persistent`
- `LowerSharedTmem`
- `MultiVersionBuffer`
- `ProducerConsumerWarpSpecialized`
- `LowerSharedBarrier`
- `PipelinePlanning`
- `InjectSoftwarePipeline`
- `RewriteWgmmaSync`
- `FlattenBuffer`
- `StorageRewrite`
- `UnrollLoop`

### 6.3 推荐的 RISCV 专用 phase

新增两个 phase：

- `LowerAndLegalizeForRISCV(mod, target)`
- `OptimizeForRISCV(mod, target)`

其中：

- `LowerAndLegalizeForRISCV` 负责将前端 TileLang TIR 清洗成一个结构化、可 codegen 的子集
- `OptimizeForRISCV` 只做有利于 `linalg/scf` codegen 的轻量规范化，不做 GPU 风格硬件 lowering


## 7. TIR 子集定义

`tilelang-riscv` MVP 不应该试图覆盖所有 TileLang 特性。  
建议首先支持一个明确的 structured 子集。

### 7.1 MVP 支持

- 标量表达式
  - `IntImm` / `FloatImm`
  - `Cast`
  - `Add/Sub/Mul/Div/Min/Max`
  - `Cmp`
  - `Select`
- 控制流
  - `For`
  - `IfThenElse`
  - `SeqStmt`
- buffer
  - `AllocBuffer`
  - `BufferLoad`
  - `BufferStore`
  - region/slice
- TileLang 高层 op
  - `tl.copy`
  - `tl.gemm`
- 规则化 loop-nest
  - elementwise
  - reduction
  - matmul

### 7.2 MVP 暂不支持

- `T.Pipelined`
- async copy
- TMA
- warp/thread binding
- barrier / sync
- fragment/shared/local 这些 GPU 特化 storage scope
- 自定义硬件 intrinsic
- 明显依赖 GPU 层级调度的 kernel

### 7.3 fallback 策略

对于暂不支持的 op，建议 codegen 明确报错，错误信息包含：

- 对应 TIR 节点类型
- 对应 call/op 名称
- 为什么不支持
- 建议改写为哪类结构化 kernel


## 8. TIR 到 MLIR 的映射策略

### 8.1 总体策略

采用 visitor 风格 codegen：

- `ExprFunctor<mlir::Value>`
- `StmtFunctor<void>`

与 `npuir` 分支的 `CodeGenTileLangNPUIRDEV` 类似，但目标 dialect 改为通用 MLIR dialect。
codegen 的核心对象应是：

- `mlir::MLIRContext`
- `mlir::OpBuilder`
- `mlir::OwningOpRef<mlir::ModuleOp>`

而不是直接输出字符串。

推荐新增：

- `src/target/codegen_linalg_riscv.h`
- `src/target/codegen_linalg_riscv.cc`

### 8.2 基础节点映射

| TIR / TileLang | MLIR |
| --- | --- |
| `ForNode` | `scf.for` |
| `IfThenElseNode` | `scf.if` |
| `IntImm/FloatImm` | `arith.constant` |
| `Add/Sub/Mul/Div` | `arith.*` |
| `Min/Max` | `arith.minimumf/maximumf` 或 cmp+select |
| `Cast` | `arith.ext*` / `arith.trunc*` / `arith.index_cast` |
| `BufferLoad` | `memref.load` 或 `tensor.extract` |
| `BufferStore` | `memref.store` 或 `tensor.insert` |

### 8.3 Buffer 与函数参数

函数 ABI 建议遵循：

- 输入/输出 buffer 参数先统一表示为 `memref`
- 计算内部如果要使用 `linalg on tensors`，则在边界处通过 `bufferization.to_tensor` 进入 tensor world
- 写回时通过 `bufferization.materialize_in_destination` 或 `memref.subview + copy` 回到 memref world

推荐做法：

- 函数参数：`memref<...>`，动态 shape 时使用 `memref<*xT>` + `reinterpret_cast`
- 内部计算：优先用 `tensor`
- 边界 copy：用 `memref.subview` / `tensor.extract_slice` / `tensor.insert_slice`

### 8.4 region/slice

region 是连接 TileLang TIR 与 `memref/tensor` 的关键。

推荐映射：

- `Buffer + Range[]`
  - 到 `memref.subview`
  - 或 `tensor.extract_slice`

这部分可以直接参考当前 `npuir` 分支中对 region 的处理方式，但删去 NPU 专用逻辑。

### 8.5 `tl.copy`

推荐统一抽象为“structured copy across buffer/tensor boundary”。

典型映射：

- `memref -> memref`
  - `memref.subview`
  - `memref.copy`
- `memref -> tensor`
  - `memref.subview`
  - `memref.copy` 到临时 buffer
  - `bufferization.to_tensor`
  - `tensor.insert_slice`
- `tensor -> memref`
  - `tensor.extract_slice`
  - `bufferization.materialize_in_destination`
- `tensor -> tensor`
  - `tensor.extract_slice`
  - `tensor.insert_slice`

这里不需要自定义 op，完全可以走已有 dialect。

### 8.6 `tl.gemm`

这是整个方案的核心。

建议：

- 如果 `tl.gemm` 满足标准二维矩阵乘语义
  - 直接映射为 `linalg.matmul`
- 如果存在批维
  - 映射为 `linalg.batch_matmul`
- 如果存在转置
  - 在 codegen 时通过 shape/iterator 映射处理
  - 或先插入显式 transpose/tensor.expand_shape 等中间转换

MVP 范围内建议只支持：

- 2D `matmul`
- 常见 dtype 组合
- 无复杂 layout permutation 的标准矩阵语义

### 8.7 Elementwise

对于规则化 elementwise loop nest，推荐转换为：

- `linalg.generic`

条件是：

- 访问模式可被分析为 affine/strided
- 输出 shape 清晰
- loop nest 不包含复杂控制流

否则 fallback 为：

- `scf.for + arith + tensor/memref load/store`

### 8.8 Reduction

对于规则 reduction，推荐：

- 优先 `linalg.reduce`
- 若模式过于复杂，fallback 到 `linalg.generic`
- 再不行，用 `scf.for`

### 8.9 不可结构化情况的处理

遇到以下情况，不应勉强进 `linalg`：

- 非规则指针算术
- 无法恢复出规则 slice 的访问
- 跨越多个不一致 shape 的复杂 update

这些情况应该：

- fallback 到 `scf + memref`
- 或直接拒绝并报错


## 9. MLIR pass pipeline 设计

### 9.1 推荐 pipeline

建议的首版 pipeline：

```text
canonicalize
cse
one-shot-bufferize
canonicalize
cse
convert-linalg-to-vector
canonicalize
cse
convert-vector-to-scf            (可选)
lower-affine                     (可选)
convert-scf-to-cf
expand-strided-metadata
finalize-memref-to-llvm
convert-vector-to-llvm
convert-func-to-llvm
reconcile-unrealized-casts
```

如果目标是充分利用 RVV，则中间阶段应尽量保留 `vector`，避免过早转 loops。

### 9.2 两条后端策略

可以保留两条可切换路线：

- 路线 A：`linalg -> loops`
  - 更容易调试
  - 先求功能正确
- 路线 B：`linalg -> vector -> llvm`
  - 更适合 RVV
  - 也是长期推荐路线

MVP 建议先打通路线 A，再逐步把核心算子切到路线 B。

### 9.3 RVV 对接建议

对 RISC-V 的长期目标，应尽量让：

- `linalg.matmul`
- `linalg.generic`
- `linalg.reduce`

在 vector 化后形成：

- `vector.transfer_read/write`
- `vector.fma`
- `vector.contract`
- `vector.reduction`

再依赖 LLVM/MLIR 的 RISC-V backend 做 RVV lowering。


## 10. 工程改动清单

### 10.1 Python 侧

建议新增或修改：

- `tilelang/engine/lower.py`
  - 新增 `linalg_riscv` target 分支
- `tilelang/engine/phase.py`
  - 新增 `LowerAndLegalizeForRISCV`
  - 新增 `OptimizeForRISCV`
- `tilelang/utils/target.py`
  - 支持解析 `riscv` / `linalg_riscv`
- `tilelang/jit/`
  - 新增 `jit_riscv.py`
  - 调用 MLIR pipeline 与后续 LLVM/RISC-V 编译工具链
- `tilelang/tladapter/`
  - 新增 native binding / pipeline driver
- `tilelang/tladapter/transforms/`
  - 增加需要的 MLIR pass wrapper

### 10.2 C++ 侧

建议新增：

- `src/target/codegen_linalg_riscv.h`
- `src/target/codegen_linalg_riscv.cc`
- `src/target/rt_mod_linalg_riscv.cc`

可选新增：

- `src/tladapter/`
  - 如果采用独立 pybind/native module，可在此放置 PassPipeline / parse / verify 封装

### 10.3 Thirdparty / 构建侧

建议新增：

- `3rdparty/llvm-project`
  - 以固定 commit 或 release tag 引入
- CMake 选项
  - `USE_LINALG_RISCV`
  - `LLVM_ROOT` / `MLIR_ROOT` 或等价路径
- Python 打包逻辑
  - 导出 `tladapter` 需要的 native module 和 Python 包

### 10.4 可选辅助 pass

- `src/transform/lower_riscv_block.cc`
  - 如果 codegen 不希望看到残余 block shell，可增加这个轻量 pass

### 10.5 注册接口

建议注册：

- `target.build.tilelang_linalg_riscv`
- `TVM_REGISTER_TARGET_KIND("linalg_riscv", ...)`

如果需要兼容简单接口，则 Python 端允许 `target="riscv"`，但内部映射到 `linalg_riscv`。


## 11. 实现步骤

### 阶段 0：文法和范围冻结

目标：

- 明确 MVP 支持的 TIR 子集
- 明确不支持项
- 明确 MLIR 引入方式
- 先不碰性能

交付物：

- 本设计文档
- 一组最小示例 kernel
- MLIR / tladapter 接入方案说明

### 阶段 1：引入真实 MLIR，并打通最小 codegen

目标：

- `lower(..., target="riscv")` 能生成真实 MLIR module，并可导出、可验证的 `.mlir`

实现：

- 在 `3rdparty/` 中接入 MLIR/LLVM
- 增加 `tladapter` native binding / pass driver
- 新增 target 分支
- 新增 codegen visitor
- 打通：
  - 常量/算术
  - `For`
  - `If`
  - `AllocBuffer`
  - `BufferLoad/Store`
  - region
  - `tl.copy`

测试：

- vector add
- simple copy
- simple reduction

### 阶段 2：支持 `tl.gemm -> linalg.matmul`

目标：

- 打通 matmul 主链路

实现：

- 在 `VisitExpr_(CallNode)` 中识别 `tl.gemm`
- 分析 A/B/C region 与 shape
- 发射 `linalg.matmul`

测试：

- 基础 matmul
- 带外层 tiled loop 的 matmul

### 阶段 3：接 bufferization + LLVM + RISC-V

目标：

- 生成 LLVM IR
- 编译成 RISC-V 目标文件
- 先在本机 x86 host 上验证同一条 LLVM lowering 通路能真实执行

实现：

- Python 侧集成 MLIR/LLVM 命令
- 支持导出：
  - `.mlir`
  - `.ll`
  - `.o`
  - `.s`
- 提供 host 侧运行封装：
  - `build_host_shared_library()`
  - `load_host_module()`
  - `run_host()`
- 当前默认走 debug loops 路线：
  - `func.func(convert-linalg-to-loops)`
  - `convert-scf-to-cf`
  - `finalize-memref-to-llvm`
  - `convert-arith/func/cf-to-llvm`

测试：

- host 模拟
- qemu / spike / rv64 仿真

当前阶段补充：

- x86 host 验证已经不是手工探针，而是仓库内正式能力：
  - `tilelang/jit/adapter/riscv/wrapper.py`
  - `tilelang/jit/adapter/riscv/adapter.py`
  - `tilelang/jit/kernel.py` 上的 `RiscvKernelAdapter` 选择逻辑
- 示例入口已经补齐到：
  - `examples/riscv/common.py`
  - `examples/riscv/example_vector_add.py`
  - `examples/riscv/example_copy.py`
  - `examples/riscv/example_reduce_sum.py`
  - `examples/riscv/example_matmul.py`
- 当前 examples 支持：
  - `--print-tir`
  - `--emit-mlir`
  - `--emit-llvm`
  - `--emit-asm`
  - `--emit-object`
  - `--run-host`
- `--run-qemu` 仍然保留为下一阶段工作

### 阶段 4：性能优化

目标：

- 利用 `vector` 与 RVV 做性能优化

实现：

- 增加 `linalg -> vector` 路线
- 增加对 transfer/contract/reduction 的专项优化


## 12. 示例映射

### 12.1 Elementwise add

TileLang 语义：

```text
for i, j:
  C[i, j] = A[i, j] + B[i, j]
```

推荐 MLIR：

- `linalg.generic` on tensors

如果模式识别失败，则：

- `scf.for`
- `memref.load`
- `arith.addf`
- `memref.store`

### 12.2 Matmul

TileLang 语义：

```text
T.gemm(A_tile, B_tile, C_tile)
```

推荐 MLIR：

- `tensor.extract_slice` / `memref.subview`
- `linalg.matmul`
- `tensor.insert_slice` / materialize back

### 12.3 Copy

TileLang 语义：

```text
T.copy(src_region, dst_region)
```

推荐 MLIR：

- `memref.subview`
- `tensor.extract_slice`
- `tensor.insert_slice`
- `memref.copy`
- `bufferization.materialize_in_destination`


## 13. 与 Triton CPU / Triton Shared 的关系

`triton-cpu` / `triton-shared` 的经验说明：

- 从一个结构化 DSL IR lower 到 `linalg/tensor/memref` 是成立的
- 但前提是先做 structured memory access analysis
- 并且只覆盖结构化访问子集

因此对 TileLang 来说，正确类比不是当前 `npuir` 分支的硬件化路径，而是：

```text
TileLang structured TIR
  -> MLIR middle layer
  -> target-specific lower
```

这也是本文档采用的总体设计。


## 14. 风险点

### 14.1 `tl.gemm` 的语义恢复

如果 `tl.gemm` 在不同 kernel 中的 region 表示方式差异较大，`linalg.matmul` 的统一映射会变复杂。  
建议 MVP 只支持标准 matmul 形态。

### 14.2 指针/region 分析

一旦 buffer 访问不是规则 slice，`linalg` 路线会很快失效。  
因此必须早做“structured access only”的约束。

### 14.3 原版 TileLang pass 的耦合

如果直接把新 backend 塞进现有完整 phase pipeline，会受到大量 GPU pass 影响。  
因此要尽量把 RISCV 路线做成独立 phase。

### 14.4 性能与正确性的阶段冲突

MVP 首先应该追求：

- IR 正确
- 编译链打通

而不是一开始就追求 RVV 极致性能。


## 15. 测试计划

### 15.1 Level 0: IR 级测试

- 检查生成的 MLIR 是否合法
- `mlir-opt --verify-diagnostics`
- 检查是否含预期 op：
  - `scf.for`
  - `linalg.matmul`
  - `linalg.generic`
  - `memref.subview`

### 15.2 Level 1: 功能测试

- vector add
- reduce sum / max
- matmul
- matmul + elementwise epilogue

### 15.3 Level 2: 编译测试

- MLIR -> LLVM IR
- LLVM IR -> RISC-V asm/object
- qemu/spike 运行验证

### 15.4 Level 3: 性能测试

- RVV 打开/关闭对比
- `linalg-to-loops` 与 `linalg-to-vector` 对比
- 不同 tile size 对比


## 16. 推荐的 MVP 范围

为了控制复杂度，建议 MVP 范围严格限定为：

- 单 kernel
- 无 GPU 特定 schedule
- 规则 buffer region
- `tl.copy`
- `tl.gemm`
- 规则 elementwise / reduction
- 输出为 MLIR + LLVM + RISC-V asm

不建议 MVP 就支持：

- FlashAttention
- complex fused kernel
- async pipeline
- expert mode 风格的显式硬件控制


## 17. 参考实现蓝图

建议按如下顺序落地：

1. 在 `3rdparty/` 中引入 MLIR/LLVM，并打通基础构建
2. 增加 `tladapter` 的 native binding / pipeline driver
3. 新建 `target="linalg_riscv"` 分支
4. 写 `CodeGenTileLangLinalgRISCV`
5. 先支持：
   - `For`
   - `If`
   - `BufferLoad/Store`
   - `tl.copy`
6. 打通 `linalg.generic`
7. 打通 `tl.gemm -> linalg.matmul`
8. 接 MLIR pipeline 到 LLVM/RISC-V
9. 再考虑 `vector` 和 RVV 优化


## 18. 最终建议

`tilelang-riscv` 最合适的技术路线不是：

- “把 TileLang 改成另一个 TVM codegen”

而是：

- “给 TileLang 增加一个 structured MLIR backend”

其中：

- `linalg` 负责结构化计算
- `tensor/memref/bufferization` 负责数据边界
- `scf` 负责控制流
- `vector` 负责 RVV 优化
- `llvm` 负责最后的 RISC-V 落地

这是当前工程复杂度、实现可行性、后续可维护性三者之间最平衡的方案。
