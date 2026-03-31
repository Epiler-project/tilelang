# TileLang-RISCV Implementation Plan

## 1. Goal

This document turns `DESIGN.md` into an executable implementation plan for the current `tilelang` repository.

The final target is:

- add a new structured backend `linalg_riscv`
- integrate real MLIR/LLVM infrastructure under `3rdparty/`
- expose a Python-facing MLIR driver/binding layer
- support the MVP TileLang/TIR subset defined in `DESIGN.md`
- emit valid MLIR
- lower MLIR to LLVM IR and RISC-V asm/object
- run a curated backend-neutral example set for correctness checking
- keep existing CUDA/HIP/Metal behavior unchanged

## 2. Frozen Decisions

These decisions should be treated as fixed unless a blocking issue appears.

- target kind: `linalg_riscv`
- Python alias: `riscv` maps internally to `linalg_riscv`
- architecture strategy: new backend branch, not modifications to existing GPU codegen
- MLIR integration strategy: real MLIR C++ API, not raw string concatenation
- lowering split: TIR -> structured MLIR -> LLVM -> RISC-V
- function ABI: `memref` at function boundaries
- internal compute IR: prefer `tensor + linalg`; fallback to `scf + memref` when structure recovery fails
- original `examples/` policy:
  - treat the upstream `examples/` tree as a source pool for coverage planning
  - do not require all original examples to run unchanged on `linalg_riscv`
  - define completeness using a curated portable subset plus explicit non-goals
- MVP feature set:
  - scalar arithmetic
    - `Add/Sub/Mul/Div/Mod/FloorDiv/FloorMod/Min/Max`
  - `For`, `IfThenElse`, `SeqStmt`
  - `AllocBuffer`, `BufferLoad`, `BufferStore`
  - structured region/slice
  - `tl.copy`
  - structured elementwise/reduction
  - `tl.gemm -> linalg.matmul`
- explicit non-goals for MVP:
  - `T.Pipelined`
  - async copy
  - TMA
  - warp/thread binding
  - fragment/shared/local GPU-specific scopes
  - barrier/sync
  - custom hardware intrinsics
  - FlashAttention and other fused GPU-style kernels
  - "support every original example unchanged"

## 3. Definition Of Done

The project is considered complete for MVP when all items below are true.

- `lower(..., target="riscv")` or `lower(..., target="linalg_riscv")` works
- the backend emits valid MLIR for:
  - vector add
  - simple copy
  - reduce sum
  - 2D matmul
- `tl.gemm` lowers to `linalg.matmul`
- the pipeline can export:
  - `.mlir`
  - `.ll`
  - `.s`
  - `.o`
- the local host simulation path is runnable
- there are automated tests under `testing/python/riscv/`
- there are runnable examples under `examples/riscv/`
- the repo documents a portable coverage matrix derived from original `examples/`
- MVP completion does not require the original upstream examples to run unchanged

## 4. Repository Mapping

`DESIGN.md` is conceptually correct, but a few paths should be adapted to the current repository layout.

### Existing files to modify

- `tilelang/utils/target.py`
- `tilelang/engine/phase.py`
- `tilelang/engine/lower.py`
- `tilelang/jit/execution_backend.py`
- `tilelang/jit/adapter/utils.py`
- `CMakeLists.txt`
- `setup.py` or equivalent packaging entry if native Python MLIR binding is bundled

### New files to add

- `3rdparty/llvm-project`
- `maint/scripts/build_llvm_mlir.sh`
- `tilelang/tladapter/__init__.py`
- `tilelang/tladapter/toolchain.py`
- `tilelang/tladapter/utils.py`
- `tilelang/tladapter/transforms/__init__.py`
- `tilelang/tladapter/transforms/mlir.py`
- `src/target/codegen_linalg_riscv.h`
- `src/target/codegen_linalg_riscv.cc`
- `src/target/rt_mod_linalg_riscv.cc`
- `tilelang/jit/adapter/riscv/__init__.py`
- `tilelang/jit/adapter/riscv/adapter.py`
- `tilelang/jit/adapter/riscv/libgen.py`
- `tilelang/jit/adapter/riscv/wrapper.py`
- `testing/python/riscv/test_riscv_target_parse.py`
- `testing/python/riscv/test_riscv_toolchain.py`
- `testing/python/riscv/test_riscv_lower_to_mlir.py`
- `testing/python/riscv/test_riscv_copy_codegen.py`
- `testing/python/riscv/test_riscv_reduce_codegen.py`
- `testing/python/riscv/test_riscv_matmul_codegen.py`
- `testing/python/riscv/test_riscv_artifact_export.py`
- `examples/riscv/example_vector_add.py`
- `examples/riscv/example_copy.py`
- `examples/riscv/example_reduce_sum.py`
- `examples/riscv/example_reduce_max.py`
- `examples/riscv/example_matmul.py`
- `examples/riscv/example_batched_gemm.py`
- `examples/riscv/common.py`

### Optional helper files

- `src/tladapter/`
- `src/transform/lower_riscv_block.cc`
- `tilelang/contrib/riscv_toolchain.py`

## 5. Toolchain Requirements

Do not start implementation before the toolchain is fixed.

### Source of truth

- vendored LLVM/MLIR source lives at `3rdparty/llvm-project`
- pin the submodule to `llvmorg-21.1.7`
- build the host toolchain with `maint/scripts/build_llvm_mlir.sh`
- discover the install from Python through `tilelang.tladapter.toolchain`

### Required tools

- Python dev environment for TileLang
- `pybind11` and `nanobind` in the active Python environment when `MLIR_ENABLE_BINDINGS_PYTHON=ON`
- MLIR/LLVM build under `3rdparty/` or an equivalent pinned install with:
  - `mlir-opt`
  - `mlir-translate`
  - `mlir-cpu-runner` if available
  - `llc`
  - `clang`
  - `ld.lld`
  - `llvm-dis`
- RISC-V backend enabled in LLVM
- Python package support for the MLIR binding/driver layer
- one runtime path for execution:
  - native x86 host simulation through the flattened memref ABI

### Recommended environment variables

Use these names consistently in the implementation.

```bash
export TILELANG_RISCV_LLVM_ROOT=/opt/llvm-riscv
export PATH=$TILELANG_RISCV_LLVM_ROOT/bin:$PATH

export TILELANG_RISCV_TRIPLE=riscv64-unknown-linux-gnu
export TILELANG_RISCV_CPU=generic-rv64
export TILELANG_RISCV_ATTRS='+m,+a,+f,+d,+c,+v'
export TILELANG_RISCV_ABI=lp64d
export TILELANG_RISCV_SYSROOT=/opt/riscv/sysroot

# Optional only: freestanding simulator execution is not part of MVP DoD.
export TILELANG_RISCV_RUNNER=qemu-riscv64
export TILELANG_RISCV_RUNNER_FLAGS='-L /opt/riscv/sysroot'
```

### Standard bootstrap

Use this sequence as the default maintenance path:

```bash
git submodule update --init --recursive 3rdparty/llvm-project
python -m pip install pybind11
python -m pip install nanobind
maint/scripts/build_llvm_mlir.sh
export TILELANG_RISCV_LLVM_ROOT=$PWD/3rdparty/llvm-project/install
export PATH=$TILELANG_RISCV_LLVM_ROOT/bin:$PATH
```

Top-level TileLang CMake should use:

```bash
-DTILELANG_RISCV_MLIR_MODE=AUTO
```

Switch it to `ON` only after `3rdparty/llvm-project/install` is ready.

### Preflight checks

These commands must pass before phase 3 starts.

```bash
mlir-opt --version
mlir-translate --version
llc --version | rg riscv
clang --version
```

The Python-side discovery layer should also succeed:

```bash
python - <<'PY'
from tilelang.tladapter import toolchain_summary
print(toolchain_summary())
PY
```

## 6. Branch And Commit Strategy

Use one feature branch and land work in narrow commits.

```bash
git switch -c feat/linalg-riscv-mvp
```

Recommended commit slicing:

1. target parsing and phase dispatch
2. C++ codegen skeleton and runtime module registration
3. basic expr/stmt lowering to MLIR
4. `tl.copy` and region/subview lowering
5. elementwise and reduction lowering
6. `tl.gemm -> linalg.matmul`
7. MLIR pipeline and artifact export
8. host/runtime adapter surface
9. tests and examples

## 7. Phase Plan

### Current status snapshot

- Phase 0 target plumbing is landed
- Phase 1 real MLIR scaffolding is landed:
  - vendored `llvm-project` submodule
  - deterministic build script
  - Python toolchain discovery helpers
  - top-level CMake gating for vendored MLIR
  - real C++ MLIR builder using vendored MLIR dialect APIs
  - `tilelang.tladapter.Pipeline` backed by vendored `mlir-opt`
  - dedicated `mlir` source runtime module for `linalg_riscv`
  - base lowering currently covers:
    - `PrimFunc` buffer/scalar params -> `func.func` args
    - `BlockRealize/Block`
    - `For -> scf.for`
    - `IfThenElse -> scf.if`
    - `AllocBuffer/BufferRealize/DeclBuffer -> memref.alloca`
    - unit `thread_extent` / `T.Kernel(..., threads=1)` shell
    - `BufferLoad/BufferStore -> memref.load/store`
    - constants, casts, arithmetic, comparisons, `Select`
  - automated coverage currently includes:
    - tiny kernel shell
    - scalar-param saxpy
    - if-guarded store
    - local alloc-buffer staging
- Phase 2 region/copy/elementwise/reduction is partially landed:
  - structured lowering currently covers:
    - static compact row-major buffer parameters with explicit strides
    - simple contiguous `match_buffer -> memref.subview`
    - `tl.tileop.copy -> memref.copy` or `scf + memref.load/store` fallback
    - `tl.tileop.fill -> scf + memref.store` fallback
    - pure `kDataPar` elementwise loop nests with identity/broadcasted ordered-subsequence
      loads -> `linalg.generic`
    - simple reduction init fallback via `tir.transform.LowerInitBlock`
    - simple full-shape sum/min/max reductions with structured lowering plus fallback paths
    - single-axis additive reduction expressions with identity inputs plus output-broadcast
      inputs -> `linalg.generic`
  - automated coverage currently includes:
    - copy loop
    - elementwise add
    - broadcast elementwise normalize
    - contiguous match-buffer subview
    - reduce-sum fallback
    - `example_reduce_max.py`
    - `example_online_softmax.py` row-sum structured reduction path
    - `example_online_softmax.py` fully structured normalize path
    - TileLang `T.copy` kernel shell
    - TileLang `T.clear` / fill kernel shell
- Phase 3 `tl.gemm -> linalg.matmul` is partially landed:
  - structured lowering currently covers:
    - `tl.tileop.gemm_py -> linalg.matmul` / `linalg.matmul_transpose_a` /
      `linalg.matmul_transpose_b` for static 2D matmul, including double-transpose via
      temporary transpose materialization
  - automated coverage currently includes:
    - TileLang `T.gemm -> linalg.matmul` kernel shell
    - TileLang `T.gemm(..., transpose_A=True)`
    - TileLang `T.gemm(..., transpose_B=True)`
    - TileLang `T.gemm(..., transpose_A=True, transpose_B=True)`
- Phase 4 artifact/export + host-sim surface is partially landed:
  - `tilelang/jit/adapter/riscv/libgen.py` currently exports:
    - `emit_mlir()`
    - `emit_llvm_ir()`
    - `emit_asm()`
    - `emit_object()`
  - `tilelang/jit/adapter/riscv/wrapper.py` currently exports:
    - `build_host_shared_library()`
    - `load_host_module()`
    - `run_host()`
  - the current default debug pipeline is:
    - `canonicalize`
    - `cse`
    - `func.func(convert-linalg-to-loops)`
    - `canonicalize`
    - `cse`
    - `convert-scf-to-cf`
    - `expand-strided-metadata`
    - `lower-affine`
    - `finalize-memref-to-llvm`
    - `convert-math-to-llvm`
    - `convert-arith-to-llvm`
    - `convert-func-to-llvm`
    - `convert-cf-to-llvm`
    - `reconcile-unrealized-casts`
  - automated coverage currently includes:
    - `.mlir/.ll/.s/.o` export
    - native x86 host shared-library build and NumPy correctness on copy
- Phase 5 runner integration and example surface is partially landed:
  - shared helper:
    - `examples/riscv/common.py`
  - runnable examples:
    - `examples/riscv/example_vector_add.py`
    - `examples/riscv/example_copy.py`
    - `examples/riscv/example_reduce_sum.py`
    - `examples/riscv/example_reduce_max.py`
    - `examples/riscv/example_matmul.py`
    - `examples/riscv/example_batched_gemm.py`
    - `examples/riscv/example_dynamic_batched_gemm.py`
    - `examples/riscv/example_gemv.py`
    - `examples/riscv/example_grouped_gemm.py`
    - `examples/riscv/example_dynamic_grouped_gemm.py`
    - `examples/riscv/example_dynamic_shape.py`
    - `examples/riscv/example_rms_norm.py`
    - `examples/riscv/example_online_softmax.py`
    - `examples/riscv/example_topk.py`
    - `examples/riscv/example_convolution.py`
  - current example CLI surface:
    - `--print-tir`
    - `--emit-mlir`
    - `--emit-llvm`
    - `--emit-asm`
    - `--emit-object`
    - `--run-host`
  - automated coverage currently includes:
    - each example runs on the local x86 host path
    - each example emits non-empty RISC-V `.s` and `.o` artifacts
  - `tilelang.compile(..., target="riscv")` now routes to `RiscvKernelAdapter`
  - `linalg_riscv` disk cache serialization / reload is now wired through the host `.so`
  - automated coverage currently includes:
    - direct JIT compile + local CPU execution
    - disk-cache reload without recompilation
- next gap has shifted to late Phase 2 through Phase 6:
  - broader region / subview / slice lowering beyond:
    - the simple contiguous case
    - compact row-major symbolic layouts
    - static-1 rank-reduced views
  - broader reduction recognition beyond:
    - single-axis full-shape sum/min/max patterns
    - additive expression reductions with identity + output-broadcast inputs
  - broader `linalg.generic` / `linalg.reduce` coverage beyond the current simple cases
  - mixed-shape `tl.gemm` beyond the current singleton-dim GEMV and rank-reduced batched slices
  - local `pytest testing/python/riscv -q` also requires a built TileLang/TVM Python environment with `tvm_ffi` importable

## Phase 0: Freeze Scope And Scaffolding

### Goal

Freeze the MVP subset and create the empty backend skeleton.

### Tasks

- add MLIR/LLVM integration entry in `3rdparty/`
- add build/package wiring for MLIR native libraries and Python-side driver
- add `linalg_riscv` and alias `riscv` to `tilelang/utils/target.py`
- add target predicates to `tilelang/jit/adapter/utils.py`
- add empty dispatch branches in:
  - `tilelang/engine/phase.py`
  - `tilelang/engine/lower.py`
- add `tilelang/tladapter/` package skeleton
- add empty C++ files:
  - `src/target/codegen_linalg_riscv.h`
  - `src/target/codegen_linalg_riscv.cc`
  - `src/target/rt_mod_linalg_riscv.cc`
- wire new C++ files into `CMakeLists.txt`

### Acceptance

- the project builds with the backend skeleton and MLIR integration stubs wired in
- `Target("linalg_riscv")` is accepted
- `determine_target("riscv")` resolves to `linalg_riscv`
- the backend does not fall into CUDA/HIP code paths
- if a temporary placeholder source module exists for plumbing, it is explicitly marked Phase 0 only and scheduled for replacement in Phase 1

## Phase 1: Real MLIR Scaffolding

### Goal

Make `lower(..., target="riscv")` build a real MLIR module in C++ and expose a printable/verifiable `.mlir` dump.

### Tasks

- add C++ MLIR context / builder / module scaffolding
- add minimal Python-facing MLIR driver surface through `tilelang/tladapter`
- remove any Phase 0 placeholder source-module-only implementation
- in `tilelang/engine/phase.py`, add:
  - `LowerAndLegalizeForRISCV(mod, target)`
  - `OptimizeForRISCV(mod, target)`
- keep only the early structured passes:
  - `BindTarget`
  - `AddWrapperForSingleBufStore`
  - `LegalizeNegativeIndex`
  - `VerifyParallelLoop`
  - `InjectAssumes`
  - `Simplify`
  - optionally `HoistNonRestrictParams`
- intentionally skip `LegalizeSafeMemoryAccess` on `linalg_riscv`
  - preserve symbolic slice extents for loop-indexed `match_buffer` / `alloc_buffer`
  - avoid rewriting dynamic bounds into nested `if_then_else` expressions before MLIR codegen
- explicitly skip GPU passes:
  - `LayoutInference`
  - `LowerTileOp`
  - `LowerL2Persistent`
  - `LowerSharedTmem`
  - `PipelinePlanning`
  - `InjectSoftwarePipeline`
  - `FlattenBuffer`
  - `StorageRewrite`
  - `UnrollLoop`
- in `tilelang/engine/lower.py`:
  - add a `linalg_riscv` branch before the normal device split/codegen path
  - return a module dump or equivalent MLIR-facing artifact for stage 1
- in `src/target/rt_mod_linalg_riscv.cc`:
  - register `target.build.tilelang_linalg_riscv`
  - add a no-compile variant if helpful

### Scope

Only support:

- module/function shell
- scalar constants
- basic arithmetic
- `For`
- `IfThenElse`
- `AllocBuffer`
- `BufferLoad`
- `BufferStore`

### Acceptance

- `lower(..., target="riscv")` returns an artifact backed by real MLIR construction
- the backend can dump valid `.mlir`
- `mlir-opt --verify-diagnostics` accepts the output
- at least one tiny kernel emits:
  - `func.func`
  - `arith.constant`
  - `scf.for`
  - `memref.load`
  - `memref.store`

Status:

- achieved for the current MVP subset
- follow-up work now focuses on region/reduction/linalg pattern lifting, not on basic MLIR construction anymore

## Phase 2: Region, Copy, Elementwise, Reduction

### Goal

Cover the non-GEMM structured MVP kernels.

### Tasks

- implement region analysis for:
  - `Buffer + Range[]`
  - contiguous slices
  - strided slices if representable
- lower regions to:
  - `memref.subview`
  - `tensor.extract_slice`
  - `tensor.insert_slice`
- implement `tl.copy` lowering
- implement elementwise recognition:
  - prefer `linalg.generic`
  - fallback to `scf.for + arith + memref`
  - current landed status:
    - pure `kDataPar` elementwise loop nests with identity/broadcasted ordered-subsequence
      loads already lower to `linalg.generic`
    - irregular slices / mixed indexing / predicates still stay on the `scf + memref` fallback
- implement reduction recognition:
  - prefer `linalg.reduce`
  - fallback to `linalg.generic`
  - final fallback to `scf.for`
  - current landed status:
    - simple single-axis full-shape sum/min/max reductions now lower to
      `linalg.fill + linalg.reduce`
    - additive reduction expressions such as `row_sum += exp2(A[i, j] - row_max[i])`
      now lower to `linalg.fill + linalg.generic`
    - more complex reduction shapes still stay on the fallback path
- add explicit unsupported-op diagnostics:
  - node type
  - op name
  - why unsupported
  - suggested rewrite

### Acceptance

- vector add emits `linalg.generic` or valid `scf.for`
- simple copy emits `memref.subview` or `memref.copy`
- reduce sum/max emits `linalg.reduce`, `linalg.generic`, or valid `scf` fallback
- unsupported kernels fail with readable error messages

## Phase 3: `tl.gemm -> linalg.matmul`

### Goal

Make the main structured GEMM path work.

### Tasks

- in the call visitor inside `src/target/codegen_linalg_riscv.cc`:
  - recognize `tl.gemm`
  - inspect source/destination regions
  - recover `M`, `N`, `K`
  - verify standard 2D matmul semantics
- lower standard GEMM to:
  - `tensor.extract_slice` or `memref.subview`
  - `linalg.matmul`
  - `tensor.insert_slice` or destination materialization
- reject non-standard cases in MVP:
  - custom layouts
  - non-2D cases
  - irregular region mappings
  - ambiguous transpose semantics

### Acceptance

- a plain 2D matmul lowers to MLIR containing `linalg.matmul`
- a tiled outer loop matmul also lowers correctly
- invalid GEMM patterns fail explicitly instead of generating malformed MLIR

## Phase 4: MLIR Pass Pipeline And Artifact Export

### Goal

Lower MLIR to LLVM IR and RISC-V code, then export artifacts.

### Tasks

- implement toolchain helpers in:
  - `tilelang/jit/adapter/riscv/libgen.py`
  - optionally `tilelang/contrib/riscv_toolchain.py`
- add a stable pipeline with two modes:
  - debug path: `linalg -> loops`
  - optimized path: `linalg -> vector -> llvm`
- current landed default pipeline is:

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

- vector/RVV-oriented optimization remains outside the current completion target

- add export helpers for:
  - `emit_mlir()`
  - `emit_llvm_ir()`
  - `emit_asm()`
  - `emit_object()`
- support output paths for:
  - `.mlir`
  - `.ll`
  - `.s`
  - `.o`

### Acceptance

- the pipeline produces valid `.ll`
- `llc` can produce RISC-V asm
- `clang --target=${TILELANG_RISCV_TRIPLE}` can link a small object if a runtime harness is supplied
- optional freestanding simulator tooling expects `ld.lld` or another RISC-V-capable linker
- `load_host_module()` can compile the same `.ll` into a native host `.so`
- a tiny copy kernel runs correctly on local x86 CPU through the flattened memref ABI

## Phase 5: Runner Integration And Examples

### Goal

Run a curated set of backend-neutral examples end to end.

### Tasks

- add a lightweight adapter in:
  - `tilelang/jit/adapter/riscv/adapter.py`
  - `tilelang/jit/adapter/riscv/wrapper.py`
- do not aim for full PyTorch runtime integration first
- support one required execution mode:
  - host simulation
- keep qemu/spike helpers as optional tooling only
- current landed runner surface:
  - `load_host_module()` for NumPy/x86 host simulation
  - `RiscvKernelAdapter` for a lightweight CPU-torch-facing wrapper over the same host path
  - `tilelang.compile(..., target="riscv")` for direct JIT execution on the same host path
  - `build_qemu_executable()` for a freestanding RISC-V ELF with embedded memref payloads
  - `run_qemu()` for qemu/spike-style execution through a simulator command
- create examples:
  - `examples/riscv/example_vector_add.py`
  - `examples/riscv/example_copy.py`
  - `examples/riscv/example_reduce_sum.py`
  - `examples/riscv/example_matmul.py`
- define the next portable expansion set from original `examples/`:
  - `examples/dynamic_shape/example_dynamic.py`
- each example should support:
  - print TIR
  - print MLIR
  - export asm
  - optional run
- current status:
  - all four examples are landed
  - additional reduction example landed:
    - `examples/riscv/example_reduce_max.py`
  - additional structured coverage example landed:
    - `examples/riscv/example_batched_gemm.py`
  - additional dynamic mixed-shape example landed:
    - `examples/riscv/example_dynamic_batched_gemm.py`
  - Tier 1 portable ports landed:
    - `examples/riscv/example_dynamic_shape.py`
    - `examples/riscv/example_rms_norm.py`
    - `examples/riscv/example_online_softmax.py`
    - `examples/riscv/example_topk.py`
    - `examples/riscv/example_convolution.py`
  - local host execution is wired and tested
  - `--run-qemu` remains available as an optional freestanding helper when an external
    simulator environment exists
  - current automated coverage includes:
    - freestanding ELF build validation on this machine
    - host/artifact coverage for `example_dynamic_shape.py`
    - host/artifact coverage for `example_rms_norm.py`
    - host/artifact coverage for `example_online_softmax.py`
    - host/artifact coverage for `example_topk.py`
    - host/artifact coverage for `example_convolution.py`
    - host/artifact coverage for `example_reduce_max.py`
    - host/artifact coverage for `example_batched_gemm.py`
    - host/artifact coverage for `example_dynamic_batched_gemm.py`
    - host/artifact coverage for `example_gemv.py`
    - host/artifact coverage for `example_grouped_gemm.py`
    - host/artifact coverage for `example_dynamic_grouped_gemm.py`
    - `tilelang.compile(..., target="riscv")` dynamic-shape host execution
    - `tilelang.compile(..., target="riscv")` reduce-max host execution
    - `tilelang.compile(..., target="riscv")` reduction-expression generic host execution
    - `tilelang.compile(..., target="riscv")` broadcast elementwise generic host execution
    - `tilelang.compile(..., target="riscv")` rank-reduced batched GEMM host execution
    - `tilelang.compile(..., target="riscv")` dynamic rank-reduced batched GEMM host execution
    - `tilelang.compile(..., target="riscv")` singleton-dim GEMV host execution
    - `tilelang.compile(..., target="riscv")` compile-time grouped GEMM host execution
    - `tilelang.compile(..., target="riscv")` dynamic grouped GEMM host execution
    - `tilelang.compile(..., target="riscv")` dynamic same-rank copy host execution
    - `tilelang.compile(..., target="riscv")` dynamic rank-reduced copy host execution
    - full `testing/python/riscv` regression currently passes on this machine:
      `89 passed, 1 skipped`
  - dynamic grouped GEMM lowering status:
    - runtime dynamic `group_count` plus `Offsets` / `Sizes` now lower as a single
      `scf.for`-driven grouped dispatch
    - loop-indexed `match_buffer` / `alloc_buffer` bindings are deferred until the loop body so
      symbolic slice extents stay valid for `memref.subview` and `linalg.matmul`
  - dynamic-shape lowering status:
    - buffer shape vars are rebound from function memrefs via `memref.dim`
    - symbolic compact row-major strides remain accepted in the JIT path
    - dynamic `T.copy` and `T.gemm` host kernels are covered as a portable `copy + gemm`
      path, not just a direct loop fallback
    - same-dtype dynamic equal-extent `T.copy` now lowers through
      `memref.subview + memref.copy` instead of scalarized copy loops
  - structured reduction lowering status:
    - single-axis full-shape `sum` / `min` / `max` now lower to `linalg.fill + linalg.reduce`
    - row-wise max kernels such as the first reduction stage in `example_online_softmax.py`
      are covered by the same structured path
    - additive reduction expressions such as the second reduction stage in
      `example_online_softmax.py` now lower to `linalg.fill + linalg.generic`
  - structured elementwise lowering status:
    - identity-load plus ordered-subsequence broadcast-load expressions now lower to
      `linalg.generic`
    - normalize-style row/column broadcast kernels and the final stage of
      `example_online_softmax.py` are covered by the same structured path
  - rank-reduced slice lowering status:
    - static-1 dimensions can now be dropped through rank-reduced `memref.subview`
    - same-dtype `tl.copy` now lowers equal-extent same-rank slices via
      `memref.subview + memref.copy`
    - same-dtype `tl.copy` now lowers logical-shape-compatible rank-reduced slices via logical
      `memref.subview + memref.copy`
    - mixed-dtype `tl.copy` still falls back to the explicit `scf + memref.load/store` path
    - `tl.gemm` accepts logical 2D operands sliced from higher-rank buffers
    - symbolic batch-count batched slices now lower through the same rank-reduced GEMM path
    - explicit singleton-dim operands such as `(K, 1)` are preserved for GEMV-style
      `tl.gemm`, instead of being rank-reduced away
  - broader completeness work should port the portable expansion set into backend-neutral
    `examples/riscv/` style entry points instead of trying to reuse the original GPU-oriented
    scripts unchanged

### Acceptance

- vector add example runs on local host
- copy example runs on local host
- reduce sum example runs on local host
- reduce max example runs on local host
- matmul example runs on local host
- batched gemm example runs on local host
- dynamic batched gemm example runs on local host
- gemv example runs on local host
- grouped gemm example runs on local host
- dynamic grouped gemm example runs on local host
- each example has a reference NumPy or Torch correctness check
- each example can emit RISC-V `.s` and `.o`

## Phase 6: Deferred Scope

These items are explicitly out of the current completion target.

- dedicated qemu/spike environment validation
- RVV/vector optimization work

## 8. Implementation Details By File

## `tilelang/utils/target.py`

- add `linalg_riscv` to supported targets
- map `riscv` to `linalg_riscv`
- allow strings such as:
  - `linalg_riscv`
  - `riscv`
  - `linalg_riscv -mtriple=riscv64-unknown-linux-gnu`
- keep `auto` behavior unchanged unless you intentionally add RISC-V autodetection later

## `tilelang/engine/phase.py`

- add helper:
  - `is_linalg_riscv_target(target)`
- add:
  - `LowerAndLegalizeForRISCV`
  - `OptimizeForRISCV`
- dispatch from existing `LowerAndLegalize` and `OptimizeForTarget`
- keep RISC-V route isolated from GPU-only passes

## `tilelang/engine/lower.py`

- add a dedicated `linalg_riscv` branch
- do not send `linalg_riscv` through CUDA/HIP/Metal codegen
- provide a return type strategy:
  - stage 1 and 2: return a MLIR-backed artifact or module dump
  - stage 4 and later: return a richer artifact object or an adapter object
- keep existing host/device split intact for old targets

## `src/target/codegen_linalg_riscv.cc`

- implement a visitor-style TIR-to-MLIR codegen on top of real MLIR C++ API
- start with:
  - constants
  - arithmetic
    - `Add/Sub/Mul/Div/Mod/FloorDiv/FloorMod/Min/Max`
  - loop lowering
  - if lowering
  - alloc/load/store
- then add:
  - region/subview
  - `tl.copy`
  - elementwise
  - reduction
  - `tl.gemm`
- add strong diagnostics for unsupported nodes

## `src/target/rt_mod_linalg_riscv.cc`

- register the global build function
- make the runtime module hold exported MLIR artifacts and later LLVM/RISC-V artifacts
- mirror the role of existing runtime modules without copying CUDA-specific behavior

## `tilelang/tladapter/`

- expose the Python-facing MLIR driver surface
- mirror the `npuir` pattern:
  - pass pipeline wrapper
  - parse / verify / serialize helpers
  - future artifact driver entrypoints

## `tilelang/jit/adapter/riscv/`

- isolate external tool invocation here
- keep the rest of the compiler independent from `subprocess`
- centralize:
  - tool detection
  - pipeline building
  - artifact path generation
  - native host `.so` build
  - flattened memref ABI packing for NumPy / ctypes host simulation
  - optional qemu/spike launching

## 9. Tests To Add

Create a new directory `testing/python/riscv`.

### Level 0: target and dispatch

- `test_riscv_target_parse.py`
  - `riscv` alias resolves correctly
  - `linalg_riscv` target is accepted
  - old targets still work

### Level 1: IR structure

- `test_riscv_lower_to_mlir.py`
  - emits valid MLIR
  - contains expected ops for simple kernels
- `test_riscv_copy_codegen.py`
  - validates `subview` or copy-like lowering
- `test_riscv_reduce_codegen.py`
  - checks reduction lowering
- `test_riscv_matmul_codegen.py`
  - checks `linalg.matmul`

### Level 2: artifact export

- `test_riscv_artifact_export.py`
  - `.mlir`, `.ll`, `.s`, `.o` are produced
  - export paths are deterministic
  - native x86 host simulation can run a copy kernel
- `test_riscv_examples.py`
  - the example CLIs run on host
  - the example CLIs emit RISC-V asm/object artifacts
- `test_riscv_jit_runtime.py`
  - `tilelang.compile(..., target="riscv")` runs through the host adapter
- `test_riscv_cache.py`
  - `linalg_riscv` disk cache reloads from cache without recompilation

### Level 3: optional runtime

- `test_riscv_qemu_smoke.py`
  - only run when `qemu-riscv64` or `TILELANG_RISCV_RUNNER` is available
  - exercises one tiny kernel end to end

## 10. Example Set

These examples are the required demo surface for MVP.

### 10.1 MVP Demo Surface

The current MVP examples remain:

- vector add
- copy
- reduce sum
- matmul

### Example A: vector add

Purpose:

- validate elementwise lowering
- check `linalg.generic` or `scf` fallback

Expected ops:

- `linalg.generic` or `scf.for`

### Example B: copy

Purpose:

- validate region/subview handling
- validate `tl.copy`

Expected ops:

- `memref.subview`
- `memref.copy` or tensor slice materialization

### Example C: reduce sum

Purpose:

- validate structured reduction

Expected ops:

- `linalg.reduce` or fallback path

### Example D: matmul

Purpose:

- validate `tl.gemm -> linalg.matmul`

Expected ops:

- `linalg.matmul`

### 10.2 Portable Coverage Matrix From Original `examples/`

Use the original `examples/` directory as a coverage source pool, not as an all-or-nothing
acceptance suite.

Static analysis of the current tree shows that most original examples are GPU-oriented:

- `201` non-test Python examples remain after excluding `test_*.py`, `regression_*.py`,
  and `conftest.py`
- `142` use `T.Kernel`
- `135` use `alloc_shared`
- `132` use `alloc_fragment`
- `121` use `T.Pipelined`
- `105` mention `TMA`

That means "support all original examples unchanged" would imply implementing a much larger
TileLang GPU execution model, not just the current `linalg + memref` backend.

Treat the original examples in three tiers:

- Tier 1: required portable completeness suite
  - `examples/elementwise/example_elementwise_add.py`
  - `examples/gemm/example_gemm.py`
  - `examples/dynamic_shape/example_dynamic.py`
  - `examples/convolution/example_convolution.py`
  - `examples/norm/rms_norm.py`
  - `examples/online_softmax/online_softmax.py`
  - `examples/topk/example_topk.py`
  - `examples/convolution/example_convolution.py`
  - current landed portable ports:
    - `examples/riscv/example_dynamic_shape.py`
    - `examples/riscv/example_rms_norm.py`
    - `examples/riscv/example_online_softmax.py`
    - `examples/riscv/example_topk.py`
    - `examples/riscv/example_convolution.py`
  - additional backend-neutral structured example coverage:
    - `examples/riscv/example_batched_gemm.py`
  - current Tier 2 progress:
    - `examples/riscv/example_gemv.py` now validates singleton-dim `tl.gemm` as a backend-neutral
      GEMV port
    - `examples/riscv/example_grouped_gemm.py` now validates compile-time fixed grouped GEMM as a
      backend-neutral Tier 2 port
    - `examples/riscv/example_dynamic_grouped_gemm.py` now validates runtime offsets/sizes-driven
      dynamic grouped GEMM as a backend-neutral Tier 2 port
  - current status:
    - the Tier 1 portable completeness suite is covered in the current backend-neutral plan
  - `examples/elementwise/example_elementwise_add.py` and `examples/gemm/example_gemm.py`
    are already covered semantically by the current MVP demos, even though the original upstream
    scripts are still GPU-oriented
- Tier 2: later structured extensions
  - `examples/grouped_gemm/example_grouped_gemm_fwd.py`
  - `examples/gemv/example_gemv.py`
  - current landed backend-neutral coverage:
    - `examples/riscv/example_gemv.py`
    - `examples/riscv/example_grouped_gemm.py`
    - `examples/riscv/example_dynamic_grouped_gemm.py`
  - selected sparse/grouped kernels after normalization
- Tier 3: explicit non-goals for the current backend
  - `examples/warp_specialize/`
  - `examples/flash_attention/`
  - `examples/flash_decoding/`
  - `examples/attention_sink/`
  - `examples/hadamard_transform/`
  - `examples/gemm_fp8/`
  - `examples/gemm_sm100/`
  - `examples/dequantize_gemm/`
  - `examples/sparse_tensorcore/`

### 10.3 Porting Rule

For Tier 1 and Tier 2 items, "support" means:

- preserve the algorithm semantics and observable inputs/outputs
- allow a backend-neutral port under `examples/riscv/` or a similar portable directory
- do not require the original GPU-tuned schedule, storage scopes, or pipeline constructs
  to remain unchanged

## 11. Example CLI Contract

Each RISC-V example should support the same flags.

```bash
python examples/riscv/example_vector_add.py --print-tir
python examples/riscv/example_vector_add.py --emit-mlir
python examples/riscv/example_vector_add.py --emit-llvm
python examples/riscv/example_vector_add.py --emit-asm
python examples/riscv/example_vector_add.py --run-host
python examples/riscv/example_vector_add.py --run-qemu
```

Recommended behavior:

- `--print-tir`: print original lowered TIR
- `--emit-mlir`: print and optionally save MLIR
- `--emit-llvm`: save `.ll`
- `--emit-asm`: save `.s`
- `--run-host`: host simulation path if available
- `--run-qemu`: cross-compile and execute under qemu

## 12. Suggested Order Of Real Execution

Do not attempt all features at once. Follow this exact order.

1. make target parsing work
2. make the empty backend compile
3. emit minimal valid MLIR for loops and memory ops
4. make vector add pass
5. make copy pass
6. make reduce sum pass
7. make matmul pass
8. add LLVM/RISC-V artifact export
9. run host simulation
10. expand portable example coverage and structured lowering support

## 13. Common Failure Modes

These are the likely blockers. Check them early.

- accidentally running GPU passes before RISC-V codegen
- trying to recover `linalg` too late from hardware-shaped TIR
- over-supporting irregular regions in MVP
- mixing `memref` and `tensor` values without clear ownership
- making JIT execution a dependency before export pipeline works
- depending on custom dialects too early
- hiding unsupported cases instead of rejecting them clearly

## 14. Minimal Validation Commands

Use these commands as the MVP smoke sequence.

```bash
git switch feat/linalg-riscv-mvp
git submodule update --init --recursive
pip install -e . -v --no-build-isolation

pytest testing/python/riscv/test_riscv_target_parse.py -q
pytest testing/python/riscv/test_riscv_lower_to_mlir.py -q
pytest testing/python/riscv/test_riscv_copy_codegen.py -q
pytest testing/python/riscv/test_riscv_reduce_codegen.py -q
pytest testing/python/riscv/test_riscv_matmul_codegen.py -q
pytest testing/python/riscv/test_riscv_artifact_export.py -q

python examples/riscv/example_vector_add.py --emit-mlir
python examples/riscv/example_copy.py --emit-mlir
python examples/riscv/example_reduce_sum.py --emit-mlir
python examples/riscv/example_reduce_max.py --emit-mlir
python examples/riscv/example_matmul.py --emit-mlir
python examples/riscv/example_batched_gemm.py --emit-mlir
python examples/riscv/example_dynamic_batched_gemm.py --emit-mlir

python examples/riscv/example_vector_add.py --emit-asm
python examples/riscv/example_reduce_max.py --emit-asm
python examples/riscv/example_matmul.py --emit-asm
python examples/riscv/example_batched_gemm.py --emit-asm
python examples/riscv/example_dynamic_batched_gemm.py --emit-asm
python examples/riscv/example_vector_add.py --run-host
python examples/riscv/example_copy.py --run-host
python examples/riscv/example_reduce_sum.py --run-host
python examples/riscv/example_reduce_max.py --run-host
python examples/riscv/example_matmul.py --run-host
python examples/riscv/example_batched_gemm.py --run-host
python examples/riscv/example_dynamic_batched_gemm.py --run-host
```

Optional runtime validation:

```bash
python examples/riscv/example_vector_add.py --run-qemu
python examples/riscv/example_reduce_sum.py --run-qemu
python examples/riscv/example_matmul.py --run-qemu
```

## 15. Final MVP Checklist

- [x] `riscv` target alias exists
- [x] `linalg_riscv` target kind is wired through Python and C++
- [x] RISC-V path uses dedicated lowering phases
- [x] MLIR emission works for loops, conditionals, loads, stores, allocs
- [x] region/subview lowering works
- [x] `tl.copy` works
- [x] elementwise lowering works
- [x] reduction lowering works
- [x] `tl.gemm -> linalg.matmul` works
- [x] MLIR verifies with `mlir-opt`
- [x] LLVM IR export works
- [x] RISC-V asm/object export works
- [x] four examples exist and run through at least MLIR emission
- [x] at least one runner path executes a kernel correctly
- [x] tests are added under `testing/python/riscv`

## 16. Post-MVP Work

Do not include these in the first implementation wave.

- batch matmul
- transpose-aware GEMM legalization
- truly dynamic group-count dispatch beyond compile-time-fixed grouped loops
- vector.contract tuning
- broader JIT runtime integration
- autotuning support for RISC-V
- benchmark suite and performance dashboards
