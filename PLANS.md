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
- run a small set of examples for correctness checking
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
- MVP feature set:
  - scalar arithmetic
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
- at least one functional path is runnable:
  - host simulation path, or
  - `qemu-riscv64`, or
  - `spike`
- there are automated tests under `testing/python/riscv/`
- there are runnable examples under `examples/riscv/`

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
- `examples/riscv/example_matmul.py`
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
  - `llvm-dis`
- RISC-V backend enabled in LLVM
- Python package support for the MLIR binding/driver layer
- one runtime path for execution:
  - `qemu-riscv64`, or
  - `spike`

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
qemu-riscv64 --version
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
8. RISC-V runner integration
9. tests and examples

## 7. Phase Plan

### Current status snapshot

- Phase 0 target plumbing is landed
- Phase 1 infrastructure is partially landed:
  - vendored `llvm-project` submodule
  - deterministic build script
  - Python toolchain discovery helpers
  - top-level CMake gating for vendored MLIR
  - minimal C++ MLIR builder that emits `module { func.func ... }`
  - `tilelang.tladapter.Pipeline` backed by vendored `mlir-opt`
  - dedicated `mlir` source runtime module for `linalg_riscv`
- remaining Phase 1 gap:
  - begin real `memref/tensor/linalg/scf` lowering instead of function-name-only scaffolding

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
  - optionally `LegalizeSafeMemoryAccess`
  - optionally `HoistNonRestrictParams`
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
- implement reduction recognition:
  - prefer `linalg.reduce`
  - fallback to `linalg.generic`
  - final fallback to `scf.for`
- add explicit unsupported-op diagnostics:
  - node type
  - op name
  - why unsupported
  - suggested rewrite

### Acceptance

- vector add emits `linalg.generic` or valid `scf.for`
- simple copy emits `memref.subview` or `memref.copy`
- reduce sum emits `linalg.reduce`, `linalg.generic`, or valid `scf` fallback
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
- start with this default pipeline:

```text
canonicalize
cse
one-shot-bufferize
canonicalize
cse
convert-linalg-to-vector
canonicalize
cse
convert-vector-to-scf
convert-scf-to-cf
expand-strided-metadata
finalize-memref-to-llvm
convert-vector-to-llvm
convert-func-to-llvm
reconcile-unrealized-casts
```

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

## Phase 5: Runner Integration And Examples

### Goal

Run a few examples end to end.

### Tasks

- add a lightweight adapter in:
  - `tilelang/jit/adapter/riscv/adapter.py`
  - `tilelang/jit/adapter/riscv/wrapper.py`
- do not aim for full PyTorch runtime integration first
- support two execution modes:
  - host simulation
  - qemu/spike execution
- create examples:
  - `examples/riscv/example_vector_add.py`
  - `examples/riscv/example_copy.py`
  - `examples/riscv/example_reduce_sum.py`
  - `examples/riscv/example_matmul.py`
- each example should support:
  - print TIR
  - print MLIR
  - export asm
  - optional run

### Acceptance

- vector add example runs
- copy example runs
- reduce sum example runs
- matmul example runs
- each example has a reference NumPy or Torch correctness check

## Phase 6: RVV Optimization

### Goal

Improve performance after correctness is stable.

### Tasks

- preserve `vector` ops longer in the pipeline
- prioritize these patterns:
  - `vector.transfer_read`
  - `vector.transfer_write`
  - `vector.contract`
  - `vector.fma`
  - `vector.reduction`
- add pass knobs for:
  - vectorization on or off
  - debug loops path or vector path
  - target CPU and feature strings

### Acceptance

- the same kernels still pass correctness tests
- generated asm shows RVV use on optimized path
- at least one microbenchmark shows improvement over loop fallback

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

### Level 3: optional runtime

- `test_riscv_qemu_smoke.py`
  - only run when `qemu-riscv64` is available
  - exercises one tiny kernel end to end

## 10. Example Set

These examples are the required demo surface for MVP.

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
10. run qemu/spike smoke tests
11. only then start RVV optimization

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
python examples/riscv/example_matmul.py --emit-mlir

python examples/riscv/example_vector_add.py --emit-asm
python examples/riscv/example_matmul.py --emit-asm
```

Optional runtime validation:

```bash
python examples/riscv/example_vector_add.py --run-qemu
python examples/riscv/example_reduce_sum.py --run-qemu
python examples/riscv/example_matmul.py --run-qemu
```

## 15. Final MVP Checklist

- [ ] `riscv` target alias exists
- [ ] `linalg_riscv` target kind is wired through Python and C++
- [ ] RISC-V path uses dedicated lowering phases
- [ ] MLIR emission works for loops, conditionals, loads, stores, allocs
- [ ] region/subview lowering works
- [ ] `tl.copy` works
- [ ] elementwise lowering works
- [ ] reduction lowering works
- [ ] `tl.gemm -> linalg.matmul` works
- [ ] MLIR verifies with `mlir-opt`
- [ ] LLVM IR export works
- [ ] RISC-V asm/object export works
- [ ] four examples exist and run through at least MLIR emission
- [ ] at least one runner path executes a kernel correctly
- [ ] tests are added under `testing/python/riscv`

## 16. Post-MVP Work

Do not include these in the first implementation wave.

- batch matmul
- transpose-aware GEMM legalization
- dynamic shape heavy kernels
- vector.contract tuning
- RVV-specific tiling heuristics
- broader JIT runtime integration
- autotuning support for RISC-V
- benchmark suite and performance dashboards
