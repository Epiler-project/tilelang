# AGENTS.md

## Mission

This repository is a TileLang fork for a new structured backend named `linalg_riscv`.

The intended lowering path is:

```text
TileLang DSL
  -> TVM TIR
  -> structured MLIR (func/arith/scf/tensor/memref/bufferization/linalg/vector)
  -> LLVM
  -> RISC-V / RVV
```

The RISC-V path is additive. It must not regress, replace, or repurpose the existing CUDA, HIP, or Metal backends.

## Source Of Truth

- `DESIGN.md` defines architecture, scope, and invariants.
- `PLANS.md` defines frozen decisions, phase order, file mapping, and MVP definition of done.
- Current code is an implementation snapshot, not the architecture spec.

When code and docs disagree, treat the docs as the intended direction and the code as the current progress snapshot. If you change architecture, scope, or milestone boundaries, update the docs in the same branch.

## Current Snapshot

This file was updated against the repository state on `2026-03-31`.

Implemented now:

- `riscv` normalizes to `linalg_riscv` in `tilelang/utils/target.py`.
- `tilelang/engine/phase.py` and `tilelang/engine/lower.py` route `linalg_riscv` through a dedicated structured MLIR path.
- `src/target/codegen_linalg_riscv.h`
  and `src/target/codegen_linalg_riscv.cc`
  exist as backend entry points, and the vendored-MLIR path now uses real MLIR C++ API construction.
- `src/target/rt_mod_linalg_riscv.cc` registers `target.build.tilelang_linalg_riscv`.
- `cmake/TileLangRISCVMLIR.cmake` supports a real vendored-MLIR path and retains a clearly marked placeholder fallback when MLIR integration is disabled.
- `maint/scripts/build_llvm_mlir.sh` exists and bootstraps a vendored LLVM/MLIR toolchain.
- `tilelang/tladapter/` contains Python-side toolchain discovery plus a pass-pipeline wrapper; `tilelang.tladapter._native` currently exists as a tool-backed compatibility layer over `mlir-opt`.
- Current lowering coverage includes:
  - `PrimFunc` buffer/scalar params -> `func.func` args
  - `BlockRealize/Block`
  - `For -> scf.for`
  - `IfThenElse -> scf.if`
  - `AllocBuffer/BufferRealize/DeclBuffer -> memref.alloca`
  - `BufferLoad/BufferStore -> memref.load/store`
  - constants, casts, arithmetic, comparisons, and `Select`
  - unit `thread_extent` / `T.Kernel(..., threads=1)` shells
  - static compact row-major buffer parameters with explicit strides
  - simple contiguous `match_buffer -> memref.subview`
  - `tl.tileop.copy -> memref.copy` or `scf + memref.load/store` fallback
  - `tl.tileop.fill -> scf + memref.store` fallback
  - identity full-shape elementwise loops -> `linalg.generic`
  - simple full-shape reductions -> structured reduction lowering with fallback paths
  - `tl.tileop.gemm_py -> linalg.matmul` and transpose variants for static 2D matmul
- `tilelang/jit/adapter/riscv/libgen.py` exports `.mlir`, `.ll`, `.s`, and `.o`.
- `tilelang/jit/adapter/riscv/wrapper.py` supports host `.so` build/load plus freestanding RISC-V ELF and runner helpers.
- `tilelang/jit/adapter/riscv/adapter.py` provides `RiscvKernelAdapter`, and `tilelang.compile(..., target="riscv")` routes through it.
- `testing/python/riscv/` now covers target parsing, toolchain discovery, tladapter pipeline execution, MLIR codegen, artifact export, examples, JIT runtime, cache reload, and optional qemu smoke.
- `examples/riscv/` now includes:
  - `example_vector_add.py`
  - `example_copy.py`
  - `example_reduce_sum.py`
  - `example_reduce_max.py`
  - `example_matmul.py`
  - `example_batched_gemm.py`
  - `example_dynamic_shape.py`
  - `example_rms_norm.py`
  - `example_online_softmax.py`
  - `example_topk.py`
  - `example_convolution.py`
- `3rdparty/llvm-project` is present in the repository.
- `3rdparty/llvm-project/install/bin` is present in this checkout.

Not implemented yet:

- A true in-process native MLIR binding for `tilelang.tladapter`; the current `_native.py` compatibility layer shells out to vendored tools.
- Broader region/subview/slice lowering beyond the simple contiguous, compact row-major, and rank-reduced MVP cases.
- Broader `linalg.generic` coverage for irregular slices, broadcasts, and predicated elementwise kernels.
- Broader reduction coverage beyond the current simple full-shape sum/min/max patterns.
- Batched / mixed-shape `tl.gemm` generalization and other non-standard matmul forms.
- Robust qemu/spike validation on every machine; runner coverage remains environment-dependent and optional.
- RVV-oriented vector pipeline and performance tuning in Phase 6.

Working interpretation of phase status:

- Phase 0 is complete.
- Phase 1 is complete.
- Phases 2, 3, 4, and 5 are partially landed.
- Phase 6 has not started as a completed optimization track.

## Guardrails

- Keep `linalg_riscv` isolated from GPU-only lowering passes and GPU runtime assumptions.
- Do not route `linalg_riscv` through CUDA, HIP, or Metal codegen paths.
- Prefer real MLIR C++ API construction over raw string concatenation. String output is only acceptable as a short-lived debug scaffold.
- Keep the public alias `riscv -> linalg_riscv`.
- Preserve the documented environment variable names:
  - `TILELANG_RISCV_LLVM_ROOT`
  - `TILELANG_LLVM_INSTALL_DIR`
  - `TILELANG_RISCV_TRIPLE`
  - `TILELANG_RISCV_CPU`
  - `TILELANG_RISCV_ATTRS`
  - `TILELANG_RISCV_ABI`
  - `TILELANG_RISCV_SYSROOT`
  - `TILELANG_RISCV_RUNNER`
  - `TILELANG_RISCV_RUNNER_FLAGS`
- Unsupported MVP cases must fail with explicit diagnostics. Do not silently emit malformed MLIR.

## MVP Scope

Supported semantics for MVP:

- Scalar arithmetic.
- `For`, `IfThenElse`, `SeqStmt`.
- `AllocBuffer`, `BufferLoad`, `BufferStore`.
- Structured region and slice handling.
- `tl.copy`.
- Structured elementwise and reduction kernels.
- `tl.gemm -> linalg.matmul`.

Explicit non-goals for MVP:

- `T.Pipelined`
- async copy
- TMA
- warp or thread binding
- fragment, shared, or local GPU-specific scopes
- barrier or sync
- custom hardware intrinsics
- FlashAttention-style fused GPU kernels

## Key Files

- `tilelang/utils/target.py`: target parsing and alias normalization.
- `tilelang/engine/phase.py`: structured RISC-V lowering path.
- `tilelang/engine/lower.py`: dispatch into the RISC-V backend.
- `cmake/TileLangRISCVMLIR.cmake`: LLVM/MLIR discovery and placeholder fallback behavior.
- `CMakeLists.txt`: backend source registration and MLIR-related compile definitions.
- `tilelang/tladapter/__init__.py`: Python-facing MLIR/toolchain entry points.
- `tilelang/tladapter/_native.py`: tool-backed `PassPipeline` compatibility layer.
- `tilelang/tladapter/toolchain.py`: Python-side LLVM/MLIR discovery.
- `tilelang/tladapter/utils.py`: pass-pipeline wrapper and helpers.
- `src/target/codegen_linalg_riscv.h`
  and `src/target/codegen_linalg_riscv.cc`: main TIR to MLIR backend entry point.
- `src/target/rt_mod_linalg_riscv.cc`: runtime module registration.
- `tilelang/jit/adapter/riscv/libgen.py`: MLIR -> LLVM -> RISC-V artifact export.
- `tilelang/jit/adapter/riscv/wrapper.py`: host wrapper and freestanding runner support.
- `tilelang/jit/adapter/riscv/adapter.py`: lightweight JIT/runtime adapter for `linalg_riscv`.
- `examples/riscv/`: backend-neutral example surface for the current RISC-V path.
- `testing/python/riscv/`: mandatory regression area for backend progress.

## Recommended Next Order

1. Expand region/subview lowering beyond the current simple contiguous and compact row-major cases.
2. Broaden structured elementwise and reduction lifting while keeping explicit `scf + memref` fallback behavior.
3. Generalize GEMM recognition for batched, mixed-shape, and less regular matmul forms.
4. Stabilize qemu/spike validation in a configured runner environment.
5. Start the optimized `linalg -> vector -> llvm` / RVV path only after correctness coverage is stable.

## Validation Checklist

Current milestone checks:

- Phase 1 is achieved when the vendored-MLIR path is enabled; the placeholder fallback must remain clearly marked as non-feature-complete.
- Phase 4 is partially achieved: deterministic `.mlir`, `.ll`, `.s`, and `.o` export helpers exist and host `.so` build/load works.
- Phase 5 is partially achieved: `RiscvKernelAdapter`, `tilelang.compile(..., target="riscv")`, example CLIs, and optional qemu smoke coverage are wired.
- Phase 6 is not achieved yet: optimized RVV/vector lowering remains future work.

Quick regression commands:

```bash
pytest testing/python/riscv -q
pytest testing/python/riscv/test_riscv_examples.py -q
```

These tests require a built TileLang/TVM Python environment with `tvm_ffi` importable.

## Working Rules For Agents

- Make narrow, phase-aligned changes.
- Do not start RVV optimization before correctness, MLIR validation, and artifact export exist.
- When you land new functionality, add or update tests under `testing/python/riscv/`.
- If a change alters scope, phase status, or file layout, update `PLANS.md` and this file.
- Keep placeholder fallback builds working when the LLVM/MLIR toolchain is unavailable, but do not present the placeholder path as feature complete.
- Treat existing CUDA, HIP, and Metal behavior as regression-sensitive.

## cc-connect

This repository may be driven through `cc-connect`, but secrets must not be stored in the repo.

- Put Feishu bot credentials only in `~/.cc-connect/config.toml`.
- Point the project `work_dir` at `/home/gsh/tilelang-riscv`.
- Prefer `mode = "yolo"` for this project because the agent will likely need shell access, external tools, local builds, and SSH-capable workflows.
- In this workstation setup, `tilelang-riscv` is served by its own dedicated `cc-connect` instance so it can run inside the `tilelang-riscv` conda environment.

Useful operator commands:

```bash
/home/gsh/.cc-connect/bin/start-tilelang-riscv.sh
/home/gsh/.cc-connect/bin/restart-projects.sh
tail -f /home/gsh/.cc-connect/logs/tilelang-riscv.tmux.log
```

## cc-connect Integration

This file is not required for Feishu integration to work, but it is useful for long-term operation through `cc-connect`.

## Scheduled tasks (cron)

Use the dedicated TileLang-RISCV instance config when adding cron jobs:

```bash
cc-connect --config /home/gsh/.cc-connect/instances/tilelang-riscv/config.toml \
  cron add -c "0 10 * * *" \
  --prompt "总结当前 RISC-V 后端进度，并给出下一步计划"
```

## Send Message To Current Chat

Use the dedicated TileLang-RISCV instance config when sending a short message into the current active chat:

```bash
cc-connect --config /home/gsh/.cc-connect/instances/tilelang-riscv/config.toml \
  send -m "short message"
```
