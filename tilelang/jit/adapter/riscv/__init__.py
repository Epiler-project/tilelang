"""RISC-V adapter scaffolding for the MLIR-backed backend."""

from .adapter import RiscvKernelAdapter
from .libgen import emit_asm, emit_llvm_ir, emit_mlir, emit_object
from .wrapper import (
    HostKernelLibrary,
    RiscvRunnerError,
    RiscvRunnerNotFoundError,
    build_host_shared_library,
    build_qemu_executable,
    load_host_module,
    resolve_riscv_linker,
    resolve_riscv_runner,
    run_host,
    run_qemu,
)

__all__ = [
    "HostKernelLibrary",
    "RiscvKernelAdapter",
    "RiscvRunnerError",
    "RiscvRunnerNotFoundError",
    "build_host_shared_library",
    "build_qemu_executable",
    "emit_asm",
    "emit_llvm_ir",
    "emit_mlir",
    "emit_object",
    "load_host_module",
    "resolve_riscv_linker",
    "resolve_riscv_runner",
    "run_host",
    "run_qemu",
]
