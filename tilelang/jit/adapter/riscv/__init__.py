"""RISC-V adapter scaffolding for the MLIR-backed backend."""

from .adapter import RiscvKernelAdapter
from .libgen import emit_asm, emit_llvm_ir, emit_mlir, emit_object
from .wrapper import HostKernelLibrary, build_host_shared_library, load_host_module, run_host

__all__ = [
    "HostKernelLibrary",
    "RiscvKernelAdapter",
    "build_host_shared_library",
    "emit_asm",
    "emit_llvm_ir",
    "emit_mlir",
    "emit_object",
    "load_host_module",
    "run_host",
]
