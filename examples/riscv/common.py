from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalizeForRISCV, OptimizeForRISCV
from tilelang.jit.adapter.riscv import (
    RiscvRunnerError,
    RiscvRunnerNotFoundError,
    emit_asm,
    emit_llvm_ir,
    emit_mlir,
    emit_object,
    load_host_module,
    run_qemu as run_qemu_module,
)
from tilelang.tladapter.toolchain import ToolchainNotFoundError


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--print-tir", action="store_true", help="Print the source TileLang/TIR function")
    parser.add_argument("--emit-mlir", action="store_true", help="Print and optionally save lowered MLIR")
    parser.add_argument("--emit-llvm", action="store_true", help="Print and optionally save lowered LLVM IR")
    parser.add_argument("--emit-asm", action="store_true", help="Print and optionally save RISC-V assembly")
    parser.add_argument("--emit-object", action="store_true", help="Save a RISC-V object file")
    parser.add_argument("--run-host", action="store_true", help="Run the kernel through the native x86 host path")
    parser.add_argument(
        "--run-qemu",
        action="store_true",
        help="Run the kernel through qemu-riscv64 or TILELANG_RISCV_RUNNER",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory used for emitted artifacts")
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    actions = (
        args.print_tir,
        args.emit_mlir,
        args.emit_llvm,
        args.emit_asm,
        args.emit_object,
        args.run_host,
        args.run_qemu,
    )
    if not any(actions):
        args.run_host = True
    return args


def build_riscv_module(func: Any, global_symbol: str):
    func = func.with_attr("global_symbol", global_symbol)
    mod = tvm.IRModule({global_symbol: func})
    target = tvm.target.Target("linalg_riscv")
    mod = LowerAndLegalizeForRISCV(mod, target)
    mod = OptimizeForRISCV(mod, target)
    rt_mod = tvm.ffi.get_global_func("target.build.tilelang_linalg_riscv")(mod, target)
    return func, rt_mod


def ensure_output_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    path.mkdir(parents=True, exist_ok=True)
    return path


def maybe_print_tir(func: Any, enabled: bool) -> None:
    if enabled:
        print(func.script())


def maybe_emit_artifacts(rt_mod: Any, stem: str, args: argparse.Namespace) -> dict[str, Path]:
    output_dir = ensure_output_dir(args.output_dir)
    outputs: dict[str, Path] = {}

    if args.emit_mlir:
        path = output_dir / f"{stem}.mlir" if output_dir is not None else None
        mlir_text = emit_mlir(rt_mod, path)
        if path is None:
            print(mlir_text)
        else:
            print(f"[mlir] {path}")
            outputs["mlir"] = path

    if args.emit_llvm:
        path = output_dir / f"{stem}.ll" if output_dir is not None else None
        llvm_ir = emit_llvm_ir(rt_mod, path)
        if path is None:
            print(llvm_ir)
        else:
            print(f"[llvm] {path}")
            outputs["llvm"] = path

    if args.emit_asm:
        path = output_dir / f"{stem}.s" if output_dir is not None else None
        asm_text = emit_asm(rt_mod, path)
        if path is None:
            print(asm_text)
        else:
            print(f"[asm] {path}")
            outputs["asm"] = path

    if args.emit_object:
        path = output_dir / f"{stem}.o" if output_dir is not None else None
        obj_bytes = emit_object(rt_mod, path)
        if path is None:
            print(f"[object-bytes] {len(obj_bytes)}")
        else:
            print(f"[object] {path}")
            outputs["object"] = path

    return outputs


def run_host(rt_mod: Any, stem: str, *args: Any, output_dir: Path | None = None) -> None:
    active_output_dir = ensure_output_dir(output_dir)
    so_path = active_output_dir / f"{stem}.so" if active_output_dir is not None else None
    library = load_host_module(rt_mod, path=so_path)
    try:
        library(*args)
    finally:
        library.close()


def run_qemu(rt_mod: Any, stem: str, *args: Any, output_dir: Path | None = None) -> None:
    active_output_dir = ensure_output_dir(output_dir)
    exe_path = active_output_dir / f"{stem}.qemu.elf" if active_output_dir is not None else None
    try:
        run_qemu_module(rt_mod, *args, path=exe_path)
    except (RiscvRunnerError, RiscvRunnerNotFoundError, ToolchainNotFoundError) as err:
        raise SystemExit(str(err)) from err
    if exe_path is not None:
        print(f"[qemu] {exe_path}")
