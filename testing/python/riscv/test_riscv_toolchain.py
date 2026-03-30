from __future__ import annotations

import os
from pathlib import Path

from tilelang.tladapter.toolchain import (
    ToolchainNotFoundError,
    resolve_llvm_dir,
    resolve_llvm_root,
    resolve_mlir_dir,
    resolve_tool,
)


def _make_fake_toolchain(root: Path) -> None:
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "lib" / "cmake" / "mlir").mkdir(parents=True, exist_ok=True)
    (root / "lib" / "cmake" / "llvm").mkdir(parents=True, exist_ok=True)
    for tool_name in ("mlir-opt", "mlir-translate", "llc", "clang"):
        tool_path = root / "bin" / tool_name
        tool_path.write_text("#!/usr/bin/env bash\n")
        tool_path.chmod(0o755)


def test_toolchain_env_root_wins(monkeypatch, tmp_path):
    fake_root = tmp_path / "llvm-install"
    _make_fake_toolchain(fake_root)
    monkeypatch.setenv("TILELANG_RISCV_LLVM_ROOT", str(fake_root))
    monkeypatch.delenv("TILELANG_LLVM_INSTALL_DIR", raising=False)

    assert resolve_llvm_root() == fake_root.resolve()
    assert resolve_mlir_dir() == (fake_root / "lib" / "cmake" / "mlir").resolve()
    assert resolve_llvm_dir() == (fake_root / "lib" / "cmake" / "llvm").resolve()
    assert resolve_tool("mlir-opt") == (fake_root / "bin" / "mlir-opt").resolve()


def test_tool_lookup_falls_back_to_path(monkeypatch, tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    fake_tool = fake_bin / "mlir-opt"
    fake_tool.write_text("#!/usr/bin/env bash\n")
    fake_tool.chmod(0o755)

    monkeypatch.delenv("TILELANG_RISCV_LLVM_ROOT", raising=False)
    monkeypatch.delenv("TILELANG_LLVM_INSTALL_DIR", raising=False)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    assert resolve_tool("mlir-opt") == fake_tool.resolve()


def test_missing_toolchain_raises(monkeypatch):
    monkeypatch.delenv("TILELANG_RISCV_LLVM_ROOT", raising=False)
    monkeypatch.delenv("TILELANG_LLVM_INSTALL_DIR", raising=False)

    try:
        resolve_llvm_root(required=True)
    except ToolchainNotFoundError:
        return

    raise AssertionError("resolve_llvm_root should have raised ToolchainNotFoundError")
