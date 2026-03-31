from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tilelang.jit.adapter.riscv import resolve_riscv_runner


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = REPO_ROOT / "examples" / "riscv" / "example_vector_add.py"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    ld_parts = [
        str(REPO_ROOT / "build" / "lib"),
        str(REPO_ROOT / "3rdparty" / "llvm-project" / "install" / "lib"),
    ]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    return env


def test_riscv_qemu_smoke_example_is_present():
    assert EXAMPLE.is_file()


def test_riscv_example_vector_add_runs_on_qemu(tmp_path):
    pytest.importorskip("tilelang.tladapter._native")
    if resolve_riscv_runner(required=False) is None:
        pytest.skip("qemu/spike runner not available on this machine")

    output_dir = tmp_path / "vector_add_qemu"
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--run-qemu", "--output-dir", str(output_dir)],
        cwd=REPO_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"qemu smoke failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    assert "qemu check passed" in result.stdout
    assert (output_dir / "vector_add.qemu.elf").is_file()
