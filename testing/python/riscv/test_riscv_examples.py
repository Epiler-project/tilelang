from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytest.importorskip("tilelang.tladapter._native")


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "riscv"
EXAMPLES = (
    "example_vector_add.py",
    "example_copy.py",
    "example_reduce_sum.py",
    "example_reduce_sum_2d.py",
    "example_reduce_max.py",
    "example_matmul.py",
    "example_gemv.py",
    "example_batched_gemm.py",
    "example_dynamic_batched_gemm.py",
    "example_grouped_gemm.py",
    "example_dynamic_grouped_gemm.py",
    "example_dynamic_shape.py",
    "example_rms_norm.py",
    "example_online_softmax.py",
    "example_topk.py",
    "example_convolution.py",
)


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


@pytest.mark.parametrize("example_name", EXAMPLES)
def test_riscv_examples_run_on_host(example_name, tmp_path):
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_ROOT / example_name), "--run-host", "--output-dir", str(tmp_path / example_name)],
        cwd=REPO_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"{example_name} failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    assert "host check passed" in result.stdout


@pytest.mark.parametrize("example_name", EXAMPLES)
def test_riscv_examples_emit_riscv_artifacts(example_name, tmp_path):
    output_dir = tmp_path / example_name
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE_ROOT / example_name),
            "--emit-asm",
            "--emit-object",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"{example_name} failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    asm_candidates = list(output_dir.glob("*.s"))
    obj_candidates = list(output_dir.glob("*.o"))
    if len(asm_candidates) != 1:
        raise AssertionError(f"{example_name} should emit exactly one asm file, found {asm_candidates}")
    if len(obj_candidates) != 1:
        raise AssertionError(f"{example_name} should emit exactly one object file, found {obj_candidates}")
    assert asm_candidates[0].stat().st_size > 0
    assert obj_candidates[0].stat().st_size > 0
