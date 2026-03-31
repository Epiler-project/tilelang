from __future__ import annotations

import torch
import tilelang
import tilelang.language as T

import pytest


pytest.importorskip("tilelang.tladapter._native")


N = 8


@T.prim_func
def tile_copy(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
    with T.Kernel(1, threads=1):
        A_shared = T.alloc_shared((N,), "float32")
        T.copy(A, A_shared)
        T.copy(A_shared, B)


def test_tilelang_compile_runs_riscv_host_adapter():
    kernel = tilelang.compile(tile_copy, out_idx=[1], target="riscv")

    data = torch.arange(N, dtype=torch.float32)
    out = kernel(data)
    kernel.close()

    assert "func.func @tile_copy" in kernel.get_kernel_source()
    assert "func.func @tile_copy" in kernel.get_host_source()
    torch.testing.assert_close(out, data)
