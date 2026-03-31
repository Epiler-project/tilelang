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


@T.prim_func
def tile_matmul_transpose_b(
    A: T.Tensor((2, 3), "float32"),
    B: T.Tensor((4, 3), "float32"),
    C: T.Tensor((2, 4), "float32"),
):
    with T.Kernel(1, threads=1):
        A_shared = T.alloc_shared((2, 3), "float32")
        B_shared = T.alloc_shared((4, 3), "float32")
        C_local = T.alloc_fragment((2, 4), "float32")
        T.clear(C_local)
        T.copy(A, A_shared)
        T.copy(B, B_shared)
        T.gemm(A_shared, B_shared, C_local, transpose_B=True)
        T.copy(C_local, C)


def test_tilelang_compile_runs_riscv_host_adapter_with_transpose_b_gemm():
    kernel = tilelang.compile(tile_matmul_transpose_b, out_idx=[2], target="riscv")

    lhs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    rhs = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    out = kernel(lhs, rhs)
    kernel.close()

    assert "linalg.matmul_transpose_b" in kernel.get_kernel_source()
    torch.testing.assert_close(out, lhs @ rhs.transpose(0, 1))


@T.prim_func
def tile_matmul_transpose_ab(
    A: T.Tensor((3, 2), "float32"),
    B: T.Tensor((4, 3), "float32"),
    C: T.Tensor((2, 4), "float32"),
):
    with T.Kernel(1, threads=1):
        A_shared = T.alloc_shared((3, 2), "float32")
        B_shared = T.alloc_shared((4, 3), "float32")
        C_local = T.alloc_fragment((2, 4), "float32")
        T.clear(C_local)
        T.copy(A, A_shared)
        T.copy(B, B_shared)
        T.gemm(A_shared, B_shared, C_local, transpose_A=True, transpose_B=True)
        T.copy(C_local, C)


def test_tilelang_compile_runs_riscv_host_adapter_with_transpose_a_and_b_gemm():
    kernel = tilelang.compile(tile_matmul_transpose_ab, out_idx=[2], target="riscv")

    lhs = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    rhs = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    out = kernel(lhs, rhs)
    kernel.close()

    assert "linalg.matmul_transpose_b" in kernel.get_kernel_source()
    torch.testing.assert_close(out, lhs.transpose(0, 1) @ rhs.transpose(0, 1))
