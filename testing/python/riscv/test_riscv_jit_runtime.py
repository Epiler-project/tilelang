from __future__ import annotations

import torch
import tilelang
import tilelang.language as T

import pytest


pytest.importorskip("tilelang.tladapter._native")


N = 8
M_DYNAMIC = T.dynamic("m")
N_DYNAMIC = T.dynamic("n")
K_DYNAMIC = T.dynamic("k")
GROUP_SIZES_GROUPED_GEMM = (2, 3)
GROUP_TOTAL_GROUPED_GEMM = sum(GROUP_SIZES_GROUPED_GEMM)
GROUP_K = 4
GROUP_N = 5
GROUP_TOTAL_DYNAMIC = T.dynamic("group_total_dynamic")
GROUP_COUNT_DYNAMIC = 3


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


@T.prim_func
def tile_dynamic_matmul(
    A: T.Tensor((M_DYNAMIC, K_DYNAMIC), "float32"),
    B: T.Tensor((K_DYNAMIC, N_DYNAMIC), "float32"),
    C: T.Tensor((M_DYNAMIC, N_DYNAMIC), "float32"),
):
    for i, j, kk in T.grid(M_DYNAMIC, N_DYNAMIC, K_DYNAMIC):
        with T.block("matmul"):
            vi = T.axis.spatial(M_DYNAMIC, i)
            vj = T.axis.spatial(N_DYNAMIC, j)
            vk = T.axis.reduce(K_DYNAMIC, kk)
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def test_tilelang_compile_runs_riscv_host_adapter_with_dynamic_shape():
    kernel = tilelang.compile(tile_dynamic_matmul, out_idx=[2], target="riscv")

    lhs0 = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    rhs0 = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out0 = kernel(lhs0, rhs0)

    lhs1 = torch.linspace(-1.0, 1.0, steps=8, dtype=torch.float32).reshape(4, 2)
    rhs1 = torch.linspace(0.25, 2.5, steps=10, dtype=torch.float32).reshape(2, 5)
    out1 = kernel(lhs1, rhs1)

    source = kernel.get_kernel_source()
    kernel.close()

    assert "func.func @tile_dynamic_matmul" in source
    assert "memref<?x?xf32>" in source
    assert "memref.dim" in source
    torch.testing.assert_close(out0, lhs0 @ rhs0)
    torch.testing.assert_close(out1, lhs1 @ rhs1)


@T.prim_func
def tile_dynamic_gemm(
    A: T.Tensor((M_DYNAMIC, K_DYNAMIC), "float32"),
    B: T.Tensor((K_DYNAMIC, N_DYNAMIC), "float32"),
    C: T.Tensor((M_DYNAMIC, N_DYNAMIC), "float32"),
):
    with T.Kernel(1, threads=1):
        A_shared = T.alloc_shared((M_DYNAMIC, K_DYNAMIC), "float32")
        B_shared = T.alloc_shared((K_DYNAMIC, N_DYNAMIC), "float32")
        C_local = T.alloc_fragment((M_DYNAMIC, N_DYNAMIC), "float32")
        T.clear(C_local)
        T.copy(A, A_shared)
        T.copy(B, B_shared)
        T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C)


def test_tilelang_compile_runs_riscv_host_adapter_with_dynamic_tile_gemm():
    kernel = tilelang.compile(tile_dynamic_gemm, out_idx=[2], target="riscv")

    lhs0 = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    rhs0 = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out0 = kernel(lhs0, rhs0)

    lhs1 = torch.linspace(-1.0, 1.0, steps=8, dtype=torch.float32).reshape(4, 2)
    rhs1 = torch.linspace(0.25, 2.5, steps=10, dtype=torch.float32).reshape(2, 5)
    out1 = kernel(lhs1, rhs1)

    source = kernel.get_kernel_source()
    kernel.close()

    assert "func.func @tile_dynamic_gemm" in source
    assert "linalg.matmul" in source
    assert "memref<?x?xf32>" in source
    torch.testing.assert_close(out0, lhs0 @ rhs0)
    torch.testing.assert_close(out1, lhs1 @ rhs1)


@T.prim_func
def reduce_max_rows(
    A: T.Tensor((4, 8), "float32"),
    B: T.Tensor((4,), "float32"),
):
    for i, k in T.grid(4, 8):
        with T.block("max"):
            vi = T.axis.spatial(4, i)
            vk = T.axis.reduce(8, k)
            with T.init():
                B[vi] = T.float32(-1.0e30)
            B[vi] = T.max(B[vi], A[vi, vk])


def test_tilelang_compile_runs_riscv_host_adapter_with_reduce_max():
    kernel = tilelang.compile(reduce_max_rows, out_idx=[1], target="riscv")

    data = torch.linspace(-3.0, 4.5, steps=32, dtype=torch.float32).reshape(4, 8)
    out = kernel(data)

    source = kernel.get_kernel_source()
    kernel.close()

    assert "func.func @reduce_max_rows" in source
    assert "linalg.reduce" in source
    assert "arith.select" in source
    torch.testing.assert_close(out, torch.max(data, dim=1).values)


@T.prim_func
def reduce_exp_sum_rows(
    A: T.Tensor((4, 8), "float32"),
    Bias: T.Tensor((4,), "float32"),
    B: T.Tensor((4,), "float32"),
):
    for i, k in T.grid(4, 8):
        with T.block("sum_exp"):
            vi = T.axis.spatial(4, i)
            vk = T.axis.reduce(8, k)
            with T.init():
                B[vi] = T.float32(0)
            B[vi] = B[vi] + T.exp2(A[vi, vk] - Bias[vi])


def test_tilelang_compile_runs_riscv_host_adapter_with_reduction_expr_generic():
    kernel = tilelang.compile(reduce_exp_sum_rows, out_idx=[2], target="riscv")

    data = torch.linspace(-2.0, 3.0, steps=32, dtype=torch.float32).reshape(4, 8)
    bias = torch.linspace(-0.5, 0.5, steps=4, dtype=torch.float32)
    out = kernel(data, bias)

    source = kernel.get_kernel_source()
    kernel.close()

    assert "func.func @reduce_exp_sum_rows" in source
    assert "linalg.generic" in source
    assert "math.exp2" in source
    torch.testing.assert_close(out, torch.exp2(data - bias.unsqueeze(1)).sum(dim=1))


@T.prim_func
def normalize_with_broadcast(
    A: T.Tensor((4, 8), "float32"),
    RowBias: T.Tensor((4,), "float32"),
    ColScale: T.Tensor((8,), "float32"),
    B: T.Tensor((4, 8), "float32"),
):
    for i, j in T.grid(4, 8):
        with T.block("normalize"):
            vi = T.axis.spatial(4, i)
            vj = T.axis.spatial(8, j)
            B[vi, vj] = (A[vi, vj] - RowBias[vi]) / ColScale[vj]


def test_tilelang_compile_runs_riscv_host_adapter_with_broadcast_elementwise_generic():
    kernel = tilelang.compile(normalize_with_broadcast, out_idx=[3], target="riscv")

    data = torch.linspace(-2.0, 2.0, steps=32, dtype=torch.float32).reshape(4, 8)
    row_bias = torch.linspace(-0.5, 0.5, steps=4, dtype=torch.float32)
    col_scale = torch.linspace(0.5, 2.0, steps=8, dtype=torch.float32)
    out = kernel(data, row_bias, col_scale)

    source = kernel.get_kernel_source()
    kernel.close()

    assert "func.func @normalize_with_broadcast" in source
    assert "linalg.generic" in source
    assert "scf.for" not in source
    torch.testing.assert_close(out, (data - row_bias.unsqueeze(1)) / col_scale.unsqueeze(0))


@T.prim_func
def tile_gemv_vector_operand(
    A: T.Tensor((4, 6), "float32"),
    X: T.Tensor((6,), "float32"),
    Y: T.Tensor((4,), "float32"),
):
    with T.Kernel(1, threads=1):
        A_shared = T.alloc_shared((4, 6), "float32")
        X_shared = T.alloc_shared((6, 1), "float32")
        Y_local = T.alloc_fragment((4, 1), "float32")
        T.copy(A, A_shared)
        T.copy(X, X_shared)
        T.clear(Y_local)
        T.gemm(A_shared, X_shared, Y_local)
        T.copy(Y_local, Y)


def test_tilelang_compile_runs_riscv_host_adapter_with_gemv_shape():
    kernel = tilelang.compile(tile_gemv_vector_operand, out_idx=[2], target="riscv")

    matrix = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    vector = torch.linspace(-1.0, 1.5, steps=6, dtype=torch.float32)
    out = kernel(matrix, vector)

    source = kernel.get_kernel_source()
    kernel.close()

    assert "func.func @tile_gemv_vector_operand" in source
    assert "linalg.matmul" in source
    torch.testing.assert_close(out, matrix @ vector)


@T.macro
def tile_grouped_gemm_step(A, B, C, group_idx, row_offset, group_rows):
    A_group = T.match_buffer(
        A[row_offset : row_offset + group_rows, 0:GROUP_K], (group_rows, GROUP_K), dtype="float32"
    )
    B_group = T.match_buffer(B[group_idx, 0:GROUP_K, 0:GROUP_N], (GROUP_K, GROUP_N), dtype="float32")
    C_group = T.match_buffer(
        C[row_offset : row_offset + group_rows, 0:GROUP_N], (group_rows, GROUP_N), dtype="float32"
    )
    A_shared = T.alloc_shared((group_rows, GROUP_K), "float32")
    B_shared = T.alloc_shared((GROUP_K, GROUP_N), "float32")
    C_local = T.alloc_fragment((group_rows, GROUP_N), "float32")
    T.copy(A_group, A_shared)
    T.copy(B_group, B_shared)
    T.clear(C_local)
    T.gemm(A_shared, B_shared, C_local)
    T.copy(C_local, C_group)


@T.prim_func
def tile_grouped_gemm_portable(
    A: T.Tensor((GROUP_TOTAL_GROUPED_GEMM, GROUP_K), "float32"),
    B: T.Tensor((len(GROUP_SIZES_GROUPED_GEMM), GROUP_K, GROUP_N), "float32"),
    C: T.Tensor((GROUP_TOTAL_GROUPED_GEMM, GROUP_N), "float32"),
):
    with T.Kernel(1, threads=1):
        tile_grouped_gemm_step(A, B, C, 0, 0, GROUP_SIZES_GROUPED_GEMM[0])
        tile_grouped_gemm_step(
            A, B, C, 1, GROUP_SIZES_GROUPED_GEMM[0], GROUP_SIZES_GROUPED_GEMM[1]
        )


def test_tilelang_compile_runs_riscv_host_adapter_with_grouped_gemm():
    kernel = tilelang.compile(tile_grouped_gemm_portable, out_idx=[2], target="riscv")

    lhs = torch.arange(GROUP_TOTAL_GROUPED_GEMM * GROUP_K, dtype=torch.float32).reshape(
        GROUP_TOTAL_GROUPED_GEMM, GROUP_K
    )
    rhs = torch.arange(
        len(GROUP_SIZES_GROUPED_GEMM) * GROUP_K * GROUP_N, dtype=torch.float32
    ).reshape(len(GROUP_SIZES_GROUPED_GEMM), GROUP_K, GROUP_N)
    out = kernel(lhs, rhs)

    refs = []
    row_offset = 0
    for group_idx, group_rows in enumerate(GROUP_SIZES_GROUPED_GEMM):
        refs.append(lhs[row_offset : row_offset + group_rows] @ rhs[group_idx])
        row_offset += group_rows
    ref = torch.cat(refs, dim=0)

    source = kernel.get_kernel_source()
    kernel.close()

    assert "func.func @tile_grouped_gemm_portable" in source
    assert source.count("linalg.matmul") == len(GROUP_SIZES_GROUPED_GEMM)
    assert "memref.subview" in source
    torch.testing.assert_close(out, ref)


@T.prim_func
def tile_dynamic_grouped_gemm(
    A: T.Tensor((GROUP_TOTAL_DYNAMIC, GROUP_K), "float32"),
    B: T.Tensor((GROUP_COUNT_DYNAMIC, GROUP_K, GROUP_N), "float32"),
    Offsets: T.Tensor((GROUP_COUNT_DYNAMIC,), "int32"),
    Sizes: T.Tensor((GROUP_COUNT_DYNAMIC,), "int32"),
    C: T.Tensor((GROUP_TOTAL_DYNAMIC, GROUP_N), "float32"),
):
    with T.Kernel(1, threads=1):
        A0 = T.match_buffer(A[Offsets[0] : Offsets[0] + Sizes[0], 0:GROUP_K], (Sizes[0], GROUP_K), dtype="float32")
        B0 = T.match_buffer(B[0, 0:GROUP_K, 0:GROUP_N], (GROUP_K, GROUP_N), dtype="float32")
        C0 = T.match_buffer(C[Offsets[0] : Offsets[0] + Sizes[0], 0:GROUP_N], (Sizes[0], GROUP_N), dtype="float32")
        A0_shared = T.alloc_shared((Sizes[0], GROUP_K), "float32")
        B0_shared = T.alloc_shared((GROUP_K, GROUP_N), "float32")
        C0_local = T.alloc_fragment((Sizes[0], GROUP_N), "float32")
        T.copy(A0, A0_shared)
        T.copy(B0, B0_shared)
        T.clear(C0_local)
        T.gemm(A0_shared, B0_shared, C0_local)
        T.copy(C0_local, C0)

        A1 = T.match_buffer(A[Offsets[1] : Offsets[1] + Sizes[1], 0:GROUP_K], (Sizes[1], GROUP_K), dtype="float32")
        B1 = T.match_buffer(B[1, 0:GROUP_K, 0:GROUP_N], (GROUP_K, GROUP_N), dtype="float32")
        C1 = T.match_buffer(C[Offsets[1] : Offsets[1] + Sizes[1], 0:GROUP_N], (Sizes[1], GROUP_N), dtype="float32")
        A1_shared = T.alloc_shared((Sizes[1], GROUP_K), "float32")
        B1_shared = T.alloc_shared((GROUP_K, GROUP_N), "float32")
        C1_local = T.alloc_fragment((Sizes[1], GROUP_N), "float32")
        T.copy(A1, A1_shared)
        T.copy(B1, B1_shared)
        T.clear(C1_local)
        T.gemm(A1_shared, B1_shared, C1_local)
        T.copy(C1_local, C1)

        A2 = T.match_buffer(A[Offsets[2] : Offsets[2] + Sizes[2], 0:GROUP_K], (Sizes[2], GROUP_K), dtype="float32")
        B2 = T.match_buffer(B[2, 0:GROUP_K, 0:GROUP_N], (GROUP_K, GROUP_N), dtype="float32")
        C2 = T.match_buffer(C[Offsets[2] : Offsets[2] + Sizes[2], 0:GROUP_N], (Sizes[2], GROUP_N), dtype="float32")
        A2_shared = T.alloc_shared((Sizes[2], GROUP_K), "float32")
        B2_shared = T.alloc_shared((GROUP_K, GROUP_N), "float32")
        C2_local = T.alloc_fragment((Sizes[2], GROUP_N), "float32")
        T.copy(A2, A2_shared)
        T.copy(B2, B2_shared)
        T.clear(C2_local)
        T.gemm(A2_shared, B2_shared, C2_local)
        T.copy(C2_local, C2)


def test_tilelang_compile_runs_riscv_host_adapter_with_dynamic_grouped_gemm():
    kernel = tilelang.compile(tile_dynamic_grouped_gemm, out_idx=[4], target="riscv")

    lhs = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    rhs = torch.arange(GROUP_COUNT_DYNAMIC * GROUP_K * GROUP_N, dtype=torch.float32).reshape(
        GROUP_COUNT_DYNAMIC, GROUP_K, GROUP_N
    )
    offsets = torch.tensor([0, 2, 3], dtype=torch.int32)
    sizes = torch.tensor([2, 1, 3], dtype=torch.int32)
    out = kernel(lhs, rhs, offsets, sizes)

    refs = []
    for group_idx in range(GROUP_COUNT_DYNAMIC):
        start = offsets[group_idx].item()
        size = sizes[group_idx].item()
        refs.append(lhs[start : start + size] @ rhs[group_idx])
    ref = torch.cat(refs, dim=0)

    source = kernel.get_kernel_source()
    kernel.close()

    assert "func.func @tile_dynamic_grouped_gemm" in source
    assert source.count("linalg.matmul") == GROUP_COUNT_DYNAMIC
    assert "memref<3xi32>" in source
    torch.testing.assert_close(out, ref)


@T.prim_func
def tile_batched_gemm_rank_reduced(
    A: T.Tensor((2, 2, 3), "float32"),
    B: T.Tensor((2, 3, 4), "float32"),
    C: T.Tensor((2, 2, 4), "float32"),
):
    with T.Kernel(1, threads=1):
        A_shared = T.alloc_shared((2, 2, 3), "float32")
        B_shared = T.alloc_shared((2, 3, 4), "float32")
        T.copy(A, A_shared)
        T.copy(B, B_shared)
        for b in T.serial(2):
            C_local = T.alloc_fragment((2, 4), "float32")
            T.clear(C_local)
            T.gemm(A_shared[b, :, :], B_shared[b, :, :], C_local)
            T.copy(C_local, C[b, :, :])


def test_tilelang_compile_runs_riscv_host_adapter_with_rank_reduced_tile_gemm():
    kernel = tilelang.compile(tile_batched_gemm_rank_reduced, out_idx=[2], target="riscv")

    lhs = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    rhs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    out = kernel(lhs, rhs)

    source = kernel.get_kernel_source()
    kernel.close()

    assert "func.func @tile_batched_gemm_rank_reduced" in source
    assert source.count("linalg.matmul") == 1
    assert "memref.subview" in source
    torch.testing.assert_close(out, torch.matmul(lhs, rhs))
