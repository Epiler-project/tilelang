from __future__ import annotations

import numpy as np
import tilelang.language as T

from examples.riscv.common import build_riscv_module, finalize_args, make_parser, maybe_emit_artifacts, maybe_print_tir, run_host, run_qemu


GROUP_SIZES = (2, 3)
GROUP_COUNT = len(GROUP_SIZES)
TOTAL_M = sum(GROUP_SIZES)
K = 4
N = 5


@T.macro
def grouped_gemm_step(A, B, C, group_idx, row_offset, group_rows):
    A_group = T.match_buffer(A[row_offset : row_offset + group_rows, 0:K], (group_rows, K), dtype="float32")
    B_group = T.match_buffer(B[group_idx, 0:K, 0:N], (K, N), dtype="float32")
    C_group = T.match_buffer(C[row_offset : row_offset + group_rows, 0:N], (group_rows, N), dtype="float32")
    A_shared = T.alloc_shared((group_rows, K), "float32")
    B_shared = T.alloc_shared((K, N), "float32")
    C_local = T.alloc_fragment((group_rows, N), "float32")
    T.copy(A_group, A_shared)
    T.copy(B_group, B_shared)
    T.clear(C_local)
    T.gemm(A_shared, B_shared, C_local)
    T.copy(C_local, C_group)


@T.prim_func
def grouped_gemm(
    A: T.Tensor((TOTAL_M, K), "float32"),
    B: T.Tensor((GROUP_COUNT, K, N), "float32"),
    C: T.Tensor((TOTAL_M, N), "float32"),
):
    with T.Kernel(1, threads=1):
        grouped_gemm_step(A, B, C, 0, 0, GROUP_SIZES[0])
        grouped_gemm_step(A, B, C, 1, GROUP_SIZES[0], GROUP_SIZES[1])


def make_inputs() -> tuple[np.ndarray, np.ndarray]:
    lhs = np.arange(TOTAL_M * K, dtype=np.float32).reshape(TOTAL_M, K)
    rhs = np.arange(GROUP_COUNT * K * N, dtype=np.float32).reshape(GROUP_COUNT, K, N)
    return lhs, rhs


def grouped_gemm_reference(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    outputs = []
    row_offset = 0
    for group_idx, group_rows in enumerate(GROUP_SIZES):
        outputs.append(lhs[row_offset : row_offset + group_rows] @ rhs[group_idx])
        row_offset += group_rows
    return np.concatenate(outputs, axis=0)


def main(argv: list[str] | None = None) -> int:
    args = finalize_args(make_parser("RISC-V grouped gemm example").parse_args(argv))
    func, rt_mod = build_riscv_module(grouped_gemm, "grouped_gemm")

    maybe_print_tir(func, args.print_tir)
    maybe_emit_artifacts(rt_mod, "grouped_gemm", args)

    if args.run_host:
        lhs, rhs = make_inputs()
        out = np.zeros((TOTAL_M, N), dtype=np.float32)
        run_host(rt_mod, "grouped_gemm", lhs, rhs, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, grouped_gemm_reference(lhs, rhs))
        print("host check passed")

    if args.run_qemu:
        lhs, rhs = make_inputs()
        out = np.zeros((TOTAL_M, N), dtype=np.float32)
        run_qemu(rt_mod, "grouped_gemm", lhs, rhs, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, grouped_gemm_reference(lhs, rhs))
        print("qemu check passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
