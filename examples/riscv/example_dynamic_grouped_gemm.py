from __future__ import annotations

import numpy as np
import tilelang.language as T

from examples.riscv.common import build_riscv_module, finalize_args, make_parser, maybe_emit_artifacts, maybe_print_tir, run_host, run_qemu


M_TOTAL = T.dynamic("m_total")
K = 4
N = 5


@T.prim_func
def dynamic_grouped_gemm(
    A: T.Tensor((M_TOTAL, K), "float32"),
    B: T.Tensor((2, K, N), "float32"),
    Splits: T.Tensor((2,), "int32"),
    C: T.Tensor((M_TOTAL, N), "float32"),
):
    with T.Kernel(1, threads=1):
        A0 = T.match_buffer(A[0 : Splits[0], 0:K], (Splits[0], K), dtype="float32")
        B0 = T.match_buffer(B[0, 0:K, 0:N], (K, N), dtype="float32")
        C0 = T.match_buffer(C[0 : Splits[0], 0:N], (Splits[0], N), dtype="float32")
        A0_shared = T.alloc_shared((Splits[0], K), "float32")
        B0_shared = T.alloc_shared((K, N), "float32")
        C0_local = T.alloc_fragment((Splits[0], N), "float32")
        T.copy(A0, A0_shared)
        T.copy(B0, B0_shared)
        T.clear(C0_local)
        T.gemm(A0_shared, B0_shared, C0_local)
        T.copy(C0_local, C0)

        A1 = T.match_buffer(A[Splits[0] : Splits[0] + Splits[1], 0:K], (Splits[1], K), dtype="float32")
        B1 = T.match_buffer(B[1, 0:K, 0:N], (K, N), dtype="float32")
        C1 = T.match_buffer(C[Splits[0] : Splits[0] + Splits[1], 0:N], (Splits[1], N), dtype="float32")
        A1_shared = T.alloc_shared((Splits[1], K), "float32")
        B1_shared = T.alloc_shared((K, N), "float32")
        C1_local = T.alloc_fragment((Splits[1], N), "float32")
        T.copy(A1, A1_shared)
        T.copy(B1, B1_shared)
        T.clear(C1_local)
        T.gemm(A1_shared, B1_shared, C1_local)
        T.copy(C1_local, C1)


def make_inputs(group_sizes: tuple[int, int] = (2, 3)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_m = sum(group_sizes)
    lhs = np.arange(total_m * K, dtype=np.float32).reshape(total_m, K)
    rhs = np.arange(2 * K * N, dtype=np.float32).reshape(2, K, N)
    splits = np.asarray(group_sizes, dtype=np.int32)
    return lhs, rhs, splits


def dynamic_grouped_gemm_reference(
    lhs: np.ndarray, rhs: np.ndarray, splits: np.ndarray
) -> np.ndarray:
    first_rows = int(splits[0])
    return np.concatenate((lhs[:first_rows] @ rhs[0], lhs[first_rows:] @ rhs[1]), axis=0)


def main(argv: list[str] | None = None) -> int:
    args = finalize_args(make_parser("RISC-V dynamic grouped gemm example").parse_args(argv))
    func, rt_mod = build_riscv_module(dynamic_grouped_gemm, "dynamic_grouped_gemm")

    maybe_print_tir(func, args.print_tir)
    maybe_emit_artifacts(rt_mod, "dynamic_grouped_gemm", args)

    if args.run_host:
        lhs, rhs, splits = make_inputs()
        out = np.zeros((lhs.shape[0], N), dtype=np.float32)
        run_host(rt_mod, "dynamic_grouped_gemm", lhs, rhs, splits, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, dynamic_grouped_gemm_reference(lhs, rhs, splits))
        print("host check passed")

    if args.run_qemu:
        lhs, rhs, splits = make_inputs()
        out = np.zeros((lhs.shape[0], N), dtype=np.float32)
        run_qemu(rt_mod, "dynamic_grouped_gemm", lhs, rhs, splits, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, dynamic_grouped_gemm_reference(lhs, rhs, splits))
        print("qemu check passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
