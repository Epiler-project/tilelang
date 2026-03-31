from __future__ import annotations

import numpy as np
import tilelang.language as T

from examples.riscv.common import build_riscv_module, finalize_args, make_parser, maybe_emit_artifacts, maybe_print_tir, run_host, run_qemu


BATCH = 2
M = 2
K = 3
N = 4


@T.prim_func
def batched_gemm(
    A: T.Tensor((BATCH, M, K), "float32"),
    B: T.Tensor((BATCH, K, N), "float32"),
    C: T.Tensor((BATCH, M, N), "float32"),
):
    with T.Kernel(1, threads=1):
        A_shared = T.alloc_shared((BATCH, M, K), "float32")
        B_shared = T.alloc_shared((BATCH, K, N), "float32")
        T.copy(A, A_shared)
        T.copy(B, B_shared)
        for b in T.serial(BATCH):
            C_local = T.alloc_fragment((M, N), "float32")
            T.clear(C_local)
            T.gemm(A_shared[b, :, :], B_shared[b, :, :], C_local)
            T.copy(C_local, C[b, :, :])


def make_inputs() -> tuple[np.ndarray, np.ndarray]:
    lhs = np.arange(BATCH * M * K, dtype=np.float32).reshape(BATCH, M, K)
    rhs = np.arange(BATCH * K * N, dtype=np.float32).reshape(BATCH, K, N)
    return lhs, rhs


def main(argv: list[str] | None = None) -> int:
    args = finalize_args(make_parser("RISC-V batched gemm example").parse_args(argv))
    func, rt_mod = build_riscv_module(batched_gemm, "batched_gemm")

    maybe_print_tir(func, args.print_tir)
    maybe_emit_artifacts(rt_mod, "batched_gemm", args)

    if args.run_host:
        lhs, rhs = make_inputs()
        out = np.zeros((BATCH, M, N), dtype=np.float32)
        run_host(rt_mod, "batched_gemm", lhs, rhs, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, lhs @ rhs)
        print("host check passed")

    if args.run_qemu:
        lhs, rhs = make_inputs()
        out = np.zeros((BATCH, M, N), dtype=np.float32)
        run_qemu(rt_mod, "batched_gemm", lhs, rhs, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, lhs @ rhs)
        print("qemu check passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
