from __future__ import annotations

import numpy as np
import tilelang.language as T

from examples.riscv.common import build_riscv_module, finalize_args, make_parser, maybe_emit_artifacts, maybe_print_tir, run_host, run_qemu


M = 4
N = 4
K = 4


@T.prim_func
def tile_matmul(
    A: T.Tensor((M, K), "float32"),
    B: T.Tensor((K, N), "float32"),
    C: T.Tensor((M, N), "float32"),
):
    with T.Kernel(1, threads=1):
        A_shared = T.alloc_shared((M, K), "float32")
        B_shared = T.alloc_shared((K, N), "float32")
        C_local = T.alloc_fragment((M, N), "float32")
        T.clear(C_local)
        T.copy(A, A_shared)
        T.copy(B, B_shared)
        T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C)


def main(argv: list[str] | None = None) -> int:
    args = finalize_args(make_parser("RISC-V TileLang matmul example").parse_args(argv))
    func, rt_mod = build_riscv_module(tile_matmul, "tile_matmul")

    maybe_print_tir(func, args.print_tir)
    maybe_emit_artifacts(rt_mod, "tile_matmul", args)

    if args.run_host:
        lhs = np.arange(M * K, dtype=np.float32).reshape(M, K)
        rhs = np.arange(K * N, dtype=np.float32).reshape(K, N)
        out = np.zeros((M, N), dtype=np.float32)
        run_host(rt_mod, "tile_matmul", lhs, rhs, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, lhs @ rhs)
        print("host check passed")

    if args.run_qemu:
        lhs = np.arange(M * K, dtype=np.float32).reshape(M, K)
        rhs = np.arange(K * N, dtype=np.float32).reshape(K, N)
        out = np.zeros((M, N), dtype=np.float32)
        run_qemu(rt_mod, "tile_matmul", lhs, rhs, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, lhs @ rhs)
        print("qemu check passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
