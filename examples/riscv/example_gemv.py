from __future__ import annotations

import numpy as np
import tilelang.language as T

from examples.riscv.common import build_riscv_module, finalize_args, make_parser, maybe_emit_artifacts, maybe_print_tir, run_host, run_qemu


M = 4
K = 6


@T.prim_func
def gemv(
    A: T.Tensor((M, K), "float32"),
    X: T.Tensor((K,), "float32"),
    Y: T.Tensor((M,), "float32"),
):
    with T.Kernel(1, threads=1):
        A_shared = T.alloc_shared((M, K), "float32")
        X_shared = T.alloc_shared((K, 1), "float32")
        Y_local = T.alloc_fragment((M, 1), "float32")
        T.copy(A, A_shared)
        T.copy(X, X_shared)
        T.clear(Y_local)
        T.gemm(A_shared, X_shared, Y_local)
        T.copy(Y_local, Y)


def make_inputs() -> tuple[np.ndarray, np.ndarray]:
    matrix = np.arange(M * K, dtype=np.float32).reshape(M, K)
    vector = np.linspace(-1.0, 1.5, num=K, dtype=np.float32)
    return matrix, vector


def main(argv: list[str] | None = None) -> int:
    args = finalize_args(make_parser("RISC-V GEMV example").parse_args(argv))
    func, rt_mod = build_riscv_module(gemv, "gemv")

    maybe_print_tir(func, args.print_tir)
    maybe_emit_artifacts(rt_mod, "gemv", args)

    if args.run_host:
        matrix, vector = make_inputs()
        out = np.zeros((M,), dtype=np.float32)
        run_host(rt_mod, "gemv", matrix, vector, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, matrix @ vector)
        print("host check passed")

    if args.run_qemu:
        matrix, vector = make_inputs()
        out = np.zeros((M,), dtype=np.float32)
        run_qemu(rt_mod, "gemv", matrix, vector, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, matrix @ vector)
        print("qemu check passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
