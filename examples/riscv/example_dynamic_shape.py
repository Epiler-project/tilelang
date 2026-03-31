from __future__ import annotations

import numpy as np
import tilelang.language as T

from examples.riscv.common import build_riscv_module, finalize_args, make_parser, maybe_emit_artifacts, maybe_print_tir, run_host, run_qemu


M = T.dynamic("m")
N = T.dynamic("n")
K = T.dynamic("k")


@T.prim_func
def dynamic_matmul(
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


def make_inputs(m: int = 3, n: int = 4, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    lhs = np.arange(m * k, dtype=np.float32).reshape(m, k)
    rhs = np.arange(k * n, dtype=np.float32).reshape(k, n)
    return lhs, rhs


def main(argv: list[str] | None = None) -> int:
    args = finalize_args(make_parser("RISC-V dynamic-shape matmul example").parse_args(argv))
    func, rt_mod = build_riscv_module(dynamic_matmul, "dynamic_matmul")

    maybe_print_tir(func, args.print_tir)
    maybe_emit_artifacts(rt_mod, "dynamic_matmul", args)

    if args.run_host:
        lhs, rhs = make_inputs()
        out = np.zeros((lhs.shape[0], rhs.shape[1]), dtype=np.float32)
        run_host(rt_mod, "dynamic_matmul", lhs, rhs, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, lhs @ rhs)
        print("host check passed")

    if args.run_qemu:
        lhs, rhs = make_inputs()
        out = np.zeros((lhs.shape[0], rhs.shape[1]), dtype=np.float32)
        run_qemu(rt_mod, "dynamic_matmul", lhs, rhs, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, lhs @ rhs)
        print("qemu check passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
