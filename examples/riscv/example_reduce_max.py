from __future__ import annotations

import numpy as np
import tilelang.language as T

from examples.riscv.common import build_riscv_module, finalize_args, make_parser, maybe_emit_artifacts, maybe_print_tir, run_host, run_qemu


M = 4
K = 8
NEG_INF = -1.0e30


@T.prim_func
def reduce_max(
    A: T.Buffer((M, K), "float32"),
    B: T.Buffer((M,), "float32"),
):
    for i, k in T.grid(M, K):
        with T.block("max"):
            vi = T.axis.spatial(M, i)
            vk = T.axis.reduce(K, k)
            with T.init():
                B[vi] = T.float32(NEG_INF)
            B[vi] = T.max(B[vi], A[vi, vk])


def main(argv: list[str] | None = None) -> int:
    args = finalize_args(make_parser("RISC-V max reduction example").parse_args(argv))
    func, rt_mod = build_riscv_module(reduce_max, "reduce_max")

    maybe_print_tir(func, args.print_tir)
    maybe_emit_artifacts(rt_mod, "reduce_max", args)

    if args.run_host:
        data = np.linspace(-4.0, 4.0, num=M * K, dtype=np.float32).reshape(M, K)
        out = np.zeros((M,), dtype=np.float32)
        run_host(rt_mod, "reduce_max", data, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, data.max(axis=1))
        print("host check passed")

    if args.run_qemu:
        data = np.linspace(-4.0, 4.0, num=M * K, dtype=np.float32).reshape(M, K)
        out = np.zeros((M,), dtype=np.float32)
        run_qemu(rt_mod, "reduce_max", data, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, data.max(axis=1))
        print("qemu check passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
