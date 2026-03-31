from __future__ import annotations

import numpy as np
import tilelang.language as T

from examples.riscv.common import build_riscv_module, finalize_args, make_parser, maybe_emit_artifacts, maybe_print_tir, run_host, run_qemu


M = 4
N = 8
LOG2E = 1.4426950408889634
NEG_INF = -1.0e30


@T.prim_func
def online_softmax(
    A: T.Buffer((M, N), "float32"),
    B: T.Buffer((M, N), "float32"),
):
    row_max = T.alloc_buffer((M,), dtype="float32")
    row_sum = T.alloc_buffer((M,), dtype="float32")

    for i, j in T.grid(M, N):
        with T.block("row_max"):
            vi = T.axis.spatial(M, i)
            vj = T.axis.reduce(N, j)
            with T.init():
                row_max[vi] = T.float32(NEG_INF)
            row_max[vi] = T.max(row_max[vi], A[vi, vj])

    for i, j in T.grid(M, N):
        with T.block("row_sum"):
            vi = T.axis.spatial(M, i)
            vj = T.axis.reduce(N, j)
            with T.init():
                row_sum[vi] = T.float32(0)
            row_sum[vi] = row_sum[vi] + T.exp2((A[vi, vj] - row_max[vi]) * T.float32(LOG2E))

    for i, j in T.grid(M, N):
        with T.block("normalize"):
            vi = T.axis.spatial(M, i)
            vj = T.axis.spatial(N, j)
            B[vi, vj] = T.exp2((A[vi, vj] - row_max[vi]) * T.float32(LOG2E)) / row_sum[vi]


def reference_softmax(data: np.ndarray) -> np.ndarray:
    row_max = np.max(data, axis=1, keepdims=True)
    numer = np.exp(data - row_max)
    return numer / np.sum(numer, axis=1, keepdims=True)


def main(argv: list[str] | None = None) -> int:
    args = finalize_args(make_parser("RISC-V online softmax example").parse_args(argv))
    func, rt_mod = build_riscv_module(online_softmax, "online_softmax")

    maybe_print_tir(func, args.print_tir)
    maybe_emit_artifacts(rt_mod, "online_softmax", args)

    if args.run_host:
        data = np.linspace(-2.0, 2.0, num=M * N, dtype=np.float32).reshape(M, N)
        out = np.zeros((M, N), dtype=np.float32)
        run_host(rt_mod, "online_softmax", data, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, reference_softmax(data), rtol=1e-5, atol=1e-5)
        print("host check passed")

    if args.run_qemu:
        data = np.linspace(-2.0, 2.0, num=M * N, dtype=np.float32).reshape(M, N)
        out = np.zeros((M, N), dtype=np.float32)
        run_qemu(rt_mod, "online_softmax", data, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, reference_softmax(data), rtol=1e-5, atol=1e-5)
        print("qemu check passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
