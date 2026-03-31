from __future__ import annotations

import numpy as np
import tilelang.language as T

from examples.riscv.common import build_riscv_module, finalize_args, make_parser, maybe_emit_artifacts, maybe_print_tir, run_host, run_qemu


M = 4
N = 8
EPS = 1e-6


@T.prim_func
def rms_norm(
    A: T.Buffer((M, N), "float32"),
    B: T.Buffer((M, N), "float32"),
):
    scale = T.alloc_buffer((M,), dtype="float32")

    for i, k in T.grid(M, N):
        with T.block("accum"):
            vi = T.axis.spatial(M, i)
            vk = T.axis.reduce(N, k)
            with T.init():
                scale[vi] = T.float32(0)
            scale[vi] = scale[vi] + A[vi, vk] * A[vi, vk]

    for i in T.serial(M):
        with T.block("factor"):
            vi = T.axis.spatial(M, i)
            scale[vi] = T.rsqrt(scale[vi] / T.float32(N) + T.float32(EPS))

    for i, j in T.grid(M, N):
        with T.block("apply"):
            vi = T.axis.spatial(M, i)
            vj = T.axis.spatial(N, j)
            B[vi, vj] = A[vi, vj] * scale[vi]


def reference_rms_norm(data: np.ndarray) -> np.ndarray:
    mean_square = np.mean(np.square(data), axis=1, keepdims=True)
    return data * np.reciprocal(np.sqrt(mean_square + np.float32(EPS)))


def main(argv: list[str] | None = None) -> int:
    args = finalize_args(make_parser("RISC-V RMSNorm example").parse_args(argv))
    func, rt_mod = build_riscv_module(rms_norm, "rms_norm")

    maybe_print_tir(func, args.print_tir)
    maybe_emit_artifacts(rt_mod, "rms_norm", args)

    if args.run_host:
        data = np.linspace(0.5, 4.0, num=M * N, dtype=np.float32).reshape(M, N)
        out = np.zeros((M, N), dtype=np.float32)
        run_host(rt_mod, "rms_norm", data, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, reference_rms_norm(data), rtol=1e-5, atol=1e-5)
        print("host check passed")

    if args.run_qemu:
        data = np.linspace(0.5, 4.0, num=M * N, dtype=np.float32).reshape(M, N)
        out = np.zeros((M, N), dtype=np.float32)
        run_qemu(rt_mod, "rms_norm", data, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, reference_rms_norm(data), rtol=1e-5, atol=1e-5)
        print("qemu check passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
