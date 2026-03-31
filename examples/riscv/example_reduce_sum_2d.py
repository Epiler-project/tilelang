from __future__ import annotations

import numpy as np
import tilelang.language as T

from examples.riscv.common import build_riscv_module, finalize_args, make_parser, maybe_emit_artifacts, maybe_print_tir, run_host, run_qemu


BATCH = 2
HEIGHT = 3
WIDTH = 4


@T.prim_func
def reduce_sum_2d(
    A: T.Buffer((BATCH, HEIGHT, WIDTH), "float32"),
    B: T.Buffer((BATCH,), "float32"),
):
    for i, h, w in T.grid(BATCH, HEIGHT, WIDTH):
        with T.block("sum"):
            vi = T.axis.spatial(BATCH, i)
            vh = T.axis.reduce(HEIGHT, h)
            vw = T.axis.reduce(WIDTH, w)
            with T.init():
                B[vi] = T.float32(0)
            B[vi] = B[vi] + A[vi, vh, vw]


def main(argv: list[str] | None = None) -> int:
    args = finalize_args(make_parser("RISC-V multi-axis reduction example").parse_args(argv))
    func, rt_mod = build_riscv_module(reduce_sum_2d, "reduce_sum_2d")

    maybe_print_tir(func, args.print_tir)
    maybe_emit_artifacts(rt_mod, "reduce_sum_2d", args)

    if args.run_host:
        data = np.linspace(-2.0, 5.0, num=BATCH * HEIGHT * WIDTH, dtype=np.float32).reshape(
            BATCH, HEIGHT, WIDTH
        )
        out = np.zeros((BATCH,), dtype=np.float32)
        run_host(rt_mod, "reduce_sum_2d", data, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, data.sum(axis=(1, 2)), rtol=1e-6, atol=1e-6)
        print("host check passed")

    if args.run_qemu:
        data = np.linspace(-2.0, 5.0, num=BATCH * HEIGHT * WIDTH, dtype=np.float32).reshape(
            BATCH, HEIGHT, WIDTH
        )
        out = np.zeros((BATCH,), dtype=np.float32)
        run_qemu(rt_mod, "reduce_sum_2d", data, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, data.sum(axis=(1, 2)), rtol=1e-6, atol=1e-6)
        print("qemu check passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
