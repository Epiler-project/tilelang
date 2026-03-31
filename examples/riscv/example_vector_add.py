from __future__ import annotations

import numpy as np
import tilelang.language as T

from examples.riscv.common import build_riscv_module, finalize_args, make_parser, maybe_emit_artifacts, maybe_print_tir, run_host, run_qemu


N = 8


@T.prim_func
def vector_add(
    A: T.Buffer((N,), "float32"),
    B: T.Buffer((N,), "float32"),
    C: T.Buffer((N,), "float32"),
):
    for i in T.serial(N):
        with T.block("add"):
            vi = T.axis.spatial(N, i)
            C[vi] = A[vi] + B[vi]


def main(argv: list[str] | None = None) -> int:
    args = finalize_args(make_parser("RISC-V vector-add example").parse_args(argv))
    func, rt_mod = build_riscv_module(vector_add, "vector_add")

    maybe_print_tir(func, args.print_tir)
    maybe_emit_artifacts(rt_mod, "vector_add", args)

    if args.run_host:
        lhs = np.arange(N, dtype=np.float32)
        rhs = np.linspace(1.0, float(N), N, dtype=np.float32)
        out = np.zeros_like(lhs)
        run_host(rt_mod, "vector_add", lhs, rhs, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, lhs + rhs)
        print("host check passed")

    if args.run_qemu:
        lhs = np.arange(N, dtype=np.float32)
        rhs = np.linspace(1.0, float(N), N, dtype=np.float32)
        out = np.zeros_like(lhs)
        run_qemu(rt_mod, "vector_add", lhs, rhs, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, lhs + rhs)
        print("qemu check passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
