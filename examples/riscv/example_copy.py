from __future__ import annotations

import numpy as np
import tilelang.language as T

from examples.riscv.common import build_riscv_module, finalize_args, make_parser, maybe_emit_artifacts, maybe_print_tir, run_host, run_qemu


N = 8


@T.prim_func
def tile_copy(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
    with T.Kernel(1, threads=1):
        A_shared = T.alloc_shared((N,), "float32")
        T.copy(A, A_shared)
        T.copy(A_shared, B)


def main(argv: list[str] | None = None) -> int:
    args = finalize_args(make_parser("RISC-V TileLang copy example").parse_args(argv))
    func, rt_mod = build_riscv_module(tile_copy, "tile_copy")

    maybe_print_tir(func, args.print_tir)
    maybe_emit_artifacts(rt_mod, "tile_copy", args)

    if args.run_host:
        data = np.arange(N, dtype=np.float32)
        out = np.zeros_like(data)
        run_host(rt_mod, "tile_copy", data, out, output_dir=args.output_dir)
        np.testing.assert_array_equal(out, data)
        print("host check passed")

    if args.run_qemu:
        data = np.arange(N, dtype=np.float32)
        out = np.zeros_like(data)
        run_qemu(rt_mod, "tile_copy", data, out, output_dir=args.output_dir)
        np.testing.assert_array_equal(out, data)
        print("qemu check passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
