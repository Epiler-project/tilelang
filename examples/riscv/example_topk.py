from __future__ import annotations

import numpy as np
import tilelang.language as T

from examples.riscv.common import build_riscv_module, finalize_args, make_parser, maybe_emit_artifacts, maybe_print_tir, run_host, run_qemu


M = 4
N = 6
TOPK = 3
NEG_INF = -1.0e30


@T.prim_func
def topk(
    A: T.Buffer((M, N), "float32"),
    Values: T.Buffer((M, TOPK), "float32"),
    Indices: T.Buffer((M, TOPK), "int32"),
):
    work = T.alloc_buffer((M, N), dtype="float32")
    max_value = T.alloc_buffer((M,), dtype="float32")
    max_index = T.alloc_buffer((M,), dtype="int32")

    for i, j in T.grid(M, N):
        with T.block("init_work"):
            vi = T.axis.spatial(M, i)
            vj = T.axis.spatial(N, j)
            work[vi, vj] = A[vi, vj]

    for k in T.serial(TOPK):
        for i, j in T.grid(M, N):
            with T.block("max_value"):
                vi = T.axis.spatial(M, i)
                vj = T.axis.reduce(N, j)
                with T.init():
                    max_value[vi] = T.float32(NEG_INF)
                max_value[vi] = T.max(max_value[vi], work[vi, vj])

        for i in T.serial(M):
            with T.block("init_index"):
                vi = T.axis.spatial(M, i)
                max_index[vi] = T.int32(-1)

        for i, j in T.grid(M, N):
            with T.block("select_index"):
                vi = T.axis.spatial(M, i)
                vj = T.axis.spatial(N, j)
                if max_index[vi] < 0 and work[vi, vj] == max_value[vi]:
                    max_index[vi] = vj

        for i in T.serial(M):
            with T.block("write_result"):
                vi = T.axis.spatial(M, i)
                Values[vi, k] = max_value[vi]
                Indices[vi, k] = max_index[vi]

        for i, j in T.grid(M, N):
            with T.block("mask_selected"):
                vi = T.axis.spatial(M, i)
                vj = T.axis.spatial(N, j)
                if vj == max_index[vi]:
                    work[vi, vj] = T.float32(NEG_INF)


def reference_topk(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-data, axis=1)[:, :TOPK]
    values = np.take_along_axis(data, order, axis=1)
    return values.astype(np.float32), order.astype(np.int32)


def make_input() -> np.ndarray:
    return np.array(
        [
            [0.5, 2.0, -1.0, 3.5, 1.25, -0.75],
            [4.0, 1.0, 2.5, -2.0, 0.25, 3.0],
            [-1.5, -0.5, 0.75, 2.25, 1.5, -3.0],
            [5.5, 4.5, 3.5, 2.5, 1.5, 0.5],
        ],
        dtype=np.float32,
    )


def main(argv: list[str] | None = None) -> int:
    args = finalize_args(make_parser("RISC-V top-k example").parse_args(argv))
    func, rt_mod = build_riscv_module(topk, "topk")

    maybe_print_tir(func, args.print_tir)
    maybe_emit_artifacts(rt_mod, "topk", args)

    if args.run_host:
        data = make_input()
        values = np.zeros((M, TOPK), dtype=np.float32)
        indices = np.zeros((M, TOPK), dtype=np.int32)
        run_host(rt_mod, "topk", data, values, indices, output_dir=args.output_dir)
        ref_values, ref_indices = reference_topk(data)
        np.testing.assert_allclose(values, ref_values, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(indices, ref_indices)
        print("host check passed")

    if args.run_qemu:
        data = make_input()
        values = np.zeros((M, TOPK), dtype=np.float32)
        indices = np.zeros((M, TOPK), dtype=np.int32)
        run_qemu(rt_mod, "topk", data, values, indices, output_dir=args.output_dir)
        ref_values, ref_indices = reference_topk(data)
        np.testing.assert_allclose(values, ref_values, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(indices, ref_indices)
        print("qemu check passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
