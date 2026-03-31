from __future__ import annotations

import numpy as np
import tilelang.language as T

from examples.riscv.common import build_riscv_module, finalize_args, make_parser, maybe_emit_artifacts, maybe_print_tir, run_host, run_qemu


M_TOTAL = T.dynamic("m_total")
GROUP_COUNT = 3
K = 4
N = 5


@T.prim_func
def dynamic_grouped_gemm(
    A: T.Tensor((M_TOTAL, K), "float32"),
    B: T.Tensor((GROUP_COUNT, K, N), "float32"),
    Offsets: T.Tensor((GROUP_COUNT,), "int32"),
    Sizes: T.Tensor((GROUP_COUNT,), "int32"),
    C: T.Tensor((M_TOTAL, N), "float32"),
):
    with T.Kernel(1, threads=1):
        A0 = T.match_buffer(A[Offsets[0] : Offsets[0] + Sizes[0], 0:K], (Sizes[0], K), dtype="float32")
        B0 = T.match_buffer(B[0, 0:K, 0:N], (K, N), dtype="float32")
        C0 = T.match_buffer(C[Offsets[0] : Offsets[0] + Sizes[0], 0:N], (Sizes[0], N), dtype="float32")
        A0_shared = T.alloc_shared((Sizes[0], K), "float32")
        B0_shared = T.alloc_shared((K, N), "float32")
        C0_local = T.alloc_fragment((Sizes[0], N), "float32")
        T.copy(A0, A0_shared)
        T.copy(B0, B0_shared)
        T.clear(C0_local)
        T.gemm(A0_shared, B0_shared, C0_local)
        T.copy(C0_local, C0)

        A1 = T.match_buffer(A[Offsets[1] : Offsets[1] + Sizes[1], 0:K], (Sizes[1], K), dtype="float32")
        B1 = T.match_buffer(B[1, 0:K, 0:N], (K, N), dtype="float32")
        C1 = T.match_buffer(C[Offsets[1] : Offsets[1] + Sizes[1], 0:N], (Sizes[1], N), dtype="float32")
        A1_shared = T.alloc_shared((Sizes[1], K), "float32")
        B1_shared = T.alloc_shared((K, N), "float32")
        C1_local = T.alloc_fragment((Sizes[1], N), "float32")
        T.copy(A1, A1_shared)
        T.copy(B1, B1_shared)
        T.clear(C1_local)
        T.gemm(A1_shared, B1_shared, C1_local)
        T.copy(C1_local, C1)

        A2 = T.match_buffer(A[Offsets[2] : Offsets[2] + Sizes[2], 0:K], (Sizes[2], K), dtype="float32")
        B2 = T.match_buffer(B[2, 0:K, 0:N], (K, N), dtype="float32")
        C2 = T.match_buffer(C[Offsets[2] : Offsets[2] + Sizes[2], 0:N], (Sizes[2], N), dtype="float32")
        A2_shared = T.alloc_shared((Sizes[2], K), "float32")
        B2_shared = T.alloc_shared((K, N), "float32")
        C2_local = T.alloc_fragment((Sizes[2], N), "float32")
        T.copy(A2, A2_shared)
        T.copy(B2, B2_shared)
        T.clear(C2_local)
        T.gemm(A2_shared, B2_shared, C2_local)
        T.copy(C2_local, C2)


def make_inputs(group_sizes: tuple[int, int, int] = (2, 1, 3)) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_m = sum(group_sizes)
    lhs = np.arange(total_m * K, dtype=np.float32).reshape(total_m, K)
    rhs = np.arange(GROUP_COUNT * K * N, dtype=np.float32).reshape(GROUP_COUNT, K, N)
    sizes = np.asarray(group_sizes, dtype=np.int32)
    offsets = np.asarray([0, group_sizes[0], group_sizes[0] + group_sizes[1]], dtype=np.int32)
    return lhs, rhs, offsets, sizes


def dynamic_grouped_gemm_reference(
    lhs: np.ndarray, rhs: np.ndarray, offsets: np.ndarray, sizes: np.ndarray
) -> np.ndarray:
    outputs = []
    for group_idx in range(GROUP_COUNT):
        start = int(offsets[group_idx])
        size = int(sizes[group_idx])
        outputs.append(lhs[start : start + size] @ rhs[group_idx])
    return np.concatenate(outputs, axis=0)


def main(argv: list[str] | None = None) -> int:
    args = finalize_args(make_parser("RISC-V dynamic grouped gemm example").parse_args(argv))
    func, rt_mod = build_riscv_module(dynamic_grouped_gemm, "dynamic_grouped_gemm")

    maybe_print_tir(func, args.print_tir)
    maybe_emit_artifacts(rt_mod, "dynamic_grouped_gemm", args)

    if args.run_host:
        lhs, rhs, offsets, sizes = make_inputs()
        out = np.zeros((lhs.shape[0], N), dtype=np.float32)
        run_host(
            rt_mod,
            "dynamic_grouped_gemm",
            lhs,
            rhs,
            offsets,
            sizes,
            out,
            output_dir=args.output_dir,
        )
        np.testing.assert_allclose(out, dynamic_grouped_gemm_reference(lhs, rhs, offsets, sizes))
        print("host check passed")

    if args.run_qemu:
        lhs, rhs, offsets, sizes = make_inputs()
        out = np.zeros((lhs.shape[0], N), dtype=np.float32)
        run_qemu(
            rt_mod,
            "dynamic_grouped_gemm",
            lhs,
            rhs,
            offsets,
            sizes,
            out,
            output_dir=args.output_dir,
        )
        np.testing.assert_allclose(out, dynamic_grouped_gemm_reference(lhs, rhs, offsets, sizes))
        print("qemu check passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
