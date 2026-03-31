from __future__ import annotations

import numpy as np
import tilelang.language as T

from examples.riscv.common import build_riscv_module, finalize_args, make_parser, maybe_emit_artifacts, maybe_print_tir, run_host, run_qemu


N = 1
C = 2
H = 4
W = 4
F = 3
KH = 3
KW = 3
STRIDE = 1
DILATION = 1
PADDING = 1
PH = H + 2 * PADDING
PW = W + 2 * PADDING
OH = (H + 2 * PADDING - DILATION * (KH - 1) - 1) // STRIDE + 1
OW = (W + 2 * PADDING - DILATION * (KW - 1) - 1) // STRIDE + 1


@T.prim_func
def convolution(
    A: T.Buffer((N, H, W, C), "float32"),
    K: T.Buffer((KH, KW, C, F), "float32"),
    O: T.Buffer((N, OH, OW, F), "float32"),
):
    padded = T.alloc_buffer((N, PH, PW, C), dtype="float32")

    for n, h, w, c in T.grid(N, PH, PW, C):
        with T.block("pad"):
            vn = T.axis.spatial(N, n)
            vh = T.axis.spatial(PH, h)
            vw = T.axis.spatial(PW, w)
            vc = T.axis.spatial(C, c)
            if vh >= PADDING and vh < H + PADDING and vw >= PADDING and vw < W + PADDING:
                padded[vn, vh, vw, vc] = A[vn, vh - PADDING, vw - PADDING, vc]
            else:
                padded[vn, vh, vw, vc] = T.float32(0)

    for n, oh, ow, f, kh, kw, c in T.grid(N, OH, OW, F, KH, KW, C):
        with T.block("conv"):
            vn = T.axis.spatial(N, n)
            voh = T.axis.spatial(OH, oh)
            vow = T.axis.spatial(OW, ow)
            vf = T.axis.spatial(F, f)
            vkh = T.axis.reduce(KH, kh)
            vkw = T.axis.reduce(KW, kw)
            vc = T.axis.reduce(C, c)
            with T.init():
                O[vn, voh, vow, vf] = T.float32(0)
            O[vn, voh, vow, vf] = O[vn, voh, vow, vf] + padded[
                vn,
                voh * STRIDE + vkh * DILATION,
                vow * STRIDE + vkw * DILATION,
                vc,
            ] * K[vkh, vkw, vc, vf]


def reference_convolution(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    padded = np.pad(data, ((0, 0), (PADDING, PADDING), (PADDING, PADDING), (0, 0)))
    out = np.zeros((N, OH, OW, F), dtype=np.float32)
    for n in range(N):
        for oh in range(OH):
            for ow in range(OW):
                for f in range(F):
                    acc = np.float32(0.0)
                    for kh in range(KH):
                        for kw in range(KW):
                            for c in range(C):
                                acc += padded[n, oh * STRIDE + kh * DILATION, ow * STRIDE + kw * DILATION, c] * kernel[kh, kw, c, f]
                    out[n, oh, ow, f] = acc
    return out


def make_inputs() -> tuple[np.ndarray, np.ndarray]:
    data = np.linspace(-1.0, 1.0, num=N * H * W * C, dtype=np.float32).reshape(N, H, W, C)
    kernel = np.linspace(-0.75, 0.75, num=KH * KW * C * F, dtype=np.float32).reshape(KH, KW, C, F)
    return data, kernel


def main(argv: list[str] | None = None) -> int:
    args = finalize_args(make_parser("RISC-V convolution example").parse_args(argv))
    func, rt_mod = build_riscv_module(convolution, "convolution")

    maybe_print_tir(func, args.print_tir)
    maybe_emit_artifacts(rt_mod, "convolution", args)

    if args.run_host:
        data, kernel = make_inputs()
        out = np.zeros((N, OH, OW, F), dtype=np.float32)
        run_host(rt_mod, "convolution", data, kernel, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, reference_convolution(data, kernel), rtol=1e-5, atol=1e-5)
        print("host check passed")

    if args.run_qemu:
        data, kernel = make_inputs()
        out = np.zeros((N, OH, OW, F), dtype=np.float32)
        run_qemu(rt_mod, "convolution", data, kernel, out, output_dir=args.output_dir)
        np.testing.assert_allclose(out, reference_convolution(data, kernel), rtol=1e-5, atol=1e-5)
        print("qemu check passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
