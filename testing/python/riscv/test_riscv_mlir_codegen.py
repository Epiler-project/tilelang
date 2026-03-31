from __future__ import annotations

import pytest

import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalizeForRISCV, OptimizeForRISCV


def _build_mlir_module(func=None, global_symbol="kernel"):
    if func is None:
        func = tvm.tir.PrimFunc([], tvm.tir.Evaluate(0))
    func = func.with_attr("global_symbol", global_symbol)
    mod = tvm.IRModule({global_symbol: func})
    target = tvm.target.Target("linalg_riscv")
    return tvm.ffi.get_global_func("target.build.tilelang_linalg_riscv")(mod, target)


def _build_mlir_from_source(source: str, global_symbol: str):
    func = tvm.script.from_source(source)
    return _build_mlir_module(func, global_symbol=global_symbol)


def _build_mlir_from_tilelang_prim(func, global_symbol: str):
    func = func.with_attr("global_symbol", global_symbol)
    mod = tvm.IRModule({global_symbol: func})
    target = tvm.target.Target("linalg_riscv")
    mod = LowerAndLegalizeForRISCV(mod, target)
    mod = OptimizeForRISCV(mod, target)
    return tvm.ffi.get_global_func("target.build.tilelang_linalg_riscv")(mod, target)


def _real_mlir_source_or_skip(rt_mod) -> str:
    source = rt_mod.inspect_source()
    if "Placeholder MLIR module" in source:
        pytest.skip("vendored MLIR lowering is disabled in this build")
    return source


def test_riscv_codegen_emits_mlir_module():
    rt_mod = _build_mlir_module()
    source = rt_mod.inspect_source()

    assert source.startswith("module {")
    assert rt_mod.kind == "mlir"

    if "Placeholder MLIR module" in source:
        assert "TILELANG_RISCV_MLIR_MODE=ON" in source
        assert "@kernel" in source
    else:
        assert "func.func @kernel()" in source
        assert "return" in source


def test_riscv_codegen_lowers_copy_loop_to_memref_and_scf():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
@T.prim_func
def copy(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    for i in T.serial(4):
        with T.block("copy"):
            vi = T.axis.spatial(4, i)
            B[vi] = A[vi]
""",
            "copy",
        )
    )

    assert "func.func @copy(%arg0: memref<4xf32>, %arg1: memref<4xf32>)" in source
    assert "scf.for" in source
    assert "memref.load" in source
    assert "memref.store" in source


def test_riscv_codegen_lowers_elementwise_add():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
@T.prim_func
def add(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
    for i in T.serial(4):
        with T.block("add"):
            vi = T.axis.spatial(4, i)
            C[vi] = A[vi] + B[vi]
""",
            "add",
        )
    )

    assert "func.func @add(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>)" in source
    assert "arith.addf" in source
    assert source.count("memref.load") >= 2


def test_riscv_codegen_lowers_scalar_params():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
@T.prim_func
def saxpy(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32"), alpha: T.float32):
    for i in T.serial(4):
        with T.block("saxpy"):
            vi = T.axis.spatial(4, i)
            B[vi] = A[vi] + alpha
""",
            "saxpy",
        )
    )

    assert "func.func @saxpy(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: f32)" in source
    assert "arith.addf" in source


def test_riscv_codegen_lowers_if_then_else():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
@T.prim_func
def if_store(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    for i in T.serial(4):
        with T.block("if_store"):
            vi = T.axis.spatial(4, i)
            if vi < 2:
                B[vi] = A[vi]
""",
            "if_store",
        )
    )

    assert "scf.if" in source
    assert "arith.cmpi slt" in source
    assert "memref.store" in source


def test_riscv_codegen_lowers_alloc_buffer():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
@T.prim_func
def staged_copy(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    C = T.alloc_buffer((4,), dtype="float32")
    for i in T.serial(4):
        with T.block("load"):
            vi = T.axis.spatial(4, i)
            C[vi] = A[vi]
    for i in T.serial(4):
        with T.block("store"):
            vi = T.axis.spatial(4, i)
            B[vi] = C[vi]
""",
            "staged_copy",
        )
    )

    assert "memref.alloca() : memref<4xf32>" in source
    assert source.count("scf.for") == 2


def test_riscv_codegen_lowers_match_buffer_to_subview():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
@T.prim_func
def copy_sub(A: T.Buffer((8,), "float32"), B: T.Buffer((4,), "float32")):
    with T.block("root"):
        T.reads()
        T.writes()
        A0 = T.match_buffer(A[2:6], (4,), dtype="float32")
        for i in T.serial(4):
            with T.block("copy"):
                vi = T.axis.spatial(4, i)
                B[vi] = A0[vi]
""",
            "copy_sub",
        )
    )

    assert "memref.subview" in source
    assert "to memref<4xf32, strided<[1], offset: 2>>" in source
    assert "memref.load %subview" in source


def test_riscv_codegen_lowers_reduce_sum_init_block():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
@T.prim_func
def reduce_sum(A: T.Buffer((4, 8), "float32"), B: T.Buffer((4,), "float32")):
    for i, k in T.grid(4, 8):
        with T.block("sum"):
            vi = T.axis.spatial(4, i)
            vk = T.axis.reduce(8, k)
            with T.init():
                B[vi] = T.float32(0)
            B[vi] = B[vi] + A[vi, vk]
""",
            "reduce_sum",
        )
    )

    assert source.count("scf.for") == 2
    assert "arith.cmpi eq" in source
    assert "arith.constant 0.000000e+00 : f32" in source
    assert "arith.addf" in source


def test_riscv_codegen_lowers_tilelang_copy_kernel():
    @T.prim_func
    def tile_copy(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((4,), "float32")
            T.copy(A, A_shared)
            T.copy(A_shared, B)

    source = _real_mlir_source_or_skip(_build_mlir_from_tilelang_prim(tile_copy, "tile_copy"))

    assert "func.func @tile_copy" in source
    assert source.count("memref.copy") == 2
    assert "memref.alloca() : memref<4xf32>" in source


def test_riscv_codegen_lowers_tilelang_fill_kernel():
    @T.prim_func
    def tile_fill(B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=1):
            tmp = T.alloc_fragment((4,), "float32")
            T.clear(tmp)
            T.copy(tmp, B)

    source = _real_mlir_source_or_skip(_build_mlir_from_tilelang_prim(tile_fill, "tile_fill"))

    assert "func.func @tile_fill" in source
    assert "arith.sitofp" in source or "arith.constant 0.000000e+00 : f32" in source
    assert "scf.for" in source
    assert "memref.store" in source
    assert "memref.copy" in source


def test_riscv_codegen_lowers_tilelang_gemm_to_linalg_matmul():
    @T.prim_func
    def tile_matmul(
        A: T.Tensor((4, 4), "float32"),
        B: T.Tensor((4, 4), "float32"),
        C: T.Tensor((4, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((4, 4), "float32")
            B_shared = T.alloc_shared((4, 4), "float32")
            C_local = T.alloc_fragment((4, 4), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)

    source = _real_mlir_source_or_skip(_build_mlir_from_tilelang_prim(tile_matmul, "tile_matmul"))

    assert "func.func @tile_matmul" in source
    assert "linalg.matmul" in source
    assert source.count("memref.copy") >= 3
    assert "memref.alloca() : memref<4x4xf32>" in source


def test_riscv_codegen_lowers_tilelang_gemm_transpose_b():
    @T.prim_func
    def tile_matmul_transpose_b(
        A: T.Tensor((2, 3), "float32"),
        B: T.Tensor((4, 3), "float32"),
        C: T.Tensor((2, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((2, 3), "float32")
            B_shared = T.alloc_shared((4, 3), "float32")
            C_local = T.alloc_fragment((2, 4), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C)

    source = _real_mlir_source_or_skip(
        _build_mlir_from_tilelang_prim(tile_matmul_transpose_b, "tile_matmul_transpose_b")
    )

    assert "func.func @tile_matmul_transpose_b" in source
    assert "linalg.matmul_transpose_b" in source
    assert "memref<2x3xf32>" in source
    assert "memref<4x3xf32>" in source


def test_riscv_codegen_lowers_tilelang_gemm_transpose_a():
    @T.prim_func
    def tile_matmul_transpose_a(
        A: T.Tensor((3, 2), "float32"),
        B: T.Tensor((3, 4), "float32"),
        C: T.Tensor((2, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((3, 2), "float32")
            B_shared = T.alloc_shared((3, 4), "float32")
            C_local = T.alloc_fragment((2, 4), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_A=True)
            T.copy(C_local, C)

    source = _real_mlir_source_or_skip(
        _build_mlir_from_tilelang_prim(tile_matmul_transpose_a, "tile_matmul_transpose_a")
    )

    assert "func.func @tile_matmul_transpose_a" in source
    assert "linalg.matmul_transpose_a" in source
    assert "memref<3x2xf32>" in source
    assert "memref<3x4xf32>" in source
