from __future__ import annotations

import pytest

from tilelang import tvm


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
