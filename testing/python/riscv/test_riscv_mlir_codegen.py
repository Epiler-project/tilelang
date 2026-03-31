from __future__ import annotations

import pytest

import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalizeForRISCV, OptimizeForRISCV


GROUP_TOTAL_DYNAMIC = T.dynamic("group_total")
GROUP_COUNT_DYNAMIC = 3
GROUP_COUNT_DYNAMIC_SYM = T.dynamic("group_count_dynamic")
BATCH_DYNAMIC = T.dynamic("batch_dynamic")


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
    assert "linalg.generic" in source or "scf.for" in source
    if "linalg.generic" not in source:
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
    assert "linalg.generic" in source
    assert "arith.addf" in source


def test_riscv_codegen_lowers_float_max():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
@T.prim_func
def max_elem(
    A: T.Buffer((4,), "float32"),
    B: T.Buffer((4,), "float32"),
    C: T.Buffer((4,), "float32"),
):
    for i in T.serial(4):
        with T.block("max_elem"):
            vi = T.axis.spatial(4, i)
            C[vi] = T.max(A[vi], B[vi])
""",
            "max_elem",
        )
    )

    assert "func.func @max_elem(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>)" in source
    assert "arith.select" in source
    assert "arith.cmpf ogt" in source


def test_riscv_codegen_lowers_int_min():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
@T.prim_func
def min_elem(
    A: T.Buffer((4,), "int32"),
    B: T.Buffer((4,), "int32"),
    C: T.Buffer((4,), "int32"),
):
    for i in T.serial(4):
        with T.block("min_elem"):
            vi = T.axis.spatial(4, i)
            C[vi] = T.min(A[vi], B[vi])
""",
            "min_elem",
        )
    )

    assert "func.func @min_elem(%arg0: memref<4xi32>, %arg1: memref<4xi32>, %arg2: memref<4xi32>)" in source
    assert "arith.select" in source
    assert "arith.cmpi slt" in source


def test_riscv_codegen_lowers_div_mod_expression_family():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
@T.prim_func
def div_mod_family(
    A: T.Buffer((4,), "int32"),
    B: T.Buffer((4,), "int32"),
    C: T.Buffer((4,), "int32"),
    D: T.Buffer((4,), "int32"),
):
    for i in T.serial(4):
        with T.block("truncmod"):
            vi = T.axis.spatial(4, i)
            B[vi] = T.truncmod(A[vi], 3)
    for i in T.serial(4):
        with T.block("floordiv"):
            vi = T.axis.spatial(4, i)
            C[vi] = T.floordiv(A[vi], 3)
    for i in T.serial(4):
        with T.block("floormod"):
            vi = T.axis.spatial(4, i)
            D[vi] = T.floormod(A[vi], 3)
""",
            "div_mod_family",
        )
    )

    assert "func.func @div_mod_family" in source
    assert "arith.remsi" in source
    assert "arith.floordivsi" in source
    assert "arith.muli" in source
    assert source.count("arith.subi") >= 1


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


def test_riscv_codegen_lowers_unary_math_calls():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
@T.prim_func
def unary_math(
    A: T.Buffer((4,), "float32"),
    B: T.Buffer((4,), "float32"),
    C: T.Buffer((4,), "float32"),
    D: T.Buffer((4,), "float32"),
    E: T.Buffer((4,), "float32"),
):
    for i in T.serial(4):
        with T.block("sqrt"):
            vi = T.axis.spatial(4, i)
            B[vi] = T.sqrt(A[vi])
    for i in T.serial(4):
        with T.block("rsqrt"):
            vi = T.axis.spatial(4, i)
            C[vi] = T.rsqrt(A[vi])
    for i in T.serial(4):
        with T.block("exp2"):
            vi = T.axis.spatial(4, i)
            D[vi] = T.exp2(A[vi])
    for i in T.serial(4):
        with T.block("log2"):
            vi = T.axis.spatial(4, i)
            E[vi] = T.log2(A[vi])
""",
            "unary_math",
        )
    )

    assert "func.func @unary_math" in source
    assert "math.sqrt" in source
    assert "math.rsqrt" in source
    assert "math.exp2" in source
    assert "math.log2" in source


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
    assert source.count("linalg.generic") == 2 or source.count("scf.for") == 2


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
    assert "linalg.generic" in source or "memref.load %subview" in source


def test_riscv_codegen_lowers_dynamic_shape_vars_from_memref_dims():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
m = T.int32(is_size_var=True)
n = T.int32(is_size_var=True)
k = T.int32(is_size_var=True)
@T.prim_func
def dyn_matmul(
    A: T.Buffer((m, k), "float32"),
    B: T.Buffer((k, n), "float32"),
    C: T.Buffer((m, n), "float32"),
):
    for i, j, kk in T.grid(m, n, k):
        with T.block("matmul"):
            vi = T.axis.spatial(m, i)
            vj = T.axis.spatial(n, j)
            vk = T.axis.reduce(k, kk)
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
""",
            "dyn_matmul",
        )
    )

    assert "func.func @dyn_matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>)" in source
    assert source.count("memref.dim") >= 3
    assert "scf.for" in source
    assert "memref.load" in source
    assert "memref.store" in source


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

    assert "linalg.fill" in source
    assert "linalg.reduce" in source
    assert "arith.addf" in source


def test_riscv_codegen_lowers_multi_axis_reduce_sum_init_block():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
@T.prim_func
def reduce_sum_2d(A: T.Buffer((2, 3, 4), "float32"), B: T.Buffer((2,), "float32")):
    for i, j, k in T.grid(2, 3, 4):
        with T.block("sum"):
            vi = T.axis.spatial(2, i)
            vj = T.axis.reduce(3, j)
            vk = T.axis.reduce(4, k)
            with T.init():
                B[vi] = T.float32(0)
            B[vi] = B[vi] + A[vi, vj, vk]
""",
            "reduce_sum_2d",
        )
    )

    assert "func.func @reduce_sum_2d(%arg0: memref<2x3x4xf32>, %arg1: memref<2xf32>)" in source
    assert "linalg.fill" in source
    assert "linalg.reduce" in source
    assert "arith.addf" in source
    assert "scf.for" not in source


def test_riscv_codegen_lowers_reduce_max_init_block():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
@T.prim_func
def reduce_max(A: T.Buffer((4, 8), "float32"), B: T.Buffer((4,), "float32")):
    for i, k in T.grid(4, 8):
        with T.block("max"):
            vi = T.axis.spatial(4, i)
            vk = T.axis.reduce(8, k)
            with T.init():
                B[vi] = T.float32(-1.0e30)
            B[vi] = T.max(B[vi], A[vi, vk])
""",
            "reduce_max",
        )
    )

    assert "linalg.fill" in source
    assert "linalg.reduce" in source
    assert "arith.select" in source
    assert "arith.cmpf ogt" in source


def test_riscv_codegen_lowers_reduce_min_init_block():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
@T.prim_func
def reduce_min(A: T.Buffer((4, 8), "int32"), B: T.Buffer((4,), "int32")):
    for i, k in T.grid(4, 8):
        with T.block("min"):
            vi = T.axis.spatial(4, i)
            vk = T.axis.reduce(8, k)
            with T.init():
                B[vi] = T.int32(2147483647)
            B[vi] = T.min(B[vi], A[vi, vk])
""",
            "reduce_min",
        )
    )

    assert "linalg.fill" in source
    assert "linalg.reduce" in source
    assert "arith.select" in source
    assert "arith.cmpi slt" in source


def test_riscv_codegen_lowers_reduction_expr_to_linalg_generic():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
@T.prim_func
def reduce_exp_sum(
    A: T.Buffer((4, 8), "float32"),
    Bias: T.Buffer((4,), "float32"),
    B: T.Buffer((4,), "float32"),
):
    for i, j in T.grid(4, 8):
        with T.block("sum_exp"):
            vi = T.axis.spatial(4, i)
            vj = T.axis.reduce(8, j)
            with T.init():
                B[vi] = T.float32(0)
            B[vi] = B[vi] + T.exp2(A[vi, vj] - Bias[vi])
""",
            "reduce_exp_sum",
        )
    )

    assert "linalg.fill" in source
    assert "linalg.generic" in source
    assert "math.exp2" in source
    assert "affine_map<(d0, d1) -> (d0, d1)>" in source
    assert "affine_map<(d0, d1) -> (d0)>" in source


def test_riscv_codegen_lowers_broadcast_elementwise_to_linalg_generic():
    source = _real_mlir_source_or_skip(
        _build_mlir_from_source(
            """
# from tvm.script import tir as T
@T.prim_func
def normalize(
    A: T.Buffer((4, 8), "float32"),
    RowBias: T.Buffer((4,), "float32"),
    ColScale: T.Buffer((8,), "float32"),
    B: T.Buffer((4, 8), "float32"),
):
    for i, j in T.grid(4, 8):
        with T.block("normalize"):
            vi = T.axis.spatial(4, i)
            vj = T.axis.spatial(8, j)
            B[vi, vj] = (A[vi, vj] - RowBias[vi]) / ColScale[vj]
""",
            "normalize",
        )
    )

    assert "linalg.generic" in source
    assert "arith.subf" in source
    assert "arith.divf" in source
    assert "affine_map<(d0, d1) -> (d0)>" in source
    assert "affine_map<(d0, d1) -> (d1)>" in source
    assert "scf.for" not in source


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


def test_riscv_codegen_lowers_dynamic_tilelang_copy_kernel_to_memref_copy():
    @T.prim_func
    def tile_dynamic_copy(
        A: T.Tensor((BATCH_DYNAMIC,), "float32"),
        B: T.Tensor((BATCH_DYNAMIC,), "float32"),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((BATCH_DYNAMIC,), "float32")
            T.copy(A, A_shared)
            T.copy(A_shared, B)

    source = _real_mlir_source_or_skip(
        _build_mlir_from_tilelang_prim(tile_dynamic_copy, "tile_dynamic_copy")
    )

    assert "func.func @tile_dynamic_copy(%arg0: memref<?xf32>, %arg1: memref<?xf32>)" in source
    assert source.count("memref.copy") == 2
    assert "memref.subview" in source


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


def test_riscv_codegen_lowers_dynamic_tilelang_gemm_to_linalg_matmul():
    m = T.dynamic("m")
    n = T.dynamic("n")
    k = T.dynamic("k")

    @T.prim_func
    def tile_dynamic_gemm(
        A: T.Tensor((m, k), "float32"),
        B: T.Tensor((k, n), "float32"),
        C: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((m, k), "float32")
            B_shared = T.alloc_shared((k, n), "float32")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)

    source = _real_mlir_source_or_skip(_build_mlir_from_tilelang_prim(tile_dynamic_gemm, "tile_dynamic_gemm"))

    assert "func.func @tile_dynamic_gemm(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>)" in source
    assert "linalg.matmul" in source
    assert "memref.dim" in source
    assert "scf.for" in source
    assert source.count("memref.copy") >= 3
    assert "memref.store" in source


def test_riscv_codegen_lowers_rank_reduced_tilelang_gemm_slices():
    batch = 2
    m = 2
    n = 4
    k = 3

    @T.prim_func
    def tile_batched_gemm(
        A: T.Tensor((batch, m, k), "float32"),
        B: T.Tensor((batch, k, n), "float32"),
        C: T.Tensor((batch, m, n), "float32"),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((batch, m, k), "float32")
            B_shared = T.alloc_shared((batch, k, n), "float32")
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            for b in T.serial(batch):
                C_local = T.alloc_fragment((m, n), "float32")
                T.clear(C_local)
                T.gemm(A_shared[b, :, :], B_shared[b, :, :], C_local)
                T.copy(C_local, C[b, :, :])

    source = _real_mlir_source_or_skip(_build_mlir_from_tilelang_prim(tile_batched_gemm, "tile_batched_gemm"))

    assert "func.func @tile_batched_gemm(%arg0: memref<2x2x3xf32>, %arg1: memref<2x3x4xf32>, %arg2: memref<2x2x4xf32>)" in source
    assert "memref.subview" in source
    assert source.count("linalg.matmul") == 1
    assert "scf.for" in source


def test_riscv_codegen_lowers_dynamic_rank_reduced_tilelang_gemm_slices():
    @T.prim_func
    def tile_dynamic_batched_gemm(
        A: T.Tensor((BATCH_DYNAMIC, 2, 3), "float32"),
        B: T.Tensor((BATCH_DYNAMIC, 3, 4), "float32"),
        C: T.Tensor((BATCH_DYNAMIC, 2, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((BATCH_DYNAMIC, 2, 3), "float32")
            B_shared = T.alloc_shared((BATCH_DYNAMIC, 3, 4), "float32")
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            for b in T.serial(BATCH_DYNAMIC):
                C_local = T.alloc_fragment((2, 4), "float32")
                T.clear(C_local)
                T.gemm(A_shared[b, :, :], B_shared[b, :, :], C_local)
                T.copy(C_local, C[b, :, :])

    source = _real_mlir_source_or_skip(
        _build_mlir_from_tilelang_prim(tile_dynamic_batched_gemm, "tile_dynamic_batched_gemm")
    )

    assert "func.func @tile_dynamic_batched_gemm(%arg0: memref<?x2x3xf32>, %arg1: memref<?x3x4xf32>, %arg2: memref<?x2x4xf32>)" in source
    assert "memref.subview" in source
    assert source.count("linalg.matmul") == 1
    assert "scf.for" in source


def test_riscv_codegen_lowers_rank_reduced_tilelang_copy_to_memref_copy():
    @T.prim_func
    def tile_rank_reduced_copy(
        A: T.Tensor((2, 4, 1), "float32"),
        B: T.Tensor((2, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((2, 4, 1), "float32")
            T.copy(A, A_shared)
            for b in T.serial(2):
                T.copy(A_shared[b, :, :], B[b, :])

    source = _real_mlir_source_or_skip(
        _build_mlir_from_tilelang_prim(tile_rank_reduced_copy, "tile_rank_reduced_copy")
    )

    assert "func.func @tile_rank_reduced_copy(%arg0: memref<2x4x1xf32>, %arg1: memref<2x4xf32>)" in source
    assert "memref.subview" in source
    assert "memref.copy" in source
    assert "scf.for" in source


def test_riscv_codegen_lowers_dynamic_rank_reduced_tilelang_copy_to_memref_copy():
    @T.prim_func
    def tile_dynamic_rank_reduced_copy(
        A: T.Tensor((BATCH_DYNAMIC, 4, 1), "float32"),
        B: T.Tensor((BATCH_DYNAMIC, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((BATCH_DYNAMIC, 4, 1), "float32")
            T.copy(A, A_shared)
            for b in T.serial(BATCH_DYNAMIC):
                T.copy(A_shared[b, :, :], B[b, :])

    source = _real_mlir_source_or_skip(
        _build_mlir_from_tilelang_prim(tile_dynamic_rank_reduced_copy, "tile_dynamic_rank_reduced_copy")
    )

    assert "func.func @tile_dynamic_rank_reduced_copy(%arg0: memref<?x4x1xf32>, %arg1: memref<?x4xf32>)" in source
    assert "memref.subview" in source
    assert "memref.copy" in source
    assert "scf.for" in source


def test_riscv_codegen_lowers_tilelang_gemv_via_singleton_dim_gemm():
    @T.prim_func
    def tile_gemv(
        A: T.Tensor((4, 6), "float32"),
        X: T.Tensor((6,), "float32"),
        Y: T.Tensor((4,), "float32"),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((4, 6), "float32")
            X_shared = T.alloc_shared((6, 1), "float32")
            Y_local = T.alloc_fragment((4, 1), "float32")
            T.copy(A, A_shared)
            T.copy(X, X_shared)
            T.clear(Y_local)
            T.gemm(A_shared, X_shared, Y_local)
            T.copy(Y_local, Y)

    source = _real_mlir_source_or_skip(_build_mlir_from_tilelang_prim(tile_gemv, "tile_gemv"))

    assert "func.func @tile_gemv" in source
    assert "linalg.matmul" in source
    assert "memref<6x1xf32>" in source


def test_riscv_codegen_lowers_portable_grouped_gemm():
    @T.macro
    def grouped_gemm_step(A, B, C, group_idx, row_offset, group_rows):
        A_group = T.match_buffer(A[row_offset : row_offset + group_rows, 0:4], (group_rows, 4), dtype="float32")
        B_group = T.match_buffer(B[group_idx, 0:4, 0:5], (4, 5), dtype="float32")
        C_group = T.match_buffer(C[row_offset : row_offset + group_rows, 0:5], (group_rows, 5), dtype="float32")
        A_shared = T.alloc_shared((group_rows, 4), "float32")
        B_shared = T.alloc_shared((4, 5), "float32")
        C_local = T.alloc_fragment((group_rows, 5), "float32")
        T.copy(A_group, A_shared)
        T.copy(B_group, B_shared)
        T.clear(C_local)
        T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C_group)

    @T.prim_func
    def tile_grouped_gemm(
        A: T.Tensor((5, 4), "float32"),
        B: T.Tensor((2, 4, 5), "float32"),
        C: T.Tensor((5, 5), "float32"),
    ):
        with T.Kernel(1, threads=1):
            grouped_gemm_step(A, B, C, 0, 0, 2)
            grouped_gemm_step(A, B, C, 1, 2, 3)

    source = _real_mlir_source_or_skip(_build_mlir_from_tilelang_prim(tile_grouped_gemm, "tile_grouped_gemm"))

    assert "func.func @tile_grouped_gemm" in source
    assert source.count("linalg.matmul") == 2
    assert source.count("memref.subview") >= 6


def test_riscv_codegen_lowers_dynamic_grouped_gemm():
    @T.prim_func
    def tile_dynamic_grouped_gemm(
        A: T.Tensor((GROUP_TOTAL_DYNAMIC, 4), "float32"),
        B: T.Tensor((GROUP_COUNT_DYNAMIC_SYM, 4, 5), "float32"),
        Offsets: T.Tensor((GROUP_COUNT_DYNAMIC_SYM,), "int32"),
        Sizes: T.Tensor((GROUP_COUNT_DYNAMIC_SYM,), "int32"),
        C: T.Tensor((GROUP_TOTAL_DYNAMIC, 5), "float32"),
    ):
        with T.Kernel(1, threads=1):
            for group_idx in T.serial(GROUP_COUNT_DYNAMIC_SYM):
                A_group = T.match_buffer(
                    A[Offsets[group_idx] : Offsets[group_idx] + Sizes[group_idx], 0:4],
                    (Sizes[group_idx], 4),
                    dtype="float32",
                )
                B_group = T.match_buffer(B[group_idx, 0:4, 0:5], (4, 5), dtype="float32")
                C_group = T.match_buffer(
                    C[Offsets[group_idx] : Offsets[group_idx] + Sizes[group_idx], 0:5],
                    (Sizes[group_idx], 5),
                    dtype="float32",
                )
                A_shared = T.alloc_shared((Sizes[group_idx], 4), "float32")
                B_shared = T.alloc_shared((4, 5), "float32")
                C_local = T.alloc_fragment((Sizes[group_idx], 5), "float32")
                T.copy(A_group, A_shared)
                T.copy(B_group, B_shared)
                T.clear(C_local)
                T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C_group)

    source = _real_mlir_source_or_skip(
        _build_mlir_from_tilelang_prim(tile_dynamic_grouped_gemm, "tile_dynamic_grouped_gemm")
    )

    assert "func.func @tile_dynamic_grouped_gemm" in source
    assert "scf.for" in source
    assert source.count("linalg.matmul") == 1
    assert "memref<?x4xf32>" in source
    assert "memref<?x4x5xf32>" in source
    assert "memref<?xi32>" in source
    assert source.count("memref.load") >= GROUP_COUNT_DYNAMIC * 2


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


def test_riscv_codegen_lowers_tilelang_gemm_transpose_a_and_b():
    @T.prim_func
    def tile_matmul_transpose_ab(
        A: T.Tensor((3, 2), "float32"),
        B: T.Tensor((4, 3), "float32"),
        C: T.Tensor((2, 4), "float32"),
    ):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((3, 2), "float32")
            B_shared = T.alloc_shared((4, 3), "float32")
            C_local = T.alloc_fragment((2, 4), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_A=True, transpose_B=True)
            T.copy(C_local, C)

    source = _real_mlir_source_or_skip(
        _build_mlir_from_tilelang_prim(tile_matmul_transpose_ab, "tile_matmul_transpose_ab")
    )

    assert "func.func @tile_matmul_transpose_ab" in source
    assert "linalg.matmul_transpose_b" in source
    assert source.count("scf.for") >= 2
    assert source.count("memref.store") >= 1
