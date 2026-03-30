from __future__ import annotations

from tilelang import tvm


def _build_mlir_source() -> str:
    func = tvm.tir.PrimFunc([], tvm.tir.Evaluate(0)).with_attr("global_symbol", "kernel")
    mod = tvm.IRModule({"kernel": func})
    target = tvm.target.Target("linalg_riscv")
    rt_mod = tvm.ffi.get_global_func("target.build.tilelang_linalg_riscv")(mod, target)
    return rt_mod.inspect_source()


def test_riscv_codegen_emits_mlir_module():
    source = _build_mlir_source()

    assert source.startswith("module {")

    if "Placeholder MLIR module" in source:
        assert "TILELANG_RISCV_MLIR_MODE=ON" in source
        assert "@kernel" in source
    else:
        assert "func.func @kernel()" in source
        assert "return" in source
