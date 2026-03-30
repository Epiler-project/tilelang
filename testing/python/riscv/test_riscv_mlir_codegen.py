from __future__ import annotations

from tilelang import tvm


def _build_mlir_module():
    func = tvm.tir.PrimFunc([], tvm.tir.Evaluate(0)).with_attr("global_symbol", "kernel")
    mod = tvm.IRModule({"kernel": func})
    target = tvm.target.Target("linalg_riscv")
    return tvm.ffi.get_global_func("target.build.tilelang_linalg_riscv")(mod, target)


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
