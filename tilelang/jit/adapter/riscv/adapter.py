from __future__ import annotations

from typing import Any, Callable

import torch
from tvm import tir
from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.base import BaseKernelAdapter
from tilelang.utils.target import determine_target

from .wrapper import HostKernelLibrary, load_host_module


class RiscvKernelAdapter(BaseKernelAdapter):
    host_mod: tvm.IRModule | None = None
    device_mod: tvm.IRModule | None = None
    rt_mod: tvm.runtime.Module | None = None
    host_kernel_source: str | None = None
    device_kernel_source: str | None = None

    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        target: str | Target,
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        host_mod: tvm.IRModule | None = None,
        device_mod: tvm.IRModule | None = None,
        rt_mod: tvm.runtime.Module | None = None,
        host_kernel_source: str | None = None,
        device_kernel_source: str | None = None,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.target = Target.canon_target(determine_target(target))
        self.host_mod = host_mod
        self.device_mod = device_mod
        self.rt_mod = rt_mod
        self.host_kernel_source = host_kernel_source
        self.device_kernel_source = device_kernel_source
        self.verbose = verbose
        self.pass_configs = pass_configs or {}
        self.compile_flags = compile_flags
        self._host_library: HostKernelLibrary | None = None

        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.dynamic_symbolic_map = self._process_dynamic_symbolic()
        self._post_init()

    @property
    def prim_func(self) -> tir.PrimFunc:
        _, func = next(iter(self.ir_module.functions.items()))
        return func

    def _process_dynamic_symbolic(self) -> dict[tir.Var, tuple[int, int]]:
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map: dict[tir.Var, tuple[int, int]] = {}
        self._dynamic_symbolic_name_map: dict[str, tuple[int, int]] = {}
        for i, param in enumerate(params):
            if param not in buffer_map:
                continue
            buffer = buffer_map[param]
            for j, shape in enumerate(buffer.shape):
                if isinstance(shape, tir.Var) and shape not in dynamic_symbolic_map:
                    dynamic_symbolic_map[shape] = (i, j)
                    self._dynamic_symbolic_name_map[shape.name] = (i, j)
        return dynamic_symbolic_map

    def _lookup_dynamic_symbolic(self, value: tir.Var) -> tuple[int, int]:
        if value in self.dynamic_symbolic_map:
            return self.dynamic_symbolic_map[value]
        if value.name in self._dynamic_symbolic_name_map:
            return self._dynamic_symbolic_name_map[value.name]
        raise KeyError(f"Dynamic symbolic variable `{value.name}` not found in kernel signature")

    def _get_host_library(self) -> HostKernelLibrary:
        if self._host_library is None:
            source_value = self.rt_mod if self.rt_mod is not None else self.device_kernel_source
            if source_value is None:
                raise RuntimeError("No MLIR source module is available for host execution")
            self._host_library = load_host_module(source_value)
        return self._host_library

    def close(self) -> None:
        if self._host_library is not None:
            self._host_library.close()
            self._host_library = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _resolve_output_shape(self, param: KernelParam, provided_args: dict[int, Any]) -> list[int]:
        shape: list[int] = []
        for dim in param.shape:
            if isinstance(dim, tir.Var):
                ref_tensor_idx, ref_shape_idx = self._lookup_dynamic_symbolic(dim)
                ref_tensor = provided_args[ref_tensor_idx]
                shape.append(int(ref_tensor.shape[ref_shape_idx]))
            else:
                shape.append(int(dim))
        return shape

    def _normalize_tensor_arg(self, value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor input, but got {type(value)}")
        if value.device.type != "cpu":
            raise ValueError("The current linalg_riscv host runner only supports CPU tensors")
        if not value.is_contiguous():
            value = value.contiguous()
        return value.detach()

    def _convert_torch_func(self) -> Callable[..., Any]:
        param_dtypes = [param.torch_dtype() for param in self.params]

        def func(*inputs: Any):
            expected_inputs = len(self.params) - len(self.result_idx)
            if len(inputs) != expected_inputs:
                raise ValueError(f"Kernel expected {expected_inputs} inputs, but {len(inputs)} were provided")

            provided_args: dict[int, Any] = {}
            input_idx = 0
            for i, param in enumerate(self.params):
                if i in self.result_idx:
                    continue
                value = inputs[input_idx]
                input_idx += 1
                if param.is_scalar():
                    if isinstance(value, torch.Tensor):
                        if value.numel() != 1:
                            raise ValueError("Scalar kernel parameters must be passed as Python scalars or 0-d tensors")
                        value = value.item()
                    provided_args[i] = value
                else:
                    provided_args[i] = self._normalize_tensor_arg(value)

            host_args: list[Any] = []
            for i, param in enumerate(self.params):
                if i in self.result_idx:
                    shape = self._resolve_output_shape(param, provided_args)
                    tensor = torch.empty(shape, dtype=param_dtypes[i], device="cpu")
                    host_args.append(tensor)
                    continue

                host_args.append(provided_args[i])

            native_args = [
                arg.numpy() if isinstance(arg, torch.Tensor) else arg
                for arg in host_args
            ]
            self._get_host_library()(*native_args)

            if len(self.result_idx) == 1:
                return host_args[self.result_idx[0]]
            return [host_args[i] for i in self.result_idx]

        return func

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        if self.rt_mod is not None and hasattr(self.rt_mod, "inspect_source"):
            return self.rt_mod.inspect_source()
        if isinstance(self.device_kernel_source, str):
            return self.device_kernel_source
        return super().get_kernel_source(kernel_only=kernel_only)
