from __future__ import annotations

from tilelang.jit.adapter.base import BaseKernelAdapter


class RiscvKernelAdapter(BaseKernelAdapter):
    def _convert_torch_func(self) -> callable:
        raise NotImplementedError(
            "The linalg_riscv backend is wired through the compiler, "
            "but the runtime adapter is not implemented yet."
        )
