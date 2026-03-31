import os

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel


class TVMFFIKernelCache(KernelCache):
    kernel_lib_path = "executable.so"

    def _save_wrapper_kernel_code_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        host_kernel_path = os.path.join(cache_path, self.host_kernel_path)
        if verbose:
            self.logger.debug(f"Saving wrapped kernel source code to file: {host_kernel_path}")
        KernelCache._safe_write_file(host_kernel_path, "w", lambda file: file.write(kernel.adapter.get_host_source()))

    def _save_so_cubin_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        kernel_lib_path = os.path.join(cache_path, self.kernel_lib_path)
        if hasattr(kernel.adapter, "executable") and kernel.adapter.executable is not None:
            executable = kernel.adapter.executable
            if verbose:
                self.logger.debug(f"Saving kernel executable to file: {executable}")
            KernelCache._safe_write_executable(executable, kernel_lib_path)
            return

        src_lib_path = getattr(kernel.adapter, "libpath", None)
        if src_lib_path is None:
            raise AttributeError("Kernel adapter does not expose either `executable` or `libpath` for cache persistence")
        if verbose:
            self.logger.debug(f"Saving kernel shared library to file: {kernel_lib_path}")
        KernelCache._safe_write_file(kernel_lib_path, "wb", lambda file: file.write(KernelCache._load_binary(src_lib_path)))
