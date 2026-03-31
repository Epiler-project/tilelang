"""Helpers for compiling MLIR-backed kernels into native host libraries."""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tilelang.tladapter.toolchain import resolve_tool

from .libgen import emit_llvm_ir, emit_mlir


@dataclass(frozen=True)
class _MemRefParam:
    shape: tuple[int | None, ...]
    dtype: np.dtype
    mlir_type: str

    @property
    def rank(self) -> int:
        return len(self.shape)


@dataclass(frozen=True)
class _ScalarParam:
    dtype_token: str
    ctype: type[ctypes._SimpleCData]
    mlir_type: str


@dataclass(frozen=True)
class _FunctionSignature:
    name: str
    params: tuple[_MemRefParam | _ScalarParam, ...]


def _run_checked(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or "unknown tool failure"
        raise RuntimeError(f"`{' '.join(cmd)}` failed: {message}")
    return proc


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    pairs = {"<": ">", "(": ")", "[": "]", "{": "}"}
    closers = set(pairs.values())

    for char in text:
        if char in pairs:
            depth += 1
        elif char in closers:
            depth -= 1
        if char == delimiter and depth == 0:
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
            continue
        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _dtype_token_to_numpy(dtype_token: str) -> np.dtype:
    if dtype_token == "f16":
        return np.dtype(np.float16)
    if dtype_token == "f32":
        return np.dtype(np.float32)
    if dtype_token == "f64":
        return np.dtype(np.float64)
    if dtype_token == "bf16":
        return np.dtype("bfloat16")
    if dtype_token == "i1":
        return np.dtype(np.bool_)
    if dtype_token == "index":
        return np.dtype(np.int64)
    if dtype_token.startswith("ui"):
        return np.dtype(f"uint{int(dtype_token[2:])}")
    if dtype_token.startswith("i"):
        return np.dtype(f"int{int(dtype_token[1:])}")
    raise TypeError(f"Unsupported MLIR dtype for host wrapper: {dtype_token}")


def _dtype_token_to_ctype(dtype_token: str) -> type[ctypes._SimpleCData]:
    mapping: dict[str, type[ctypes._SimpleCData]] = {
        "f32": ctypes.c_float,
        "f64": ctypes.c_double,
        "i1": ctypes.c_bool,
        "i8": ctypes.c_int8,
        "i16": ctypes.c_int16,
        "i32": ctypes.c_int32,
        "i64": ctypes.c_int64,
        "ui8": ctypes.c_uint8,
        "ui16": ctypes.c_uint16,
        "ui32": ctypes.c_uint32,
        "ui64": ctypes.c_uint64,
        "index": ctypes.c_int64,
    }
    if dtype_token in mapping:
        return mapping[dtype_token]
    raise TypeError(f"Unsupported MLIR scalar dtype for host wrapper: {dtype_token}")


def _parse_memref_type(type_text: str) -> _MemRefParam:
    assert type_text.startswith("memref<") and type_text.endswith(">")
    inner = type_text[len("memref<") : -1].strip()
    shape_and_dtype = _split_top_level(inner)[0]
    dims_and_dtype = [piece.strip() for piece in shape_and_dtype.split("x") if piece.strip()]
    if len(dims_and_dtype) == 1:
        shape: tuple[int | None, ...] = ()
        dtype_token = dims_and_dtype[0]
    else:
        raw_shape = dims_and_dtype[:-1]
        dtype_token = dims_and_dtype[-1]
        shape = tuple(None if dim == "?" else int(dim) for dim in raw_shape)
    return _MemRefParam(shape=shape, dtype=_dtype_token_to_numpy(dtype_token), mlir_type=type_text)


def _parse_param_type(type_text: str) -> _MemRefParam | _ScalarParam:
    normalized = type_text.strip()
    if normalized.startswith("memref<"):
        return _parse_memref_type(normalized)
    return _ScalarParam(dtype_token=normalized, ctype=_dtype_token_to_ctype(normalized), mlir_type=normalized)


def _parse_function_signatures(mlir_source: str) -> list[_FunctionSignature]:
    matches = list(re.finditer(r"func\.func\s+@(?P<name>[\w$.-]+)\((?P<params>[^)]*)\)\s*(?:->\s*[^({]+)?\{", mlir_source))
    signatures: list[_FunctionSignature] = []
    for match in matches:
        params_text = match.group("params").strip()
        params: list[_MemRefParam | _ScalarParam] = []
        if params_text:
            for param_text in _split_top_level(params_text):
                _, type_text = param_text.split(":", maxsplit=1)
                params.append(_parse_param_type(type_text))
        signatures.append(_FunctionSignature(name=match.group("name"), params=tuple(params)))
    return signatures


def _select_signature(mlir_source: str, function_name: str | None) -> _FunctionSignature:
    signatures = _parse_function_signatures(mlir_source)
    if not signatures:
        raise ValueError("No `func.func` definitions found in the MLIR module")
    if function_name is None:
        return signatures[0]
    for signature in signatures:
        if signature.name == function_name:
            return signature
    raise ValueError(f"Function `{function_name}` not found in the MLIR module")


def resolve_host_triple() -> str:
    if os.environ.get("TILELANG_HOST_TRIPLE"):
        return os.environ["TILELANG_HOST_TRIPLE"]
    clang = resolve_tool("clang")
    proc = _run_checked([str(clang), "-dumpmachine"])
    return proc.stdout.strip()


def build_host_shared_library(
    value: Any,
    path: str | os.PathLike[str],
    *,
    triple: str | None = None,
    pipeline=None,
    clang_flags: list[str] | None = None,
) -> Path:
    llvm_ir = emit_llvm_ir(value, pipeline=pipeline)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tilelang-host-compile-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        ll_path = temp_dir_path / f"{out_path.stem}.ll"
        ll_path.write_text(llvm_ir)
        cmd = [
            str(resolve_tool("clang")),
            "-shared",
            "-fPIC",
            "-O2",
            "-Wno-override-module",
            "-x",
            "ir",
            str(ll_path),
            "-o",
            str(out_path),
        ]
        active_triple = triple or resolve_host_triple()
        if active_triple:
            cmd.extend(["-target", active_triple])
        if clang_flags:
            cmd.extend(clang_flags)
        _run_checked(cmd)
    return out_path


class HostKernelLibrary:
    def __init__(
        self,
        library_path: str | os.PathLike[str],
        signature: _FunctionSignature,
        *,
        owned_tempdir: tempfile.TemporaryDirectory[str] | None = None,
    ) -> None:
        self.path = Path(library_path)
        self.signature = signature
        self.function_name = signature.name
        self._owned_tempdir = owned_tempdir
        self._cdll = ctypes.CDLL(str(self.path))
        self._entry = getattr(self._cdll, self.function_name)
        self._entry.argtypes = self._build_argtypes()
        self._entry.restype = None

    def close(self) -> None:
        if self._owned_tempdir is not None:
            self._owned_tempdir.cleanup()
            self._owned_tempdir = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _build_argtypes(self) -> list[type[Any]]:
        argtypes: list[type[Any]] = []
        for spec in self.signature.params:
            if isinstance(spec, _MemRefParam):
                argtypes.extend([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64])
                argtypes.extend([ctypes.c_int64] * spec.rank)
                argtypes.extend([ctypes.c_int64] * spec.rank)
            else:
                argtypes.append(spec.ctype)
        return argtypes

    def _flatten_memref_arg(self, spec: _MemRefParam, value: Any) -> list[Any]:
        array = np.asarray(value)
        if array.dtype != spec.dtype:
            raise TypeError(f"Expected {spec.dtype} for {spec.mlir_type}, but got {array.dtype}")
        if array.ndim != spec.rank:
            raise ValueError(f"Expected rank-{spec.rank} array for {spec.mlir_type}, but got rank {array.ndim}")
        if not array.flags.c_contiguous:
            raise ValueError("Host execution currently requires C-contiguous NumPy arrays")
        for expected_dim, actual_dim in zip(spec.shape, array.shape):
            if expected_dim is not None and expected_dim != actual_dim:
                raise ValueError(
                    f"Expected shape {spec.shape} for {spec.mlir_type}, but got {tuple(int(dim) for dim in array.shape)}"
                )
        ptr = array.ctypes.data
        strides = [stride // array.dtype.itemsize for stride in array.strides]
        flattened = [ctypes.c_void_p(ptr), ctypes.c_void_p(ptr), ctypes.c_int64(0)]
        flattened.extend(ctypes.c_int64(int(dim)) for dim in array.shape)
        flattened.extend(ctypes.c_int64(int(stride)) for stride in strides)
        return flattened

    def _flatten_scalar_arg(self, spec: _ScalarParam, value: Any) -> Any:
        scalar = value.item() if isinstance(value, np.generic) else value
        return spec.ctype(scalar)

    def _prepare_args(self, *args: Any) -> list[Any]:
        if len(args) != len(self.signature.params):
            raise ValueError(f"Expected {len(self.signature.params)} arguments, but got {len(args)}")
        flattened: list[Any] = []
        for spec, value in zip(self.signature.params, args):
            if isinstance(spec, _MemRefParam):
                flattened.extend(self._flatten_memref_arg(spec, value))
            else:
                flattened.append(self._flatten_scalar_arg(spec, value))
        return flattened

    def __call__(self, *args: Any) -> None:
        self._entry(*self._prepare_args(*args))


def load_host_module(
    value: Any,
    *,
    function_name: str | None = None,
    path: str | os.PathLike[str] | None = None,
    triple: str | None = None,
    pipeline=None,
    clang_flags: list[str] | None = None,
) -> HostKernelLibrary:
    mlir_source = emit_mlir(value)
    signature = _select_signature(mlir_source, function_name)
    owned_tempdir: tempfile.TemporaryDirectory[str] | None = None
    if path is None:
        owned_tempdir = tempfile.TemporaryDirectory(prefix="tilelang-host-lib-")
        out_path = Path(owned_tempdir.name) / f"{signature.name}.so"
    else:
        out_path = Path(path)
    build_host_shared_library(
        value,
        out_path,
        triple=triple,
        pipeline=pipeline,
        clang_flags=clang_flags,
    )
    return HostKernelLibrary(out_path, signature, owned_tempdir=owned_tempdir)


def load_host_module_from_binary(
    mlir_source: str,
    path: str | os.PathLike[str],
    *,
    function_name: str | None = None,
) -> HostKernelLibrary:
    signature = _select_signature(mlir_source, function_name)
    return HostKernelLibrary(path, signature)


def run_host(
    value: Any,
    *args: Any,
    function_name: str | None = None,
    triple: str | None = None,
    pipeline=None,
    clang_flags: list[str] | None = None,
) -> None:
    load_host_module(
        value,
        function_name=function_name,
        triple=triple,
        pipeline=pipeline,
        clang_flags=clang_flags,
    )(*args)


__all__ = [
    "HostKernelLibrary",
    "build_host_shared_library",
    "load_host_module_from_binary",
    "load_host_module",
    "resolve_host_triple",
    "run_host",
]
