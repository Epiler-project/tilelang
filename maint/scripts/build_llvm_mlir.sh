#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLVM_PROJECT_DIR="${TILELANG_LLVM_PROJECT_DIR:-${ROOT_DIR}/3rdparty/llvm-project}"
LLVM_BUILD_DIR="${TILELANG_LLVM_BUILD_DIR:-${LLVM_PROJECT_DIR}/build-host}"
LLVM_INSTALL_DIR="${TILELANG_LLVM_INSTALL_DIR:-${LLVM_PROJECT_DIR}/install}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
ENABLE_PYTHON_BINDINGS="${ENABLE_PYTHON_BINDINGS:-ON}"
PYBIND11_CMAKE_DIR="${PYBIND11_CMAKE_DIR:-}"
NANOBIND_CMAKE_DIR="${NANOBIND_CMAKE_DIR:-}"
LLVM_INCLUDE_TESTS="${LLVM_INCLUDE_TESTS:-OFF}"
LLVM_INCLUDE_BENCHMARKS="${LLVM_INCLUDE_BENCHMARKS:-OFF}"
CLANG_INCLUDE_TESTS="${CLANG_INCLUDE_TESTS:-OFF}"
MLIR_INCLUDE_TESTS="${MLIR_INCLUDE_TESTS:-OFF}"

if [[ ! -d "${LLVM_PROJECT_DIR}/llvm" ]]; then
  echo "llvm-project source tree not found at ${LLVM_PROJECT_DIR}" >&2
  echo "Initialize submodules first: git submodule update --init --recursive 3rdparty/llvm-project" >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required" >&2
  exit 1
fi

if ! command -v ninja >/dev/null 2>&1; then
  echo "ninja is required" >&2
  exit 1
fi

if [[ "${ENABLE_PYTHON_BINDINGS}" != "OFF" && -z "${PYBIND11_CMAKE_DIR}" ]]; then
  if "${PYTHON_BIN}" -m pybind11 --cmakedir >/dev/null 2>&1; then
    PYBIND11_CMAKE_DIR="$("${PYTHON_BIN}" -m pybind11 --cmakedir)"
  fi
fi

if [[ "${ENABLE_PYTHON_BINDINGS}" != "OFF" && -z "${NANOBIND_CMAKE_DIR}" ]]; then
  if "${PYTHON_BIN}" -m nanobind --cmake_dir >/dev/null 2>&1; then
    NANOBIND_CMAKE_DIR="$("${PYTHON_BIN}" -m nanobind --cmake_dir)"
  fi
fi

if [[ "${ENABLE_PYTHON_BINDINGS}" != "OFF" && -z "${PYBIND11_CMAKE_DIR}" ]]; then
  echo "pybind11 with CMake metadata is required when MLIR Python bindings are enabled." >&2
  echo "Install it into the active environment with:" >&2
  echo "  ${PYTHON_BIN} -m pip install pybind11" >&2
  echo "or set PYBIND11_CMAKE_DIR explicitly." >&2
  exit 1
fi

if [[ "${ENABLE_PYTHON_BINDINGS}" != "OFF" && -z "${NANOBIND_CMAKE_DIR}" ]]; then
  echo "nanobind with CMake metadata is required when MLIR Python bindings are enabled." >&2
  echo "Install it into the active environment with:" >&2
  echo "  ${PYTHON_BIN} -m pip install nanobind" >&2
  echo "or set NANOBIND_CMAKE_DIR explicitly." >&2
  exit 1
fi

cmake_args=(
  -G Ninja
  -S "${LLVM_PROJECT_DIR}/llvm"
  -B "${LLVM_BUILD_DIR}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_DIR}"
  -DLLVM_ENABLE_PROJECTS="mlir;clang"
  -DLLVM_TARGETS_TO_BUILD="X86;RISCV"
  -DLLVM_ENABLE_ASSERTIONS=ON
  -DLLVM_INSTALL_UTILS=ON
  -DLLVM_INCLUDE_TOOLS=ON
  -DLLVM_INCLUDE_TESTS="${LLVM_INCLUDE_TESTS}"
  -DLLVM_INCLUDE_BENCHMARKS="${LLVM_INCLUDE_BENCHMARKS}"
  -DCLANG_INCLUDE_TESTS="${CLANG_INCLUDE_TESTS}"
  -DMLIR_INCLUDE_TESTS="${MLIR_INCLUDE_TESTS}"
  -DPython3_EXECUTABLE="${PYTHON_BIN}"
)

if [[ "${ENABLE_PYTHON_BINDINGS}" != "OFF" ]]; then
  cmake_args+=(
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}"
    -Dnanobind_DIR="${NANOBIND_CMAKE_DIR}"
  )
else
  cmake_args+=(-DMLIR_ENABLE_BINDINGS_PYTHON=OFF)
fi

cmake "${cmake_args[@]}"

cmake --build "${LLVM_BUILD_DIR}" -j"${JOBS}"
cmake --build "${LLVM_BUILD_DIR}" --target install -j"${JOBS}"

cat <<EOF
LLVM/MLIR toolchain installed to:
  ${LLVM_INSTALL_DIR}

Recommended environment variables:
  export TILELANG_RISCV_LLVM_ROOT=${LLVM_INSTALL_DIR}
  export PATH=${LLVM_INSTALL_DIR}/bin:\$PATH
EOF
