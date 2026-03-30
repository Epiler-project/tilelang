#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLVM_PROJECT_DIR="${TILELANG_LLVM_PROJECT_DIR:-${ROOT_DIR}/3rdparty/llvm-project}"
LLVM_BUILD_DIR="${TILELANG_LLVM_BUILD_DIR:-${LLVM_PROJECT_DIR}/build-host}"
LLVM_INSTALL_DIR="${TILELANG_LLVM_INSTALL_DIR:-${LLVM_PROJECT_DIR}/install}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

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

cmake -G Ninja \
  -S "${LLVM_PROJECT_DIR}/llvm" \
  -B "${LLVM_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_DIR}" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="X86;RISCV" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="${PYTHON_BIN}"

cmake --build "${LLVM_BUILD_DIR}" -j"${JOBS}"
cmake --build "${LLVM_BUILD_DIR}" --target install -j"${JOBS}"

cat <<EOF
LLVM/MLIR toolchain installed to:
  ${LLVM_INSTALL_DIR}

Recommended environment variables:
  export TILELANG_RISCV_LLVM_ROOT=${LLVM_INSTALL_DIR}
  export PATH=${LLVM_INSTALL_DIR}/bin:\$PATH
EOF
