#include "codegen_linalg_riscv.h"

#include <sstream>

#ifndef TILELANG_ENABLE_LINALG_RISCV_MLIR
#define TILELANG_ENABLE_LINALG_RISCV_MLIR 0
#endif

#if TILELANG_ENABLE_LINALG_RISCV_MLIR
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#endif

#include <tvm/ir/attrs.h>

namespace tvm {
namespace codegen {

namespace {

std::string BuildPlaceholderModule(const Array<String> &function_names) {
  std::ostringstream os;
  os << "module {\n";
  os << "  // Placeholder MLIR module for the linalg_riscv backend.\n";
  os << "  // Rebuild TileLang with TILELANG_RISCV_MLIR_MODE=ON after the vendored\n";
  os << "  // LLVM/MLIR toolchain is installed to enable the real C++ MLIR builder.\n";
  for (const auto &name : function_names) {
    os << "  // pending lowering for @" << name << "\n";
  }
  os << "}\n";
  return os.str();
}

#if TILELANG_ENABLE_LINALG_RISCV_MLIR
std::string BuildMinimalMLIRModule(const Array<String> &function_names) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();

  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::func::FuncDialect>();

  mlir::OpBuilder builder(&context);
  mlir::Location loc = builder.getUnknownLoc();
  mlir::ModuleOp module = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

  for (const auto &name : function_names) {
    auto func = builder.create<mlir::func::FuncOp>(
        loc, name.operator std::string(), builder.getFunctionType({}, {}));
    mlir::Block *entry_block = func.addEntryBlock();
    builder.setInsertionPointToEnd(entry_block);
    builder.create<mlir::func::ReturnOp>(loc);
    builder.setInsertionPointAfter(func);
  }

  std::string mlir_text;
  llvm::raw_string_ostream os(mlir_text);
  module.print(os);
  os.flush();
  return mlir_text;
}
#endif

} // namespace

void CodeGenTileLangLinalgRISCV::AddFunction(const GlobalVar &gvar,
                                             const tir::PrimFunc &func) {
  if (auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
    function_names_.push_back(global_symbol.value());
  } else {
    function_names_.push_back(gvar->name_hint);
  }
}

std::string CodeGenTileLangLinalgRISCV::Finish() const {
#if TILELANG_ENABLE_LINALG_RISCV_MLIR
  return BuildMinimalMLIRModule(function_names_);
#else
  return BuildPlaceholderModule(function_names_);
#endif
}

} // namespace codegen
} // namespace tvm
