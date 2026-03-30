#include "codegen_linalg_riscv.h"

#include <tvm/ir/attrs.h>

namespace tvm {
namespace codegen {

void CodeGenTileLangLinalgRISCV::AddFunction(const GlobalVar &gvar,
                                             const tir::PrimFunc &func) {
  if (auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
    function_names_.push_back(global_symbol.value());
  } else {
    function_names_.push_back(gvar->name_hint);
  }
}

std::string CodeGenTileLangLinalgRISCV::Finish() const {
  std::ostringstream os;
  os << "module {\n";
  os << "  // Placeholder MLIR module for the linalg_riscv backend.\n";
  os << "  // Phase 1 replaces this stub with real MLIR C++ API construction.\n";
  for (const auto &name : function_names_) {
    os << "  // pending lowering for @" << name << "\n";
  }
  os << "}\n";
  return os.str();
}

} // namespace codegen
} // namespace tvm
