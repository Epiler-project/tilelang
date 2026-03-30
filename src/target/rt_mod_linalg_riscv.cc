#include "codegen_linalg_riscv.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target_kind.h>

#include "target/source/codegen_source_base.h"

namespace tvm {
namespace codegen {

ffi::Module BuildTileLangLinalgRISCV(IRModule mod, Target target) {
  (void)target;
  CodeGenTileLangLinalgRISCV cg;
  for (const auto &kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "CodeGenTileLangLinalgRISCV: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto func = Downcast<tir::PrimFunc>(kv.second);
    cg.AddFunction(gvar, func);
  }

  std::string code = cg.Finish();
  return CSourceModuleCreate(code, "mlir", cg.GetFunctionNames());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.tilelang_linalg_riscv",
                        BuildTileLangLinalgRISCV);
}

TVM_REGISTER_TARGET_KIND("linalg_riscv", kDLCPU);

} // namespace codegen
} // namespace tvm
