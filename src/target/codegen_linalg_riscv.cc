#include "codegen_linalg_riscv.h"

#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef TILELANG_ENABLE_LINALG_RISCV_MLIR
#define TILELANG_ENABLE_LINALG_RISCV_MLIR 0
#endif

#if TILELANG_ENABLE_LINALG_RISCV_MLIR
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#endif

#include <tvm/ir/attrs.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace codegen {

namespace {

using FunctionEntry = std::pair<std::string, tir::PrimFunc>;

std::string BuildPlaceholderModule(const std::vector<FunctionEntry>& functions) {
  std::ostringstream os;
  os << "module {\n";
  os << "  // Placeholder MLIR module for the linalg_riscv backend.\n";
  os << "  // Rebuild TileLang with TILELANG_RISCV_MLIR_MODE=ON after the vendored\n";
  os << "  // LLVM/MLIR toolchain is installed to enable the real C++ MLIR builder.\n";
  for (const auto& [name, func] : functions) {
    (void)func;
    os << "  // pending lowering for @" << name << "\n";
  }
  os << "}\n";
  return os.str();
}

#if TILELANG_ENABLE_LINALG_RISCV_MLIR
class TIRToMLIRLowerer final : private tir::StmtFunctor<void(const tir::Stmt&)>,
                               private tir::ExprFunctor<mlir::Value(const PrimExpr&)> {
public:
  TIRToMLIRLowerer()
      : context_(),
        builder_(&context_),
        loc_(builder_.getUnknownLoc()),
        module_(mlir::ModuleOp::create(loc_)) {
    registry_.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                     mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
    context_.appendDialectRegistry(registry_);
    context_.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                         mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
    builder_.setInsertionPointToStart(module_.getBody());
  }

  std::string Lower(const std::vector<FunctionEntry>& functions) {
    for (const auto& [name, func] : functions) {
      LowerFunction(name, func);
    }

    std::string mlir_text;
    llvm::raw_string_ostream os(mlir_text);
    module_.print(os);
    os.flush();
    return mlir_text;
  }

private:
  using tir::ExprFunctor<mlir::Value(const PrimExpr&)>::VisitExpr;
  using tir::StmtFunctor<void(const tir::Stmt&)>::VisitStmt;

  struct SavedBinding {
    bool had_value{false};
    mlir::Value value;
  };

  using ValueMap = std::unordered_map<const Object*, mlir::Value>;

  SavedBinding SaveAndSet(ValueMap& map, const Object* key, mlir::Value value) {
    SavedBinding saved;
    auto it = map.find(key);
    if (it != map.end()) {
      saved.had_value = true;
      saved.value = it->second;
    }
    map[key] = value;
    return saved;
  }

  void RestoreBinding(ValueMap& map, const Object* key, const SavedBinding& saved) {
    if (saved.had_value) {
      map[key] = saved.value;
    } else {
      map.erase(key);
    }
  }

  mlir::Type LowerScalarType(DataType dtype) {
    ICHECK_EQ(dtype.lanes(), 1) << "Vector lanes are not supported yet for linalg_riscv";
    if (dtype.is_bool()) {
      return builder_.getI1Type();
    }
    if (dtype.is_int() || dtype.is_uint()) {
      return builder_.getIntegerType(dtype.bits());
    }
    if (dtype.is_float16()) {
      return builder_.getF16Type();
    }
    if (dtype.is_bfloat16()) {
      return builder_.getBF16Type();
    }
    if (dtype.is_float() && dtype.bits() == 32) {
      return builder_.getF32Type();
    }
    if (dtype.is_float() && dtype.bits() == 64) {
      return builder_.getF64Type();
    }
    LOG(FATAL) << "Unsupported scalar dtype in linalg_riscv MLIR lowering: " << dtype;
    TVM_FFI_UNREACHABLE();
  }

  mlir::Value ConstantIntLike(int64_t value, mlir::Type type) {
    if (type.isIndex()) {
      return builder_.create<mlir::arith::ConstantIndexOp>(loc_, value);
    }
    return builder_.create<mlir::arith::ConstantIntOp>(loc_, type, value);
  }

  mlir::MemRefType LowerMemRefType(DataType element_dtype, const Array<PrimExpr>& shape_exprs) {
    llvm::SmallVector<int64_t, 4> shape;
    shape.reserve(shape_exprs.size());
    for (const PrimExpr& dim : shape_exprs) {
      if (const auto* imm = dim.as<IntImmNode>()) {
        shape.push_back(imm->value);
      } else {
        shape.push_back(mlir::ShapedType::kDynamic);
      }
    }
    return mlir::MemRefType::get(shape, LowerScalarType(element_dtype));
  }

  llvm::SmallVector<mlir::Value, 4> LowerDynamicSizes(const Array<PrimExpr>& shape_exprs) {
    llvm::SmallVector<mlir::Value, 4> dynamic_sizes;
    for (const PrimExpr& dim : shape_exprs) {
      if (!dim.as<IntImmNode>()) {
        dynamic_sizes.push_back(AsIndex(VisitExpr(dim), dim.dtype()));
      }
    }
    return dynamic_sizes;
  }

  void ValidateContiguousBuffer(const tir::Buffer& buffer) {
    ICHECK_EQ(buffer->dtype.lanes(), 1)
        << "Vector element buffers are not supported yet for linalg_riscv";
    ICHECK(buffer->strides.empty())
        << "Strided buffers are not supported yet for linalg_riscv: " << buffer->name;
    ICHECK(tir::is_zero(buffer->elem_offset))
        << "Non-zero elem_offset is not supported yet for linalg_riscv: " << buffer->name;
  }

  mlir::Value CastValueToType(mlir::Value value, mlir::Type target_type, bool source_unsigned) {
    mlir::Type source_type = value.getType();
    if (source_type == target_type) {
      return value;
    }

    if (source_type.isIndex() && target_type.isIndex()) {
      return value;
    }
    if (source_type.isIndex() && mlir::isa<mlir::IntegerType>(target_type)) {
      return builder_.create<mlir::arith::IndexCastOp>(loc_, target_type, value);
    }
    if (mlir::isa<mlir::IntegerType>(source_type) && target_type.isIndex()) {
      return builder_.create<mlir::arith::IndexCastOp>(loc_, target_type, value);
    }

    if (mlir::isa<mlir::IntegerType>(source_type) && mlir::isa<mlir::IntegerType>(target_type)) {
      unsigned source_width = mlir::cast<mlir::IntegerType>(source_type).getWidth();
      unsigned target_width = mlir::cast<mlir::IntegerType>(target_type).getWidth();
      if (source_width < target_width) {
        if (source_unsigned) {
          return builder_.create<mlir::arith::ExtUIOp>(loc_, target_type, value);
        }
        return builder_.create<mlir::arith::ExtSIOp>(loc_, target_type, value);
      }
      if (source_width > target_width) {
        return builder_.create<mlir::arith::TruncIOp>(loc_, target_type, value);
      }
      return value;
    }

    if (mlir::isa<mlir::FloatType>(source_type) && mlir::isa<mlir::FloatType>(target_type)) {
      unsigned source_width = mlir::cast<mlir::FloatType>(source_type).getWidth();
      unsigned target_width = mlir::cast<mlir::FloatType>(target_type).getWidth();
      if (source_width < target_width) {
        return builder_.create<mlir::arith::ExtFOp>(loc_, target_type, value);
      }
      if (source_width > target_width) {
        return builder_.create<mlir::arith::TruncFOp>(loc_, target_type, value);
      }
      return value;
    }

    if (mlir::isa<mlir::IntegerType>(source_type) && mlir::isa<mlir::FloatType>(target_type)) {
      if (source_unsigned) {
        return builder_.create<mlir::arith::UIToFPOp>(loc_, target_type, value);
      }
      return builder_.create<mlir::arith::SIToFPOp>(loc_, target_type, value);
    }

    if (mlir::isa<mlir::FloatType>(source_type) && mlir::isa<mlir::IntegerType>(target_type)) {
      if (source_unsigned) {
        return builder_.create<mlir::arith::FPToUIOp>(loc_, target_type, value);
      }
      return builder_.create<mlir::arith::FPToSIOp>(loc_, target_type, value);
    }

    if (source_type.isIndex() && mlir::isa<mlir::FloatType>(target_type)) {
      mlir::Type i64_type = builder_.getIntegerType(64);
      mlir::Value as_int = builder_.create<mlir::arith::IndexCastOp>(loc_, i64_type, value);
      return builder_.create<mlir::arith::SIToFPOp>(loc_, target_type, as_int);
    }

    if (mlir::isa<mlir::FloatType>(source_type) && target_type.isIndex()) {
      mlir::Type i64_type = builder_.getIntegerType(64);
      mlir::Value as_int = builder_.create<mlir::arith::FPToSIOp>(loc_, i64_type, value);
      return builder_.create<mlir::arith::IndexCastOp>(loc_, target_type, as_int);
    }

    LOG(FATAL) << "Unsupported MLIR cast encountered in linalg_riscv lowering";
    TVM_FFI_UNREACHABLE();
  }

  mlir::Value CastValue(mlir::Value value, DataType source_dtype, DataType target_dtype) {
    if (target_dtype.is_bool()) {
      return LowerConditionValue(value, source_dtype);
    }
    mlir::Type target_type = LowerScalarType(target_dtype);
    if (source_dtype == target_dtype && value.getType() == target_type) {
      return value;
    }
    bool source_unsigned = source_dtype.is_uint() || source_dtype.is_bool();
    return CastValueToType(value, target_type, source_unsigned);
  }

  mlir::Value LowerConditionValue(mlir::Value value, DataType source_dtype) {
    mlir::Type value_type = value.getType();
    if (value_type.isInteger(1)) {
      return value;
    }
    if (value_type.isIndex()) {
      return builder_.create<mlir::arith::CmpIOp>(
          loc_, mlir::arith::CmpIPredicate::ne, value, ConstantIntLike(0, builder_.getIndexType()));
    }
    if (mlir::isa<mlir::IntegerType>(value_type)) {
      mlir::Type zero_type = value_type;
      return builder_.create<mlir::arith::CmpIOp>(
          loc_, mlir::arith::CmpIPredicate::ne, value, ConstantIntLike(0, zero_type));
    }
    if (mlir::isa<mlir::FloatType>(value_type)) {
      mlir::Type zero_type = value_type;
      mlir::Value zero = builder_.create<mlir::arith::ConstantFloatOp>(
          loc_, mlir::cast<mlir::FloatType>(zero_type), llvm::APFloat(0.0));
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::UNE, value,
                                                  zero);
    }
    LOG(FATAL) << "Unsupported condition value type in linalg_riscv lowering for TIR dtype "
               << source_dtype;
    TVM_FFI_UNREACHABLE();
  }

  mlir::Value LowerCondition(const PrimExpr& expr) {
    return LowerConditionValue(VisitExpr(expr), expr.dtype());
  }

  mlir::Value AsIndex(mlir::Value value, DataType source_dtype) {
    if (value.getType().isIndex()) {
      return value;
    }
    return CastValueToType(value, builder_.getIndexType(), source_dtype.is_uint() || source_dtype.is_bool());
  }

  mlir::Value LookupVarValue(const tir::Var& var) const {
    auto scalar_it = scalar_values_.find(var.get());
    if (scalar_it != scalar_values_.end()) {
      return scalar_it->second;
    }
    auto buffer_it = buffer_values_.find(var.get());
    if (buffer_it != buffer_values_.end()) {
      return buffer_it->second;
    }
    LOG(FATAL) << "Unbound TIR var during linalg_riscv lowering: " << var->name_hint;
    TVM_FFI_UNREACHABLE();
  }

  mlir::Value LookupBufferValue(const tir::Buffer& buffer) const {
    auto it = buffer_values_.find(buffer.get());
    if (it != buffer_values_.end()) {
      return it->second;
    }
    it = buffer_values_.find(buffer->data.get());
    if (it != buffer_values_.end()) {
      return it->second;
    }
    LOG(FATAL) << "Unbound TIR buffer during linalg_riscv lowering: " << buffer->name;
    TVM_FFI_UNREACHABLE();
  }

  void RestoreBindings(ValueMap& map,
                       const std::vector<std::pair<const Object*, SavedBinding>>& saved_bindings) {
    for (auto it = saved_bindings.rbegin(); it != saved_bindings.rend(); ++it) {
      RestoreBinding(map, it->first, it->second);
    }
  }

  void BindBufferAliases(const tir::Buffer& buffer, mlir::Value value,
                         std::vector<std::pair<const Object*, SavedBinding>>* saved_bindings) {
    saved_bindings->emplace_back(buffer.get(), SaveAndSet(buffer_values_, buffer.get(), value));
    saved_bindings->emplace_back(buffer->data.get(),
                                 SaveAndSet(buffer_values_, buffer->data.get(), value));
  }

  mlir::Value CreateAlloca(const Array<PrimExpr>& shape_exprs, DataType element_dtype) {
    mlir::MemRefType memref_type = LowerMemRefType(element_dtype, shape_exprs);
    llvm::SmallVector<mlir::Value, 4> dynamic_sizes = LowerDynamicSizes(shape_exprs);
    return builder_.create<mlir::memref::AllocaOp>(loc_, memref_type, dynamic_sizes);
  }

  template <typename F>
  void EmitConditionalRegion(const PrimExpr& predicate, F&& body_builder) {
    if (tir::is_one(predicate)) {
      body_builder();
      return;
    }
    mlir::Value cond = LowerCondition(predicate);
    mlir::scf::IfOp if_op = builder_.create<mlir::scf::IfOp>(loc_, cond, false);
    mlir::OpBuilder::InsertionGuard guard(builder_);
    builder_.setInsertionPoint(if_op.thenYield());
    body_builder();
  }

  void LowerFunction(const std::string& name, const tir::PrimFunc& func) {
    scalar_values_.clear();
    buffer_values_.clear();

    llvm::SmallVector<mlir::Type, 8> input_types;
    input_types.reserve(func->params.size());

    for (const tir::Var& param : func->params) {
      if (func->buffer_map.count(param)) {
        const tir::Buffer& buffer = func->buffer_map[param];
        ValidateContiguousBuffer(buffer);
        input_types.push_back(LowerMemRefType(buffer->dtype, buffer->shape));
      } else {
        ICHECK(!param.dtype().is_handle())
            << "Handle scalar params without buffer_map are not supported yet: " << param;
        input_types.push_back(LowerScalarType(param.dtype()));
      }
    }

    mlir::func::FuncOp func_op = builder_.create<mlir::func::FuncOp>(
        loc_, name, builder_.getFunctionType(input_types, llvm::ArrayRef<mlir::Type>{}));
    mlir::Block* entry_block = func_op.addEntryBlock();

    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPointToStart(entry_block);

      for (size_t i = 0; i < func->params.size(); ++i) {
        const tir::Var& param = func->params[i];
        mlir::Value arg = entry_block->getArgument(static_cast<unsigned>(i));
        if (func->buffer_map.count(param)) {
          const tir::Buffer& buffer = func->buffer_map[param];
          buffer_values_[param.get()] = arg;
          buffer_values_[buffer.get()] = arg;
          buffer_values_[buffer->data.get()] = arg;
        } else {
          scalar_values_[param.get()] = arg;
        }
      }

      VisitStmt(func->body);
      builder_.create<mlir::func::ReturnOp>(loc_);
    }
  }

  void VisitStmt_(const tir::SeqStmtNode* op) final {
    for (const tir::Stmt& stmt : op->seq) {
      VisitStmt(stmt);
    }
  }

  void VisitStmt_(const tir::EvaluateNode* op) final {
    if (!op->value.as<IntImmNode>() || !tir::is_zero(op->value)) {
      (void)VisitExpr(op->value);
    }
  }

  void VisitStmt_(const tir::ForNode* op) final {
    ICHECK(op->kind == tir::ForKind::kSerial || op->kind == tir::ForKind::kUnrolled)
        << "Only serial/unrolled loops are supported in the current linalg_riscv lowering";
    ICHECK(!op->thread_binding.defined())
        << "Thread-bound loops are not supported in linalg_riscv lowering";

    mlir::Value lower = AsIndex(VisitExpr(op->min), op->min.dtype());
    mlir::Value extent = AsIndex(VisitExpr(op->extent), op->extent.dtype());
    mlir::Value step =
        op->step.defined() ? AsIndex(VisitExpr(op->step.value()), op->step.value().dtype())
                           : ConstantIntLike(1, builder_.getIndexType());
    mlir::Value upper = builder_.create<mlir::arith::AddIOp>(loc_, lower, extent);
    mlir::scf::ForOp for_op = builder_.create<mlir::scf::ForOp>(loc_, lower, upper, step);

    SavedBinding saved_loop_var = SaveAndSet(scalar_values_, op->loop_var.get(), for_op.getInductionVar());
    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPoint(for_op.getBody()->getTerminator());
      VisitStmt(op->body);
    }
    RestoreBinding(scalar_values_, op->loop_var.get(), saved_loop_var);
  }

  void VisitStmt_(const tir::IfThenElseNode* op) final {
    mlir::Value cond = LowerCondition(op->condition);
    bool has_else = op->else_case.defined();
    mlir::scf::IfOp if_op = builder_.create<mlir::scf::IfOp>(loc_, cond, has_else);

    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPoint(if_op.thenYield());
      VisitStmt(op->then_case);
    }

    if (has_else) {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPoint(if_op.elseYield());
      VisitStmt(op->else_case.value());
    }
  }

  void VisitStmt_(const tir::BufferStoreNode* op) final {
    mlir::Value memref = LookupBufferValue(op->buffer);
    llvm::SmallVector<mlir::Value, 4> indices;
    indices.reserve(op->indices.size());
    for (const PrimExpr& index : op->indices) {
      indices.push_back(AsIndex(VisitExpr(index), index.dtype()));
    }

    auto emit_store = [&]() {
      mlir::Value value = VisitExpr(op->value);
      value = CastValue(value, op->value.dtype(), op->buffer->dtype);
      builder_.create<mlir::memref::StoreOp>(loc_, value, memref, indices);
    };

    if (op->predicate.defined()) {
      EmitConditionalRegion(op->predicate.value(), emit_store);
    } else {
      emit_store();
    }
  }

  void VisitStmt_(const tir::DeclBufferNode* op) final {
    std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
    auto it = buffer_values_.find(op->buffer->data.get());
    ICHECK(it != buffer_values_.end())
        << "DeclBuffer lowered before its data binding was materialized: " << op->buffer->name;
    BindBufferAliases(op->buffer, it->second, &saved_bindings);
    VisitStmt(op->body);
    RestoreBindings(buffer_values_, saved_bindings);
  }

  void VisitStmt_(const tir::AllocateNode* op) final {
    auto emit_body = [&]() {
      std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
      mlir::Value alloc = CreateAlloca(op->extents, op->dtype);
      saved_bindings.emplace_back(op->buffer_var.get(),
                                  SaveAndSet(buffer_values_, op->buffer_var.get(), alloc));
      VisitStmt(op->body);
      RestoreBindings(buffer_values_, saved_bindings);
    };
    EmitConditionalRegion(op->condition, emit_body);
  }

  void VisitStmt_(const tir::BufferRealizeNode* op) final {
    ValidateContiguousBuffer(op->buffer);
    Array<PrimExpr> extents;
    for (const Range& range : op->bounds) {
      extents.push_back(range->extent);
    }

    std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
    mlir::Value alloc = CreateAlloca(extents, op->buffer->dtype);
    BindBufferAliases(op->buffer, alloc, &saved_bindings);
    EmitConditionalRegion(op->condition, [&]() { VisitStmt(op->body); });
    RestoreBindings(buffer_values_, saved_bindings);
  }

  void VisitStmt_(const tir::AttrStmtNode* op) final { VisitStmt(op->body); }

  void VisitStmt_(const tir::BlockNode* op) final {
    ICHECK(!op->init.defined()) << "Reduction blocks are not supported yet in linalg_riscv lowering";
    ICHECK(op->match_buffers.empty())
        << "match_buffer regions are not supported yet in linalg_riscv lowering";

    std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
    saved_bindings.reserve(op->alloc_buffers.size() * 2);
    for (const tir::Buffer& buffer : op->alloc_buffers) {
      ValidateContiguousBuffer(buffer);
      mlir::Value alloc = CreateAlloca(buffer->shape, buffer->dtype);
      BindBufferAliases(buffer, alloc, &saved_bindings);
    }

    VisitStmt(op->body);
    RestoreBindings(buffer_values_, saved_bindings);
  }

  void VisitStmt_(const tir::BlockRealizeNode* op) final {
    ICHECK_EQ(op->iter_values.size(), op->block->iter_vars.size())
        << "BlockRealize iter_values must match block iter_vars";

    std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
    saved_bindings.reserve(op->iter_values.size());
    for (size_t i = 0; i < op->iter_values.size(); ++i) {
      const tir::IterVar& iter_var = op->block->iter_vars[i];
      mlir::Value iter_value = VisitExpr(op->iter_values[i]);
      saved_bindings.emplace_back(iter_var->var.get(),
                                  SaveAndSet(scalar_values_, iter_var->var.get(), iter_value));
    }

    EmitConditionalRegion(op->predicate, [&]() { VisitStmt(op->block); });
    RestoreBindings(scalar_values_, saved_bindings);
  }

  void VisitStmtDefault_(const Object* op) final {
    LOG(FATAL) << "Unsupported TIR stmt for linalg_riscv MLIR lowering: " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  mlir::Value VisitExpr_(const tir::VarNode* op) final {
    return LookupVarValue(tvm::ffi::GetRef<tir::Var>(op));
  }

  mlir::Value VisitExpr_(const tir::BufferLoadNode* op) final {
    ICHECK(!op->predicate.defined() || tir::is_one(op->predicate.value()))
        << "Predicated buffer loads are not supported yet in linalg_riscv lowering";

    mlir::Value memref = LookupBufferValue(op->buffer);
    llvm::SmallVector<mlir::Value, 4> indices;
    indices.reserve(op->indices.size());
    for (const PrimExpr& index : op->indices) {
      indices.push_back(AsIndex(VisitExpr(index), index.dtype()));
    }
    mlir::Value load = builder_.create<mlir::memref::LoadOp>(loc_, memref, indices);
    return CastValue(load, op->buffer->dtype, op->dtype);
  }

  mlir::Value VisitExpr_(const tir::AddNode* op) final {
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
    if (op->dtype.is_float()) {
      return builder_.create<mlir::arith::AddFOp>(loc_, lhs, rhs);
    }
    return builder_.create<mlir::arith::AddIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::SubNode* op) final {
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
    if (op->dtype.is_float()) {
      return builder_.create<mlir::arith::SubFOp>(loc_, lhs, rhs);
    }
    return builder_.create<mlir::arith::SubIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::MulNode* op) final {
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
    if (op->dtype.is_float()) {
      return builder_.create<mlir::arith::MulFOp>(loc_, lhs, rhs);
    }
    return builder_.create<mlir::arith::MulIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::DivNode* op) final {
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
    if (op->dtype.is_float()) {
      return builder_.create<mlir::arith::DivFOp>(loc_, lhs, rhs);
    }
    if (op->dtype.is_uint() || op->dtype.is_bool()) {
      return builder_.create<mlir::arith::DivUIOp>(loc_, lhs, rhs);
    }
    return builder_.create<mlir::arith::DivSIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::CastNode* op) final {
    mlir::Value value = VisitExpr(op->value);
    return CastValue(value, op->value.dtype(), op->dtype);
  }

  mlir::Value VisitExpr_(const tir::EQNode* op) final {
    DataType compare_dtype = op->a.dtype();
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
    if (compare_dtype.is_float()) {
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OEQ, lhs,
                                                  rhs);
    }
    return builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::eq, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::NENode* op) final {
    DataType compare_dtype = op->a.dtype();
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
    if (compare_dtype.is_float()) {
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::UNE, lhs,
                                                  rhs);
    }
    return builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::ne, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::LTNode* op) final {
    DataType compare_dtype = op->a.dtype();
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
    if (compare_dtype.is_float()) {
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OLT, lhs,
                                                  rhs);
    }
    mlir::arith::CmpIPredicate predicate =
        compare_dtype.is_uint() || compare_dtype.is_bool() ? mlir::arith::CmpIPredicate::ult
                                                           : mlir::arith::CmpIPredicate::slt;
    return builder_.create<mlir::arith::CmpIOp>(loc_, predicate, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::LENode* op) final {
    DataType compare_dtype = op->a.dtype();
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
    if (compare_dtype.is_float()) {
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OLE, lhs,
                                                  rhs);
    }
    mlir::arith::CmpIPredicate predicate =
        compare_dtype.is_uint() || compare_dtype.is_bool() ? mlir::arith::CmpIPredicate::ule
                                                           : mlir::arith::CmpIPredicate::sle;
    return builder_.create<mlir::arith::CmpIOp>(loc_, predicate, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::GTNode* op) final {
    DataType compare_dtype = op->a.dtype();
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
    if (compare_dtype.is_float()) {
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OGT, lhs,
                                                  rhs);
    }
    mlir::arith::CmpIPredicate predicate =
        compare_dtype.is_uint() || compare_dtype.is_bool() ? mlir::arith::CmpIPredicate::ugt
                                                           : mlir::arith::CmpIPredicate::sgt;
    return builder_.create<mlir::arith::CmpIOp>(loc_, predicate, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::GENode* op) final {
    DataType compare_dtype = op->a.dtype();
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
    if (compare_dtype.is_float()) {
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OGE, lhs,
                                                  rhs);
    }
    mlir::arith::CmpIPredicate predicate =
        compare_dtype.is_uint() || compare_dtype.is_bool() ? mlir::arith::CmpIPredicate::uge
                                                           : mlir::arith::CmpIPredicate::sge;
    return builder_.create<mlir::arith::CmpIOp>(loc_, predicate, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::AndNode* op) final {
    mlir::Value lhs = LowerCondition(op->a);
    mlir::Value rhs = LowerCondition(op->b);
    return builder_.create<mlir::arith::AndIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::OrNode* op) final {
    mlir::Value lhs = LowerCondition(op->a);
    mlir::Value rhs = LowerCondition(op->b);
    return builder_.create<mlir::arith::OrIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::NotNode* op) final {
    mlir::Value value = LowerCondition(op->a);
    mlir::Value one = ConstantIntLike(1, builder_.getI1Type());
    return builder_.create<mlir::arith::XOrIOp>(loc_, value, one);
  }

  mlir::Value VisitExpr_(const tir::SelectNode* op) final {
    mlir::Value cond = LowerCondition(op->condition);
    mlir::Value true_value = CastValue(VisitExpr(op->true_value), op->true_value.dtype(), op->dtype);
    mlir::Value false_value =
        CastValue(VisitExpr(op->false_value), op->false_value.dtype(), op->dtype);
    return builder_.create<mlir::arith::SelectOp>(loc_, cond, true_value, false_value);
  }

  mlir::Value VisitExpr_(const IntImmNode* op) final {
    mlir::Type type = LowerScalarType(op->dtype);
    return ConstantIntLike(op->value, type);
  }

  mlir::Value VisitExpr_(const FloatImmNode* op) final {
    mlir::FloatType type = mlir::cast<mlir::FloatType>(LowerScalarType(op->dtype));
    return builder_.create<mlir::arith::ConstantFloatOp>(loc_, type, llvm::APFloat(op->value));
  }

  mlir::Value VisitExprDefault_(const Object* op) final {
    LOG(FATAL) << "Unsupported TIR expr for linalg_riscv MLIR lowering: " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  mlir::DialectRegistry registry_;
  mlir::MLIRContext context_;
  mlir::OpBuilder builder_;
  mlir::Location loc_;
  mlir::ModuleOp module_;
  ValueMap scalar_values_;
  ValueMap buffer_values_;
};

std::string BuildStructuredMLIRModule(const std::vector<FunctionEntry>& functions) {
  TIRToMLIRLowerer lowerer;
  return lowerer.Lower(functions);
}
#endif

}  // namespace

void CodeGenTileLangLinalgRISCV::AddFunction(const GlobalVar& gvar, const tir::PrimFunc& func) {
  std::string name;
  if (auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
    name = global_symbol.value();
  } else {
    name = gvar->name_hint;
  }
  function_names_.push_back(name);
  functions_.emplace_back(name, func);
}

std::string CodeGenTileLangLinalgRISCV::Finish() const {
#if TILELANG_ENABLE_LINALG_RISCV_MLIR
  return BuildStructuredMLIRModule(functions_);
#else
  return BuildPlaceholderModule(functions_);
#endif
}

}  // namespace codegen
}  // namespace tvm
