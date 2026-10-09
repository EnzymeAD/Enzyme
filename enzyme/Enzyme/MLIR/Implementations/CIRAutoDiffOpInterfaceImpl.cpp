//===- CIRAutoDiffOpInterfaceImpl.cpp - Interface external model ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR SCF dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "Dialect/Ops.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/CIRDerivatives.inc"
} // namespace

namespace {
struct SwitchFlatBranchOpInterface
    : public BranchOpInterface::ExternalModel<SwitchFlatBranchOpInterface,
                                              cir::SwitchFlatOp> {

  SuccessorOperands getSuccessorOperands(Operation *op, unsigned index) const {
    auto sw = cast<cir::SwitchFlatOp>(op);
    assert(index < sw->getNumSuccessors() && "invalid successor index");
    if (index == 0)
      return SuccessorOperands(sw.getDefaultOperandsMutable());
    return SuccessorOperands(sw.getCaseOperandsMutable()[index - 1]);
  }

  std::optional<BlockArgument>
  getSuccessorBlockArgument(Operation *op, unsigned operandIndex) const {
    for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i) {
      if (auto arg = mlir::detail::getBranchSuccessorArgument(
              getSuccessorOperands(op, i), operandIndex, op->getSuccessor(i)))
        return arg;
    }
    return std::nullopt;
  }
};
} // namespace

class AutoDiffCIRFuncOpFunctionInterface
    : public AutoDiffFunctionInterface::ExternalModel<
          AutoDiffCIRFuncOpFunctionInterface, cir::FuncOp> {
public:
  static Type packedType(MLIRContext *ctx, TypeRange types) {
    SmallVector<Type> members(types.begin(), types.end());
    SmallVector<cir::RecordMemberKind> kinds(members.size(),
                                             cir::RecordMemberKind::Data);
    return cir::StructType::get(ctx, members, /*packed=*/false,
                                /*is_class=*/false, kinds);
  }

  void transformResultTypes(Operation *self,
                            SmallVectorImpl<Type> &resultTypes) const {
    if (resultTypes.size() <= 1)
      return;
    Type packed = packedType(self->getContext(), resultTypes);
    resultTypes.clear();
    resultTypes.push_back(packed);
  }
  void detachFromPrimalDefinition(Operation *self) const {
    // cir.func has comdat as a unit attr, not a symbol ref: nothing to
    // retarget.
  }
  Operation *createCall(Operation *self, OpBuilder &b, Location loc,
                        ValueRange args) const {
    auto fn = cast<cir::FuncOp>(self);
    Type res = fn.getFunctionType().getReturnTypes().empty()
                   ? Type()
                   : fn.getFunctionType().getReturnType();
    return cir::CallOp::create(b, loc, SymbolRefAttr::get(fn), res, args);
  }
  Operation *createReturn(Operation *, OpBuilder &b, Location loc,
                          ValueRange args) const {
    if (args.size() <= 1)
      return cir::ReturnOp::create(b, loc, args);
    Type packed = packedType(b.getContext(), args.getTypes());
    Value result =
        cir::ConstantOp::create(b, loc, packed, cir::ZeroAttr::get(packed));
    for (auto &&[i, v] : llvm::enumerate(args))
      result = cir::InsertMemberOp::create(b, loc, result, i, v);
    return cir::ReturnOp::create(b, loc, ValueRange{result});
  }
};

struct CIRReturnOpFunctionReturnInterface
    : public FunctionReturnOpInterface::ExternalModel<
          CIRReturnOpFunctionReturnInterface, cir::ReturnOp> {};

void mlir::enzyme::registerCIRDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, cir::CIRDialect *) {
    cir::SwitchFlatOp::attachInterface<SwitchFlatBranchOpInterface>(*context);
    registerInterfaces(context);
    registerCIRAutoDiffTypeInterfaces(context);
    cir::FuncOp::attachInterface<AutoDiffCIRFuncOpFunctionInterface>(*context);
    cir::ReturnOp::attachInterface<CIRReturnOpFunctionReturnInterface>(
        *context);
  });
}
