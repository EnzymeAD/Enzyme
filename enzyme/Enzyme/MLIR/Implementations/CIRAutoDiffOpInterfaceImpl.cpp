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
  void transformResultTypes(Operation *, SmallVectorImpl<Type> &types) const {
    assert(types.size() <= 1 && "TODO: pack multiple results into cir.record");
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
    return cir::ReturnOp::create(b, loc, args);
  }
};

void mlir::enzyme::registerCIRDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, cir::CIRDialect *) {
    cir::SwitchFlatOp::attachInterface<SwitchFlatBranchOpInterface>(*context);
    registerInterfaces(context);
    registerCIRAutoDiffTypeInterfaces(context);
    cir::FuncOp::attachInterface<AutoDiffCIRFuncOpFunctionInterface>(*context);
  });
}
