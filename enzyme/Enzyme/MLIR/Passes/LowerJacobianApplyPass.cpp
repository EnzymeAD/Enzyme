//===- LowerJacobianApplyPass.cpp - Lower Jacobian apply ops --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers enzyme.jvp_apply/enzyme.vjp_apply to
// enzyme.fwddiff/enzyme.autodiff when the Jacobian originates from a
// single-input single-result enzyme.jacobian.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Ops.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERJACOBIANAPPLYPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

struct JacobianInfo {
  JacobianOp jacobian;
};

static FailureOr<JacobianInfo>
getJacobianInfo(Operation *applyOp, Value jacobianValue,
                SymbolTableCollection &symbolTable) {
  auto jacobianOp = jacobianValue.getDefiningOp<JacobianOp>();
  if (!jacobianOp) {
    applyOp->emitOpError(
        "expects first operand to come from enzyme.jacobian");
    return failure();
  }

  if (jacobianOp.getInputs().size() != 1 || jacobianOp.getOutputs().size() != 1) {
    applyOp->emitOpError(
        "requires enzyme.jacobian to have exactly one primal input and one "
        "result");
    return failure();
  }

  auto primalFunction =
      symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
          jacobianOp, jacobianOp.getFnAttr());
  if (!primalFunction) {
    applyOp->emitOpError(
        "requires enzyme.jacobian to reference a valid function symbol");
    return failure();
  }

  if (primalFunction.getNumArguments() != 1 || primalFunction.getNumResults() != 1) {
    applyOp->emitOpError(
        "requires referenced function to have exactly one argument and one "
        "result");
    return failure();
  }

  return JacobianInfo{jacobianOp};
}

struct LowerJacobianApplyPass
    : public enzyme::impl::LowerJacobianApplyPassBase<LowerJacobianApplyPass> {
  void runOnOperation() override {
    SymbolTableCollection symbolTable;
    SmallVector<JVPApplyOp> jvpApplyOps;
    SmallVector<VJPApplyOp> vjpApplyOps;

    getOperation()->walk([&](JVPApplyOp op) { jvpApplyOps.push_back(op); });
    getOperation()->walk([&](VJPApplyOp op) { vjpApplyOps.push_back(op); });

    bool hadFailure = false;

    for (JVPApplyOp applyOp : jvpApplyOps) {
      auto info =
          getJacobianInfo(applyOp, applyOp.getJacobian(), symbolTable);
      if (mlir::failed(info)) {
        hadFailure = true;
        break;
      }

      OpBuilder builder(applyOp);
      auto *ctx = builder.getContext();

      SmallVector<Attribute> inActivityAttrs = {
          ActivityAttr::get(ctx, Activity::enzyme_dup)};
      SmallVector<Attribute> retActivityAttrs = {
          ActivityAttr::get(ctx, Activity::enzyme_dupnoneed)};
      ArrayAttr inActivity = ArrayAttr::get(ctx, inActivityAttrs);
      ArrayAttr retActivity = ArrayAttr::get(ctx, retActivityAttrs);

      auto fwdDiff = ForwardDiffOp::create(
          builder, applyOp.getLoc(),
          TypeRange{applyOp.getOutput().getType()},
          info->jacobian.getFnAttr(),
          ValueRange{info->jacobian.getInputs().front(), applyOp.getVector()},
          inActivity, retActivity, info->jacobian.getWidthAttr(),
          info->jacobian.getStrongZeroAttr());

      applyOp.replaceAllUsesWith(fwdDiff.getOutputs());
      applyOp.erase();
      if (info->jacobian->use_empty())
        info->jacobian.erase();
    }

    for (VJPApplyOp applyOp : vjpApplyOps) {
      if (hadFailure)
        break;

      auto info =
          getJacobianInfo(applyOp, applyOp.getJacobian(), symbolTable);
      if (mlir::failed(info)) {
        hadFailure = true;
        break;
      }

      OpBuilder builder(applyOp);
      auto *ctx = builder.getContext();

      SmallVector<Attribute> inActivityAttrs = {
          ActivityAttr::get(ctx, Activity::enzyme_active)};
      SmallVector<Attribute> retActivityAttrs = {
          ActivityAttr::get(ctx, Activity::enzyme_activenoneed)};
      ArrayAttr inActivity = ArrayAttr::get(ctx, inActivityAttrs);
      ArrayAttr retActivity = ArrayAttr::get(ctx, retActivityAttrs);

      auto revDiff = AutoDiffOp::create(
          builder, applyOp.getLoc(),
          TypeRange{applyOp.getOutput().getType()},
          info->jacobian.getFnAttr(),
          ValueRange{info->jacobian.getInputs().front(), applyOp.getVector()},
          inActivity, retActivity, info->jacobian.getWidthAttr(),
          info->jacobian.getStrongZeroAttr());

      applyOp.replaceAllUsesWith(revDiff.getOutputs());
      applyOp.erase();
      if (info->jacobian->use_empty())
        info->jacobian.erase();
    }

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
