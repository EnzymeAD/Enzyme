//===- PrintActivityAnalysis.cpp - Pass to print activity analysis --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to print the results of running activity
// analysis.
//===----------------------------------------------------------------------===//
#include "Analysis/ActivityAnalysis.h"
#include "Dialect/Ops.h"
#include "Passes/PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "llvm/Demangle/Demangle.h"

using namespace mlir;

namespace {
using llvm::errs;

/// Parse calls to __enzyme_autodiff to enzyme.autodiff ops. Specific to
/// importing LLVM IR tests to MLIR. This will hopefully be replaced with a more
/// robust mechanism.
LogicalResult inferEnzymeAutodiffOps(ModuleOp moduleOp) {
  MLIRContext *ctx = moduleOp.getContext();
  SymbolTableCollection symbolTable;
  FunctionOpInterface enzymeOp = nullptr;
  for (Operation &op : moduleOp.getBody()->getOperations()) {
    if (auto funcOp = dyn_cast<FunctionOpInterface>(&op))
      if (StringRef(llvm::demangle(funcOp.getName().str()))
              .starts_with("__enzyme_autodiff")) {
        enzymeOp = funcOp;
        break;
      }
  }
  if (!enzymeOp) {
    moduleOp.emitError("Failed to find __enzyme_autodiff op");
    return failure();
  }

  auto uses = enzymeOp.getSymbolUses(moduleOp);
  for (SymbolTable::SymbolUse use : *uses) {
    if (!isa<CallOpInterface>(use.getUser()))
      continue;
    auto callOp = cast<CallOpInterface>(use.getUser());
    auto operands = callOp.getArgOperands();

    // Get the callee
    auto addressOf = cast<LLVM::AddressOfOp>(operands[0].getDefiningOp());
    auto callee = addressOf.getFunction(symbolTable);
    SmallVector<Attribute> argActivities;
    argActivities.reserve(callee.getNumArguments());
    for (Value operand : operands.drop_front(1)) {
      if (isa<FloatType>(operand.getType())) {
        argActivities.push_back(
            enzyme::ActivityAttr::get(ctx, enzyme::Activity::enzyme_out));
      } else if (operand.getType().isIntOrIndex()) {
        argActivities.push_back(
            enzyme::ActivityAttr::get(ctx, enzyme::Activity::enzyme_const));
      }
    }

    // TODO: this is not aware of activity annotations
    auto argOperands = operands.drop_front(1);

    // Replace the call with the op
    OpBuilder builder{ctx};
    builder.setInsertionPoint(callOp);
    auto autodiffOp = builder.create<enzyme::AutoDiffOp>(
        callOp.getLoc(), callOp->getResultTypes(), callee.getName(),
        argOperands, builder.getArrayAttr(argActivities));
    enzyme::runDataFlowActivityAnalysis(autodiffOp, callee);
    return success();
  }
}

struct PrintActivityAnalysisPass
    : public enzyme::PrintActivityAnalysisPassBase<PrintActivityAnalysisPass> {
  void runOnOperation() override {
    auto moduleOp = cast<ModuleOp>(getOperation());

    if (failed(inferEnzymeAutodiffOps(moduleOp))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createPrintActivityAnalysisPass() {
  return std::make_unique<PrintActivityAnalysisPass>();
}
} // namespace enzyme
} // namespace mlir
