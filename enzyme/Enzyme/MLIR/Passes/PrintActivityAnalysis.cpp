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
#include "Analysis/DataFlowActivityAnalysis.h"
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
/// robust mechanism (e.g. Polygeist).
LogicalResult inferEnzymeAutodiffOps(ModuleOp moduleOp) {
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
    SmallVector<enzyme::Activity> argActivities;
    argActivities.reserve(callee.getNumArguments());
    for (Value operand : operands.drop_front(1)) {
      if (isa<FloatType>(operand.getType())) {
        argActivities.push_back(enzyme::Activity::enzyme_out);
      } else if (operand.getType().isIntOrIndex()) {
        argActivities.push_back(enzyme::Activity::enzyme_const);
      }
    }
    enzyme::runDataFlowActivityAnalysis(callee, argActivities);
  }
  return success();
}

struct PrintActivityAnalysisPass
    : public enzyme::PrintActivityAnalysisPassBase<PrintActivityAnalysisPass> {

  /// Do the simplest possible inference of argument and result activities, or
  /// take the user's explicit override if provided
  void initializeArgAndResActivities(
      FunctionOpInterface callee,
      MutableArrayRef<enzyme::Activity> argActivities,
      MutableArrayRef<enzyme::Activity> resActivities) const {
    for (const auto &[idx, argType] :
         llvm::enumerate(callee.getArgumentTypes())) {
      if (inactiveArgs || argType.isIntOrIndex())
        argActivities[idx] = enzyme::Activity::enzyme_const;
      else if (isa<FloatType>(argType))
        argActivities[idx] = enzyme::Activity::enzyme_out;
      else
        argActivities[idx] = enzyme::Activity::enzyme_dup;
    }

    for (const auto &[idx, resType] :
         llvm::enumerate(callee.getResultTypes())) {
      if (duplicatedRet)
        resActivities[idx] = (enzyme::Activity::enzyme_dup);
      else if (isa<FloatType>(resType))
        resActivities[idx] = (enzyme::Activity::enzyme_out);
      else
        resActivities[idx] = (enzyme::Activity::enzyme_const);
    }
  }

  void runOnOperation() override {
    auto moduleOp = cast<ModuleOp>(getOperation());

    if (funcToAnalyze == "") {
      if (failed(inferEnzymeAutodiffOps(moduleOp))) {
        signalPassFailure();
      }
      return;
    }

    auto *op = moduleOp.lookupSymbol(funcToAnalyze);
    if (!op) {
      moduleOp.emitError() << "Failed to find requested function "
                           << funcToAnalyze;
      return signalPassFailure();
    }

    if (!isa<FunctionOpInterface>(op)) {
      moduleOp.emitError() << "Operation " << funcToAnalyze
                           << " was not a FunctionOpInterface";
      return signalPassFailure();
    }

    auto callee = cast<FunctionOpInterface>(op);
    SmallVector<enzyme::Activity> argActivities{callee.getNumArguments()},
        resultActivities{callee.getNumResults()};
    initializeArgAndResActivities(callee, argActivities, resultActivities);

    enzyme::runDataFlowActivityAnalysis(callee, argActivities);
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
