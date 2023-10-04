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

    if (funcsToAnalyze.empty()) {
      moduleOp.walk([this](FunctionOpInterface callee) {
        if (callee.isPrivate())
          return;

        SmallVector<enzyme::Activity> argActivities{callee.getNumArguments()},
            resultActivities{callee.getNumResults()};
        initializeArgAndResActivities(callee, argActivities, resultActivities);

        enzyme::runDataFlowActivityAnalysis(callee, argActivities,
                                            /*print=*/true, verbose);
      });
      return;
    }

    for (std::string funcName : funcsToAnalyze) {
      Operation *op = moduleOp.lookupSymbol(funcName);
      if (!op) {
        continue;
      }

      if (!isa<FunctionOpInterface>(op)) {
        moduleOp.emitError()
            << "Operation " << funcName << " was not a FunctionOpInterface";
        return signalPassFailure();
      }

      auto callee = cast<FunctionOpInterface>(op);
      SmallVector<enzyme::Activity> argActivities{callee.getNumArguments()},
          resultActivities{callee.getNumResults()};
      initializeArgAndResActivities(callee, argActivities, resultActivities);

      enzyme::runDataFlowActivityAnalysis(callee, argActivities,
                                          /*print=*/true, verbose);
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
