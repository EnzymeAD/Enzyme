//===- PrintAliasAnalysis.cpp - Pass to print alias analysis --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to print the results of running alias
// analysis.
//===----------------------------------------------------------------------===//

#include "Analysis/AliasAnalysis.h"
#include "Dialect/Ops.h"
#include "Passes/PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/IR/FunctionInterfaces.h"

using namespace mlir;

namespace {
using llvm::errs;

struct PrintAliasAnalysisPass
    : public enzyme::PrintAliasAnalysisPassBase<PrintAliasAnalysisPass> {

  void runOnOperation() override {
    DataFlowSolver solver;

    solver.load<dataflow::enzyme::AliasAnalysis>();
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      return signalPassFailure();
    }

    SmallVector<std::pair<Attribute, Value>> taggedPointers;
    getOperation()->walk([&](FunctionOpInterface funcOp) {
      for (auto arg : funcOp.getArguments()) {
        if (funcOp.getArgAttr(arg.getArgNumber(), "enzyme.tag")) {
          taggedPointers.push_back(
              {funcOp.getArgAttr(arg.getArgNumber(), "enzyme.tag"), arg});
        }
      }

      funcOp.walk([&](Operation *op) {
        if (op->hasAttr("tag")) {
          for (OpResult result : op->getResults()) {
            taggedPointers.push_back({op->getAttr("tag"), result});
          }
        }
      });
    });

    for (const auto &[tag, value] : taggedPointers) {
      const auto *state =
          solver.lookupState<dataflow::enzyme::AliasClassLattice>(value);
      if (state) {
        errs() << "tag " << tag
               << " canonical allocation: " << state->getCanonicalAllocation()
               << "\n";
      }
    }

    // Compare all tagged pointers
    for (unsigned i = 0; i < taggedPointers.size() - 1; i++) {
      for (unsigned j = i + 1; j < taggedPointers.size(); j++) {
        const auto &[tagA, a] = taggedPointers[i];
        const auto &[tagB, b] = taggedPointers[j];

        const auto *lhs =
            solver.lookupState<dataflow::enzyme::AliasClassLattice>(a);
        const auto *rhs =
            solver.lookupState<dataflow::enzyme::AliasClassLattice>(b);
        if (!(lhs && rhs))
          continue;

        errs() << tagA << " and " << tagB << ": " << lhs->alias(*rhs) << "\n";
      }
    }

    getOperation()->walk([&solver](Operation *op) {
      if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
        for (auto arg : funcOp.getArguments()) {
          auto *state =
              solver.lookupState<dataflow::enzyme::AliasClassLattice>(arg);
          if (state) {
            if (state->isEntry)
              funcOp.setArgAttr(arg.getArgNumber(), "enzyme.ac_entry",
                                UnitAttr::get(funcOp.getContext()));

            for (auto aliasClass : state->aliasClasses)
              funcOp.setArgAttr(arg.getArgNumber(), "enzyme.ac", aliasClass);
          }
        }
      }
      if (op->hasAttr("tag")) {
        for (OpResult result : op->getResults()) {
          auto *state =
              solver.lookupState<dataflow::enzyme::AliasClassLattice>(result);
          if (state) {
            for (auto aliasClass : state->aliasClasses) {
              op->setAttr("ac", aliasClass);
            }
          }
        }
      }
    });
  }
};
} // namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createPrintAliasAnalysisPass() {
  return std::make_unique<PrintAliasAnalysisPass>();
}
} // namespace enzyme
} // namespace mlir
