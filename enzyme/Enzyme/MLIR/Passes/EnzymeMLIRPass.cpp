//===- EnzymeMLIRPass.cpp - Replace calls with their derivatives ------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower gpu kernels in NVVM/gpu dialects into
// a generic parallel for representation
//===----------------------------------------------------------------------===//

#include "Dialect/Ops.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace {
struct DifferentiatePass : public DifferentiatePassBase<DifferentiatePass> {
  MEnzymeLogic Logic;

  void runOnOperation() override;

  static DIFFE_TYPE mode_from_fn(FunctionOpInterface fn, DerivativeMode mode) {
    DIFFE_TYPE retType = DIFFE_TYPE::CONSTANT;
    if (fn.getNumResults() != 0) {
      if (mode == DerivativeMode::ReverseModeCombined)
        retType = DIFFE_TYPE::OUT_DIFF;
      else
        retType = DIFFE_TYPE::DUP_ARG;
    }
    return retType;
  }

  template <typename T>
  LogicalResult HandleAutoDiff(SymbolTableCollection &symbolTable, T CI) {
    std::vector<DIFFE_TYPE> constants;
    SmallVector<mlir::Value, 2> args;

    size_t truei = 0;
    auto activityAttr = CI.getActivity();

    for (unsigned i = 0; i < CI.getInputs().size(); ++i) {
      mlir::Value res = CI.getInputs()[i];

      auto mop = activityAttr[truei];
      auto iattr = cast<mlir::enzyme::ActivityAttr>(mop);
      DIFFE_TYPE ty = (DIFFE_TYPE)(iattr.getValue());

      constants.push_back(ty);
      args.push_back(res);
      if (ty == DIFFE_TYPE::DUP_ARG || ty == DIFFE_TYPE::DUP_NONEED) {
        ++i;
        res = CI.getInputs()[i];
        args.push_back(res);
      }

      truei++;
    }

    auto *symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
    auto fn = cast<FunctionOpInterface>(symbolOp);

    auto mode = DerivativeMode::ForwardMode;
    DIFFE_TYPE retType = mode_from_fn(fn, mode);

    MTypeAnalysis TA;
    auto type_args = TA.getAnalyzedTypeInfo(fn);
    bool freeMemory = true;
    size_t width = 1;

    std::vector<bool> volatile_args;
    for (auto &a : fn.getFunctionBody().getArguments()) {
      (void)a;
      volatile_args.push_back(!(mode == DerivativeMode::ReverseModeCombined));
    }

    FunctionOpInterface newFunc = Logic.CreateForwardDiff(
        fn, retType, constants, TA,
        /*should return*/ false, mode, freeMemory, width,
        /*addedType*/ nullptr, type_args, volatile_args,
        /*augmented*/ nullptr);
    if (!newFunc)
      return failure();

    OpBuilder builder(CI);
    auto dCI = builder.create<func::CallOp>(CI.getLoc(), newFunc.getName(),
                                            newFunc.getResultTypes(), args);
    CI.replaceAllUsesWith(dCI);
    CI->erase();
    return success();
  }

  template <typename T>
  LogicalResult HandleAutoDiffReverse(SymbolTableCollection &symbolTable,
                                      T CI) {
    std::vector<DIFFE_TYPE> constants;
    SmallVector<mlir::Value, 2> args;

    size_t truei = 0;
    auto activityAttr = CI.getActivity();

    for (unsigned i = 0; i < CI.getInputs().size() - 1; ++i) {
      mlir::Value res = CI.getInputs()[i];

      auto mop = activityAttr[truei];
      auto iattr = cast<mlir::enzyme::ActivityAttr>(mop);
      DIFFE_TYPE ty = (DIFFE_TYPE)(iattr.getValue());

      constants.push_back(ty);
      args.push_back(res);
      if (ty == DIFFE_TYPE::DUP_ARG || ty == DIFFE_TYPE::DUP_NONEED) {
        ++i;
        res = CI.getInputs()[i];
        args.push_back(res);
      }

      truei++;
    }

    auto *symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
    auto fn = cast<FunctionOpInterface>(symbolOp);

    auto mode = DerivativeMode::ReverseModeCombined;
    DIFFE_TYPE retType = mode_from_fn(fn, mode);

    // Add the return gradient
    mlir::Value res = CI.getInputs()[CI.getInputs().size() - 1];
    args.push_back(res);

    MTypeAnalysis TA;
    auto type_args = TA.getAnalyzedTypeInfo(fn);
    bool freeMemory = true;
    size_t width = 1;

    std::vector<bool> volatile_args;
    for (auto &a : fn.getFunctionBody().getArguments()) {
      (void)a;
      volatile_args.push_back(!(mode == DerivativeMode::ReverseModeCombined));
    }

    FunctionOpInterface newFunc = Logic.CreateReverseDiff(
        fn, retType, constants, TA,
        /*should return*/ false, mode, freeMemory, width,
        /*addedType*/ nullptr, type_args, volatile_args,
        /*augmented*/ nullptr, symbolTable);
    if (!newFunc)
      return failure();

    OpBuilder builder(CI);
    auto dCI = builder.create<func::CallOp>(CI.getLoc(), newFunc.getName(),
                                            newFunc.getResultTypes(), args);
    CI.replaceAllUsesWith(dCI);
    CI->erase();
    return success();
  }

  void lowerEnzymeCalls(SymbolTableCollection &symbolTable,
                        FunctionOpInterface op) {
    {
      SmallVector<Operation *> toLower;
      op->walk([&](enzyme::ForwardDiffOp dop) {
        auto *symbolOp =
            symbolTable.lookupNearestSymbolFrom(dop, dop.getFnAttr());
        auto callableOp = cast<FunctionOpInterface>(symbolOp);

        lowerEnzymeCalls(symbolTable, callableOp);
        toLower.push_back(dop);
      });

      for (auto T : toLower) {
        if (auto F = dyn_cast<enzyme::ForwardDiffOp>(T)) {
          auto res = HandleAutoDiff(symbolTable, F);
          if (!res.succeeded()) {
            signalPassFailure();
            return;
          }
        } else {
          llvm_unreachable("Illegal type");
        }
      }
    };

    {
      SmallVector<Operation *> toLower;
      op->walk([&](enzyme::AutoDiffOp dop) {
        auto *symbolOp =
            symbolTable.lookupNearestSymbolFrom(dop, dop.getFnAttr());
        auto callableOp = cast<FunctionOpInterface>(symbolOp);

        lowerEnzymeCalls(symbolTable, callableOp);
        toLower.push_back(dop);
      });

      for (auto T : toLower) {
        if (auto F = dyn_cast<enzyme::AutoDiffOp>(T)) {
          auto res = HandleAutoDiffReverse(symbolTable, F);
          if (!res.succeeded()) {
            signalPassFailure();
            return;
          }
        } else {
          llvm_unreachable("Illegal type");
        }
      }
    }
  };
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createDifferentiatePass() {
  return std::make_unique<DifferentiatePass>();
}
} // namespace enzyme
} // namespace mlir

void DifferentiatePass::runOnOperation() {
  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(getOperation());
  getOperation()->walk(
      [&](FunctionOpInterface op) { lowerEnzymeCalls(symbolTable, op); });
}
