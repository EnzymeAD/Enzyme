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


  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::PassManager pm(nf->getContext());
    std::string error_message;
    //llvm::raw_string_ostream error_stream(error_message);
    mlir::LogicalResult result =
        mlir::parsePassPipeline(postpasses, pm);
    if (!mlir::failed(result)) {
      pm.getDependentDialects(registry);
    }

    registry.insert<mlir::arith::ArithDialect, mlir::complex::ComplexDialect,
                    mlir::cf::ControlFlowDialect, mlir::tensor::TensorDialect>();
  }

  static std::vector<DIFFE_TYPE> mode_from_fn(FunctionOpInterface fn,
                                              DerivativeMode mode) {
    std::vector<DIFFE_TYPE> retTypes;
    for (auto ty : fn.getResultTypes()) {
      if (isa<IntegerType>(ty)) {
        retTypes.push_back(DIFFE_TYPE::CONSTANT);
        continue;
      }

      if (mode == DerivativeMode::ReverseModeCombined)
        retTypes.push_back(DIFFE_TYPE::OUT_DIFF);
      else
        retTypes.push_back(DIFFE_TYPE::DUP_ARG);
    }
    return retTypes;
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
      DIFFE_TYPE ty;

      switch (iattr.getValue()) {
      case mlir::enzyme::Activity::enzyme_active:
        ty = DIFFE_TYPE::OUT_DIFF;
        break;
      case mlir::enzyme::Activity::enzyme_dup:
        ty = DIFFE_TYPE::DUP_ARG;
        break;
      case mlir::enzyme::Activity::enzyme_const:
        ty = DIFFE_TYPE::CONSTANT;
        break;
      case mlir::enzyme::Activity::enzyme_dupnoneed:
        ty = DIFFE_TYPE::DUP_NONEED;
        break;
      case mlir::enzyme::Activity::enzyme_activenoneed:
        ty = DIFFE_TYPE::OUT_DIFF;
        assert(0 && "unsupported arg activenoneed");
        break;
      case mlir::enzyme::Activity::enzyme_constnoneed:
        ty = DIFFE_TYPE::CONSTANT;
        assert(0 && "unsupported arg constnoneed");
        break;
      }

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
    std::vector<DIFFE_TYPE> retType;

    std::vector<bool> returnPrimals;
    for (auto act : CI.getRetActivity()) {
      auto iattr = cast<mlir::enzyme::ActivityAttr>(act);
      auto val = iattr.getValue();
      DIFFE_TYPE ty;
      bool primalNeeded = true;
      switch (val) {
      case mlir::enzyme::Activity::enzyme_active:
        ty = DIFFE_TYPE::OUT_DIFF;
        break;
      case mlir::enzyme::Activity::enzyme_dup:
        ty = DIFFE_TYPE::DUP_ARG;
        break;
      case mlir::enzyme::Activity::enzyme_const:
        ty = DIFFE_TYPE::CONSTANT;
        break;
      case mlir::enzyme::Activity::enzyme_dupnoneed:
        ty = DIFFE_TYPE::DUP_NONEED;
        primalNeeded = false;
        break;
      case mlir::enzyme::Activity::enzyme_activenoneed:
        ty = DIFFE_TYPE::OUT_DIFF;
        primalNeeded = false;
        break;
      case mlir::enzyme::Activity::enzyme_constnoneed:
        ty = DIFFE_TYPE::CONSTANT;
        primalNeeded = false;
        break;
      }
      retType.push_back(ty);
      returnPrimals.push_back(primalNeeded);
    }

    MTypeAnalysis TA;
    auto type_args = TA.getAnalyzedTypeInfo(fn);
    bool freeMemory = true;
    size_t width = CI.getWidth();

    std::vector<bool> volatile_args;
    for (auto &a : fn.getFunctionBody().getArguments()) {
      (void)a;
      volatile_args.push_back(!(mode == DerivativeMode::ReverseModeCombined));
    }

    FunctionOpInterface newFunc = Logic.CreateForwardDiff(
        fn, retType, constants, TA, returnPrimals, mode, freeMemory, width,
        /*addedType*/ nullptr, type_args, volatile_args,
        /*augmented*/ nullptr, postpasses);
    if (!newFunc)
      return failure();

    OpBuilder builder(CI);
    auto dCI = builder.create<func::CallOp>(CI.getLoc(), newFunc.getName(),
                                            newFunc.getResultTypes(), args);
    if (dCI.getNumResults() != CI.getNumResults()) {
      CI.emitError() << "Incorrect number of results for enzyme operation: "
                     << *CI << " expected " << *dCI;
      return failure();
    }
    CI.replaceAllUsesWith(dCI);
    CI->erase();
    return success();
  }

  template <typename T>
  LogicalResult HandleAutoDiffReverse(SymbolTableCollection &symbolTable,
                                      T CI) {
    std::vector<DIFFE_TYPE> arg_activities;
    SmallVector<mlir::Value, 2> args;

    size_t call_idx = 0;
    {
      for (auto act : CI.getActivity()) {
        mlir::Value res = CI.getInputs()[call_idx];
        ++call_idx;

        auto iattr = cast<mlir::enzyme::ActivityAttr>(act);
        auto val = iattr.getValue();
        DIFFE_TYPE ty;
        switch (val) {
        case mlir::enzyme::Activity::enzyme_active:
          ty = DIFFE_TYPE::OUT_DIFF;
          break;
        case mlir::enzyme::Activity::enzyme_dup:
          ty = DIFFE_TYPE::DUP_ARG;
          break;
        case mlir::enzyme::Activity::enzyme_const:
          ty = DIFFE_TYPE::CONSTANT;
          break;
        case mlir::enzyme::Activity::enzyme_dupnoneed:
          ty = DIFFE_TYPE::DUP_NONEED;
          break;
        case mlir::enzyme::Activity::enzyme_activenoneed:
          ty = DIFFE_TYPE::OUT_DIFF;
          assert(0 && "unsupported arg activenoneed");
          break;
        case mlir::enzyme::Activity::enzyme_constnoneed:
          ty = DIFFE_TYPE::CONSTANT;
          assert(0 && "unsupported arg constnoneed");
          break;
        }
        arg_activities.push_back(ty);
        args.push_back(res);
        if (ty == DIFFE_TYPE::DUP_ARG || ty == DIFFE_TYPE::DUP_NONEED) {
          res = CI.getInputs()[call_idx];
          ++call_idx;
          args.push_back(res);
        }
      }
    }

    auto *symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
    auto fn = cast<FunctionOpInterface>(symbolOp);

    auto mode = DerivativeMode::ReverseModeCombined;
    std::vector<DIFFE_TYPE> retType;
    std::vector<bool> returnPrimals;
    std::vector<bool> returnShadows;

    // Add the return gradient
    for (auto act : CI.getRetActivity()) {
      auto iattr = cast<mlir::enzyme::ActivityAttr>(act);
      auto val = iattr.getValue();
      DIFFE_TYPE ty;
      bool primalNeeded = true;
      switch (val) {
      case mlir::enzyme::Activity::enzyme_active:
        ty = DIFFE_TYPE::OUT_DIFF;
        break;
      case mlir::enzyme::Activity::enzyme_dup:
        ty = DIFFE_TYPE::DUP_ARG;
        break;
      case mlir::enzyme::Activity::enzyme_const:
        ty = DIFFE_TYPE::CONSTANT;
        break;
      case mlir::enzyme::Activity::enzyme_dupnoneed:
        ty = DIFFE_TYPE::DUP_NONEED;
        primalNeeded = false;
        break;
      case mlir::enzyme::Activity::enzyme_activenoneed:
        ty = DIFFE_TYPE::OUT_DIFF;
        primalNeeded = false;
        break;
      case mlir::enzyme::Activity::enzyme_constnoneed:
        ty = DIFFE_TYPE::CONSTANT;
        primalNeeded = false;
        break;
      }
      retType.push_back(ty);
      returnPrimals.push_back(primalNeeded);
      returnShadows.push_back(false);
      if (ty == DIFFE_TYPE::OUT_DIFF) {
        mlir::Value res = CI.getInputs()[call_idx];
        ++call_idx;
        args.push_back(res);
      }
    }

    MTypeAnalysis TA;
    auto type_args = TA.getAnalyzedTypeInfo(fn);
    bool freeMemory = true;
    size_t width = CI.getWidth();

    std::vector<bool> volatile_args;
    for (auto &a : fn.getFunctionBody().getArguments()) {
      (void)a;
      volatile_args.push_back(!(mode == DerivativeMode::ReverseModeCombined));
    }

    FunctionOpInterface newFunc =
        Logic.CreateReverseDiff(fn, retType, arg_activities, TA, returnPrimals,
                                returnShadows, mode, freeMemory, width,
                                /*addedType*/ nullptr, type_args, volatile_args,
                                /*augmented*/ nullptr, postpasses);
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
