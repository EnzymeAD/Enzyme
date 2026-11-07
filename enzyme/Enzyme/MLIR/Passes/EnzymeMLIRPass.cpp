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
#include "Passes/RemovalUtils.h"

#include "Dialect/LLVMExt/LLVMExt.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_DIFFERENTIATEPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {
struct DifferentiatePass
    : public enzyme::impl::DifferentiatePassBase<DifferentiatePass> {
  using DifferentiatePassBase::DifferentiatePassBase;

  void runOnOperation() override;

  // Whether a differentiation failed (its errors are already reported).
  bool diffFailed = false;

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::OpPassManager pm;
    mlir::LogicalResult result = mlir::parsePassPipeline(postpasses, pm);
    if (!mlir::failed(result)) {
      pm.getDependentDialects(registry);
    }

    // Derivative rules may build math and llvm_ext ops absent from the input.
    registry.insert<mlir::arith::ArithDialect, mlir::complex::ComplexDialect,
                    mlir::cf::ControlFlowDialect, mlir::tensor::TensorDialect,
                    mlir::memref::MemRefDialect, mlir::math::MathDialect,
                    mlir::enzyme::EnzymeDialect, mlir::LLVM::LLVMDialect,
                    mlir::enzyme::llvm_ext::LLVMExtDialect>();
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
  LogicalResult HandleAutoDiff(MEnzymeLogic &Logic,
                               SymbolTableCollection &symbolTable, T CI) {
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
    bool omp = false;
    size_t width = CI.getWidth();

    std::vector<bool> overwritten_args;
    for (auto &a : fn.getFunctionBody().getArguments()) {
      (void)a;
      overwritten_args.push_back(
          !(mode == DerivativeMode::ReverseModeCombined));
    }

    FunctionOpInterface newFunc = Logic.CreateForwardDiff(
        fn, retType, constants, TA, returnPrimals, mode, freeMemory, width,
        /*addedType*/ nullptr, type_args, overwritten_args,
        /*augmented*/ nullptr, omp, postpasses, verifyPostPasses,
        CI.getStrongZero());
    if (!newFunc)
      return failure();

    OpBuilder builder(CI);
    // Ask the function how it is called, as the reverse handler does: what is
    // being differentiated is often an llvm.func, and a func.call to one of
    // those is not a call at all.
    auto iface = dyn_cast<AutoDiffFunctionInterface>(newFunc.getOperation());
    if (!iface) {
      newFunc.getOperation()->emitError()
          << "this function operation does not implement "
             "AutoDiffFunctionInterface";
      return failure();
    }
    Operation *dCI = iface.createCall(builder, CI.getLoc(), args);
    if (dCI->getNumResults() != CI.getNumResults()) {
      CI.emitError() << "Incorrect number of results for enzyme operation: "
                     << *CI << " expected " << *dCI;
      return failure();
    }
    CI.replaceAllUsesWith(dCI);
    CI->erase();
    return success();
  }

  template <typename T>
  LogicalResult HandleAutoDiffReverse(MEnzymeLogic &Logic,
                                      SymbolTableCollection &symbolTable,
                                      T CI) {

    auto *symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
    auto fn = cast<FunctionOpInterface>(symbolOp);
    assert(fn);
    if (CI.getActivity().size() != fn.getNumArguments()) {
      llvm::errs() << "Incorrect number of argument activities on autodiff op"
                   << "CI: " << CI << ", expected " << fn.getNumArguments()
                   << " found " << CI.getActivity().size() << "\n";
      return failure();
    }
    if (CI.getRetActivity().size() != fn.getNumResults()) {
      llvm::errs() << "Incorrect number of result activities on autodiff op"
                   << "CI: " << CI << ", expected " << fn.getNumResults()
                   << " found " << CI.getRetActivity().size() << "\n";
      return failure();
    }

    std::vector<DIFFE_TYPE> arg_activities;
    SmallVector<mlir::Value, 2> args;

    size_t call_idx = 0;
    {
      for (auto act : CI.getActivity()) {
        if (call_idx >= CI.getInputs().size()) {
          llvm::errs() << "Too few arguments to autodiff op"
                       << " CI: " << CI << "\n";
          return failure();
        }
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
          if (call_idx >= CI.getInputs().size()) {
            llvm::errs() << "Too few arguments to autodiff op"
                         << "CI: " << CI << "\n";
            return failure();
          }
          res = CI.getInputs()[call_idx];
          ++call_idx;
          args.push_back(res);
        }
      }
    }

    bool omp = false;
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
        if (call_idx >= CI.getInputs().size()) {
          llvm::errs() << "Too few arguments to autodiff op"
                       << "CI: " << CI << "\n";
          return failure();
        }
        mlir::Value res = CI.getInputs()[call_idx];
        ++call_idx;
        args.push_back(res);
      }
    }

    MTypeAnalysis TA;
    auto type_args = TA.getAnalyzedTypeInfo(fn);
    bool freeMemory = true;
    size_t width = CI.getWidth();

    std::vector<bool> overwritten_args;
    for (auto &a : fn.getFunctionBody().getArguments()) {
      (void)a;
      overwritten_args.push_back(
          !(mode == DerivativeMode::ReverseModeCombined));
    }

    FunctionOpInterface newFunc = Logic.CreateReverseDiff(
        fn, retType, arg_activities, TA, returnPrimals, returnShadows, mode,
        freeMemory, CI.getAtomicAdd(), width,
        /*addedType*/ nullptr, type_args, overwritten_args,
        /*augmented*/ nullptr, omp, postpasses, verifyPostPasses,
        CI.getStrongZero(), markReadonly);
    if (!newFunc)
      return failure();

    OpBuilder builder(CI);
    if (auto iface =
            dyn_cast<AutoDiffFunctionInterface>(newFunc.getOperation())) {
      auto dCI = iface.createCall(builder, CI.getLoc(), args);
      CI.replaceAllUsesWith(dCI);
    } else {
      newFunc.getOperation()->emitError()
          << "this function operation does not implement "
             "AutoDiffFunctionInterface";
      return failure();
    }
    CI->erase();
    return success();
  }

  LogicalResult HandleSplitModeAutoDiff(MEnzymeLogic &Logic,
                                        SymbolTableCollection &symbolTable,
                                        enzyme::AutoDiffSplitModePrimalOp CI) {
    auto tape = CI.getTape();

    auto *symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
    auto fn = cast<FunctionOpInterface>(symbolOp);
    assert(fn);
    if (CI.getActivity().size() != fn.getNumArguments()) {
      llvm::errs() << "Incorrect number of argument activities on autodiff op"
                   << " CI: " << CI << ", expected " << fn.getNumArguments()
                   << " found " << CI.getActivity().size() << "\n";
      return failure();
    }
    if (CI.getRetActivity().size() != fn.getNumResults()) {
      llvm::errs() << "Incorrect number of result activities on autodiff op"
                   << " CI: " << CI << ", expected " << fn.getNumResults()
                   << " found " << CI.getRetActivity().size() << "\n";
      return failure();
    }

    std::vector<DIFFE_TYPE> arg_activities;
    SmallVector<mlir::Value, 2> args;

    size_t call_idx = 0;
    {
      for (auto act : CI.getActivity()) {
        if (call_idx >= CI.getInputs().size()) {
          llvm::errs() << "Too few arguments to autodiff op"
                       << " CI: " << CI << "\n";
          return failure();
        }
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
      }
    }

    bool omp = false;
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
    }

    std::vector<bool> volatile_args(
        fn.getNumArguments(), !(mode == DerivativeMode::ReverseModeCombined));

    MTypeAnalysis TA;
    auto type_args = TA.getAnalyzedTypeInfo(fn);
    bool freeMemory = true;
    size_t width = CI.getWidth();

    auto ruleToCall = Logic.CreateSplitModeDiff(
        fn, retType, arg_activities, TA, returnPrimals, returnShadows, mode,
        freeMemory, width,
        /*addedType*/ nullptr, type_args, volatile_args,
        /*augmented*/ nullptr, omp, postpasses, verifyPostPasses,
        CI.getStrongZero());

    if (!ruleToCall)
      return CI->emitError()
             << "failed to create reverse-mode adjoint for callee "
             << fn.getNameAttr() << "\n";
    auto rule =
        SymbolTable::lookupNearestSymbolFrom<enzyme::CustomReverseRuleOp>(
            CI, ruleToCall);
    if (!rule || failed(finalizeCustomReverseRule(rule)))
      return failure();

    // The rule's augmented primal returns its caches as typed values. The
    // tape this op hands to user code stands for them: follow it through
    // pushes and pops to each reverse, which takes the values back.
    OpBuilder builder(CI);
    auto primalCall = enzyme::CallAugmentedPrimalOp::create(
        builder, CI.getLoc(), CI.getOutputs().getTypes(),
        getCustomReverseRuleCacheTypes(rule), ruleToCall, CI.getOperands());
    for (auto [oldRes, newRes] :
         llvm::zip_equal(CI.getOutputs(), primalCall.getOutputs())) {
      oldRes.replaceAllUsesWith(newRes);
    }

    SetVector<Operation *> toDelete;
    llvm::DenseMap<Value, SmallVector<Value>> tapeToCaches;
    tapeToCaches[tape] = SmallVector<Value>(primalCall.getCaches().begin(),
                                            primalCall.getCaches().end());

    SmallVector<Value, 2> tapeWorklist = {tape};
    while (!tapeWorklist.empty()) {
      Value curTape = tapeWorklist.pop_back_val();
      SmallVector<Value> values = tapeToCaches[curTape];
      for (auto tapeUser : curTape.getUsers()) {
        if (auto revCall =
                dyn_cast<enzyme::AutoDiffSplitModeReverseOp>(tapeUser)) {
          OpBuilder builder(revCall);
          auto newRevCall = enzyme::CallCustomReverseOp::create(
              builder, revCall.getLoc(), revCall.getResultTypes(), ruleToCall,
              revCall.getInputs(), values);
          revCall.replaceAllUsesWith(newRevCall.getResults());

          toDelete.insert(revCall);
        } else if (auto pushOp = dyn_cast<enzyme::PushOp>(tapeUser)) {
          assert(pushOp.getValue() == curTape);
          CacheInfo info(pushOp.getCache());

          // One cache per value the tape stands for.
          OpBuilder builder(info.initOp);
          SmallVector<Value> newCaches;
          for (Value v : values)
            newCaches.push_back(enzyme::InitOp::create(
                builder, info.initOp->getLoc(),
                enzyme::CacheType::get(v.getContext(), v.getType())));

          builder.setInsertionPoint(info.pushOp);
          for (auto [v, c] : llvm::zip_equal(values, newCaches))
            enzyme::PushOp::create(builder, info.pushOp->getLoc(), c, v);

          builder.setInsertionPoint(info.popOp);
          SmallVector<Value> popped;
          for (auto [v, c] : llvm::zip_equal(values, newCaches))
            popped.push_back(enzyme::PopOp::create(
                builder, info.popOp->getLoc(), v.getType(), c));

          Value poppedTape = info.popOp.getResult();
          tapeToCaches[poppedTape] = popped;
          tapeWorklist.push_back(poppedTape);

          toDelete.insert(info.initOp);
          toDelete.insert(info.pushOp);
          toDelete.insert(info.popOp);
        } else {
          tapeUser->emitError()
              << "todo: support tape going through this operation";
          return failure();
        }
      }
    }

    // Everything that carried the tape goes, along with the op itself.
    auto worklist = toDelete.takeVector();
    for (Operation *op : worklist)
      op->dropAllUses();
    for (Operation *op : worklist)
      op->erase();
    CI->erase();

    return success();
  }

  void lowerEnzymeCalls(MEnzymeLogic &Logic, SymbolTableCollection &symbolTable,
                        FunctionOpInterface op) {
    {
      SmallVector<Operation *> toLower;
      op->walk([&](enzyme::ForwardDiffOp dop) {
        auto *symbolOp =
            symbolTable.lookupNearestSymbolFrom(dop, dop.getFnAttr());
        auto callableOp = cast<FunctionOpInterface>(symbolOp);

        lowerEnzymeCalls(Logic, symbolTable, callableOp);
        toLower.push_back(dop);
      });

      for (auto T : toLower) {
        if (auto F = dyn_cast<enzyme::ForwardDiffOp>(T)) {
          auto res = HandleAutoDiff(Logic, symbolTable, F);
          if (!res.succeeded()) {
            diffFailed = true;
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

        lowerEnzymeCalls(Logic, symbolTable, callableOp);
        toLower.push_back(dop);
      });

      for (auto T : toLower) {
        if (auto F = dyn_cast<enzyme::AutoDiffOp>(T)) {
          auto res = HandleAutoDiffReverse(Logic, symbolTable, F);
          if (!res.succeeded()) {
            diffFailed = true;
            signalPassFailure();
            return;
          }
        } else {
          llvm_unreachable("Illegal type");
        }
      }
    }

    {
      SmallVector<Operation *> toLower;
      op->walk([&](enzyme::AutoDiffSplitModePrimalOp dop) {
        auto *symbolOp =
            symbolTable.lookupNearestSymbolFrom(dop, dop.getFnAttr());
        auto callableOp = cast<FunctionOpInterface>(symbolOp);

        lowerEnzymeCalls(Logic, symbolTable, callableOp);
        toLower.push_back(dop);
      });

      for (auto T : toLower) {
        if (auto F = dyn_cast<enzyme::AutoDiffSplitModePrimalOp>(T)) {
          auto res = HandleSplitModeAutoDiff(Logic, symbolTable, F);
          if (!res.succeeded()) {
            diffFailed = true;
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

void DifferentiatePass::runOnOperation() {
  MEnzymeLogic Logic(dataflowActivity);
  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(getOperation());
  getOperation()->walk([&](FunctionOpInterface op) {
    lowerEnzymeCalls(Logic, symbolTable, op);
  });
  getOperation()->walk([&](FunctionOpInterface op) { removeSummaries(op); });

  // Lower custom-rule calls for pipelines that only handle standard dialects.
  if (lowerCustomRules && !diffFailed &&
      failed(lowerCustomReverseRulesToFunc(getOperation())))
    signalPassFailure();
}
