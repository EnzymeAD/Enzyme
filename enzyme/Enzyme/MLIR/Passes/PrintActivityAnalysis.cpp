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
//
//===----------------------------------------------------------------------===//
#include "Analysis/ActivityAnalysis.h"
#include "Analysis/ActivityAnnotations.h"
#include "Analysis/DataFlowActivityAnalysis.h"
#include "Dialect/Ops.h"
#include "Interfaces/EnzymeLogic.h"
#include "Passes/PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/TypeSwitch.h"

#include "llvm/Demangle/Demangle.h"

using namespace mlir;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_PRINTACTIVITYANALYSISPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

enzyme::Activity getDefaultActivity(Type argType) {
  if (argType.isIntOrIndex())
    return enzyme::Activity::enzyme_const;

  if (isa<FloatType, ComplexType>(argType))
    return enzyme::Activity::enzyme_active;

  if (auto T = dyn_cast<TensorType>(argType))
    return getDefaultActivity(T.getElementType());

  if (isa<LLVM::LLVMPointerType, MemRefType>(argType))
    return enzyme::Activity::enzyme_dup;

  return enzyme::Activity::enzyme_const;
}

struct PrintActivityAnalysisPass
    : public enzyme::impl::PrintActivityAnalysisPassBase<
          PrintActivityAnalysisPass> {
  using PrintActivityAnalysisPassBase::PrintActivityAnalysisPassBase;

  /// Do the simplest possible inference of argument and result activities, or
  /// take the user's explicit override if provided
  void initializeArgAndResActivities(
      FunctionOpInterface callee,
      MutableArrayRef<enzyme::Activity> argActivities,
      MutableArrayRef<enzyme::Activity> resActivities) const {
    for (const auto &[idx, argType] :
         llvm::enumerate(callee.getArgumentTypes())) {
      if (callee.getArgAttr(idx, "enzyme.const") || inactiveArgs)
        argActivities[idx] = enzyme::Activity::enzyme_const;
      else
        argActivities[idx] = getDefaultActivity(argType);
    }

    for (const auto &[idx, resType] :
         llvm::enumerate(callee.getResultTypes())) {
      if (duplicatedRet)
        resActivities[idx] = (enzyme::Activity::enzyme_dup);
      else
        resActivities[idx] = getDefaultActivity(resType);
    }
  }

  void inferArgActivitiesFromEnzymeAutodiff(
      FunctionOpInterface callee, CallOpInterface autodiff_call,
      MutableArrayRef<enzyme::Activity> argActivities,
      MutableArrayRef<enzyme::Activity> resultActivities) {
    unsigned argIdx = 1;
    for (const auto &[paramIdx, paramType] :
         llvm::enumerate(callee.getArgumentTypes())) {
      Value arg = autodiff_call.getArgOperands()[argIdx];
      if (auto loadOp =
              dyn_cast_if_present<LLVM::LoadOp>(arg.getDefiningOp())) {
        if (auto addressOf = dyn_cast_if_present<LLVM::AddressOfOp>(
                loadOp.getAddr().getDefiningOp())) {
          if (addressOf.getGlobalName() == "enzyme_const") {
            argActivities[paramIdx] = enzyme::Activity::enzyme_const;
          } else if (addressOf.getGlobalName() == "enzyme_dup") {
            argActivities[paramIdx] = enzyme::Activity::enzyme_dup;
            // Skip the shadow
            argIdx++;
          } else if (addressOf.getGlobalName() == "enzyme_dupnoneed") {
            argActivities[paramIdx] = enzyme::Activity::enzyme_dupnoneed;
            // Skip the shadow
            argIdx++;
          }
        }
        // Skip the enzyme_* annotation
        argIdx++;
      } else {
        argActivities[paramIdx] =
            llvm::TypeSwitch<Type, enzyme::Activity>(paramType)
                .Case<FloatType, ComplexType>(
                    [](auto type) { return enzyme::Activity::enzyme_active; })
                .Case<LLVM::LLVMPointerType, MemRefType>([&](auto type) {
                  // Skip the shadow
                  argIdx++;
                  return enzyme::Activity::enzyme_dup;
                })
                .Default(
                    [](Type type) { return enzyme::Activity::enzyme_const; });
      }
      argIdx++;
    }

    for (const auto &[resIdx, resType] :
         llvm::enumerate(callee.getResultTypes())) {
      resultActivities[resIdx] =
          llvm::TypeSwitch<Type, enzyme::Activity>(resType)
              .Case<FloatType, ComplexType>(
                  [](auto type) { return enzyme::Activity::enzyme_active; })
              .Default(
                  [](Type type) { return enzyme::Activity::enzyme_const; });
    }
  }

  void runActivityAnalysis(const enzyme::ActivityPrinterConfig &config,
                           FunctionOpInterface callee,
                           ArrayRef<enzyme::Activity> argActivities,
                           ArrayRef<enzyme::Activity> resultActivities,
                           bool print = true) {
    if (config.relative) {
      enzyme::runActivityAnnotations(callee, argActivities, config);
    } else if (config.dataflow) {
      enzyme::runDataFlowActivityAnalysis(callee, argActivities,
                                          /*print=*/true, verbose, annotate);
    } else {

      SmallPtrSet<Block *, 4> blocksNotForAnalysis;

      mlir::enzyme::MTypeResults TR; // TODO
      SmallPtrSet<mlir::Value, 1> constant_values;
      SmallPtrSet<mlir::Value, 1> activevals_;
      for (auto &&[arg, act] :
           llvm::zip(callee.getFunctionBody().getArguments(), argActivities)) {
        if (act == enzyme::Activity::enzyme_const)
          constant_values.insert(arg);
        else
          activevals_.insert(arg);
      }
      SmallVector<DIFFE_TYPE> ReturnActivity;
      for (auto act : resultActivities) {
        if (act != enzyme::Activity::enzyme_const)
          ReturnActivity.push_back(DIFFE_TYPE::DUP_ARG);
        else
          ReturnActivity.push_back(DIFFE_TYPE::CONSTANT);
      }

      enzyme::ActivityAnalyzer activityAnalyzer(
          blocksNotForAnalysis, constant_values, activevals_, ReturnActivity);

      callee.walk([&](Operation *op) {

      });
      MLIRContext *ctx = callee.getContext();
      callee.walk([&](Operation *op) {
        if (print)
          llvm::outs() << " Operation: " << *op << "\n";
        for (auto &reg : op->getRegions()) {
          for (auto &blk : reg.getBlocks()) {
            for (auto &arg : blk.getArguments()) {
              bool icv = activityAnalyzer.isConstantValue(TR, arg);
              if (annotate)
                op->setAttr("enzyme.arg_icv" +
                                std::to_string(arg.getArgNumber()),
                            BoolAttr::get(ctx, icv));
              if (print)
                llvm::outs() << " arg: " << arg << " icv=" << icv << "\n";
            }
          }
        }

        bool ici = activityAnalyzer.isConstantOperation(TR, op);
        if (annotate)
          op->setAttr("enzyme.ici", BoolAttr::get(ctx, ici));
        if (print)
          llvm::outs() << " op ici=" << ici << "\n";

        for (auto res : op->getResults()) {
          bool icv = activityAnalyzer.isConstantValue(TR, res);
          if (annotate)
            op->setAttr("enzyme.res_icv" +
                            std::to_string(res.getResultNumber()),
                        BoolAttr::get(ctx, icv));
          if (print)
            llvm::outs() << " res: " << res << " icv=" << icv << "\n";
        }
      });
    }
  }

  void runOnOperation() override {
    enzyme::ActivityPrinterConfig config;
    config.dataflow = dataflow;
    config.relative = relative;
    config.annotate = annotate;
    config.inferFromAutodiff = inferFromAutodiff;
    config.verbose = verbose;

    auto moduleOp = cast<ModuleOp>(getOperation());

    if (inferFromAutodiff) {
      // Infer the activity attributes from the __enzyme_autodiff call
      Operation *autodiff_decl = moduleOp.lookupSymbol("__enzyme_autodiff");
      if (!autodiff_decl) {
        for (auto &subOp : *moduleOp.getBody()) {
          if (auto func = dyn_cast<FunctionOpInterface>(&subOp)) {
            if (func.getName().contains("__enzyme_autodiff")) {
              autodiff_decl = &subOp;
              break;
            }
          }
        }
      }
      if (!autodiff_decl) {
        moduleOp.emitError("Failed to find __enzyme_autodiff symbol");
        return signalPassFailure();
      }
      auto uses = SymbolTable::getSymbolUses(autodiff_decl, moduleOp);
      assert(uses && "failed to find symbol uses of autodiff decl");

      for (SymbolTable::SymbolUse use : *uses) {
        auto autodiff_call = cast<CallOpInterface>(use.getUser());
        FlatSymbolRefAttr calleeAttr =
            cast<LLVM::AddressOfOp>(
                autodiff_call.getArgOperands().front().getDefiningOp())
                .getGlobalNameAttr();
        auto callee =
            cast<FunctionOpInterface>(moduleOp.lookupSymbol(calleeAttr));

        SmallVector<enzyme::Activity> argActivities{callee.getNumArguments()},
            resultActivities{callee.getNumResults()};
        // Populate the argument activities based on either the type or the
        // supplied annotation. First argument is the callee
        inferArgActivitiesFromEnzymeAutodiff(callee, autodiff_call,
                                             argActivities, resultActivities);
        runActivityAnalysis(config, callee, argActivities, resultActivities);
      }
      return;
    }

    if (funcsToAnalyze.empty()) {
      moduleOp.walk([this, config](FunctionOpInterface callee) {
        if (callee.isExternal() || callee.isPrivate())
          return;

        SmallVector<enzyme::Activity> argActivities{callee.getNumArguments()},
            resultActivities{callee.getNumResults()};
        initializeArgAndResActivities(callee, argActivities, resultActivities);

        runActivityAnalysis(config, callee, argActivities, resultActivities);
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

      runActivityAnalysis(config, callee, argActivities, resultActivities);
    }
  }
};
} // namespace
