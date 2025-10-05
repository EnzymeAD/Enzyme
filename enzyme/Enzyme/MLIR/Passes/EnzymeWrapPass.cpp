//===- EnzymeWrapPass.cpp - Replace calls with their derivatives ------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to create wrapper functions which differentiate
// ops.
//===----------------------------------------------------------------------===//

#include "Dialect/Ops.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "PassDetails.h"
#include "Passes/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "enzyme"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_DIFFERENTIATEWRAPPERPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

std::vector<DIFFE_TYPE> parseActivityString(StringRef inp) {
  if (inp.size() == 0)
    return {};
  std::vector<DIFFE_TYPE> ArgActivity;
  SmallVector<StringRef, 1> split;
  StringRef(inp.data(), inp.size()).split(split, ',');
  for (auto &str : split) {
    if (str == "enzyme_dup")
      ArgActivity.push_back(DIFFE_TYPE::DUP_ARG);
    else if (str == "enzyme_const")
      ArgActivity.push_back(DIFFE_TYPE::CONSTANT);
    else if (str == "enzyme_dupnoneed")
      ArgActivity.push_back(DIFFE_TYPE::DUP_NONEED);
    else if (str == "enzyme_active")
      ArgActivity.push_back(DIFFE_TYPE::OUT_DIFF);
    else {
      llvm::errs() << "unknown activity to parse, found: '" << str << "'\n";
      assert(0 && " unknown constant");
    }
  }
  return ArgActivity;
}

namespace {
struct DifferentiateWrapperPass
    : public enzyme::impl::DifferentiateWrapperPassBase<
          DifferentiateWrapperPass> {
  using DifferentiateWrapperPassBase::DifferentiateWrapperPassBase;

  void runOnOperation() override {
    MEnzymeLogic Logic;
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(getOperation());

    Operation *symbolOp = nullptr;
    if (infn != "")
      symbolOp = symbolTable.lookupSymbolIn<Operation *>(
          getOperation(), StringAttr::get(getOperation()->getContext(), infn));
    else {
      for (auto &op : getOperation()->getRegion(0).front()) {
        auto fn = dyn_cast<FunctionOpInterface>(symbolOp);
        if (!fn)
          continue;
        assert(symbolOp == nullptr);
        symbolOp = &op;
      }
    }
    if (!symbolOp) {
      llvm::errs() << " Could not find function '" << infn << "' to differentiate\n";
      signalPassFailure();
      return;
    }
    auto fn = cast<FunctionOpInterface>(symbolOp);
    bool omp = false;
    std::string postpasses = "";
    bool verifyPostPasses = true;
    bool strongZero = false;

    std::vector<DIFFE_TYPE> ArgActivity =
        parseActivityString(argTys.getValue());

    if (ArgActivity.size() != fn.getFunctionBody().front().getNumArguments()) {
      fn->emitError()
          << "Incorrect number of arg activity states for function, found "
          << ArgActivity.size() << " expected "
          << fn.getFunctionBody().front().getNumArguments();
      return;
    }

    std::vector<DIFFE_TYPE> RetActivity =
        parseActivityString(retTys.getValue());
    if (RetActivity.size() !=
        cast<FunctionType>(fn.getFunctionType()).getNumResults()) {
      fn->emitError()
          << "Incorrect number of ret activity states for function, found "
          << RetActivity.size() << " expected "
          << cast<FunctionType>(fn.getFunctionType()).getNumResults();
      return;
    }
    std::vector<bool> returnPrimal;
    std::vector<bool> returnShadow;
    for (auto act : RetActivity) {
      returnPrimal.push_back(act == DIFFE_TYPE::DUP_ARG);
      returnShadow.push_back(false);
    }

    MTypeAnalysis TA;
    auto type_args = TA.getAnalyzedTypeInfo(fn);

    bool freeMemory = true;
    size_t width = 1;

    std::vector<bool> volatile_args;
    for (auto &a : fn.getFunctionBody().getArguments()) {
      (void)a;
      volatile_args.push_back(!(mode == DerivativeMode::ReverseModeCombined));
    }

    FunctionOpInterface newFunc;
    if (mode == DerivativeMode::ForwardMode) {
      newFunc = Logic.CreateForwardDiff(
          fn, RetActivity, ArgActivity, TA, returnPrimal, mode, freeMemory,
          width,
          /*addedType*/ nullptr, type_args, volatile_args,
          /*augmented*/ nullptr, omp, postpasses, verifyPostPasses, strongZero);
    } else {
      newFunc = Logic.CreateReverseDiff(
          fn, RetActivity, ArgActivity, TA, returnPrimal, returnShadow, mode,
          freeMemory, width,
          /*addedType*/ nullptr, type_args, volatile_args,
          /*augmented*/ nullptr, omp, postpasses, verifyPostPasses, strongZero);
    }
    if (!newFunc) {
      signalPassFailure();
      return;
    }
    if (outfn == "") {
      fn->erase();
      SymbolTable::setSymbolVisibility(newFunc,
                                       SymbolTable::Visibility::Public);
      SymbolTable::setSymbolName(cast<FunctionOpInterface>(newFunc),
                                 (std::string)infn);
    } else {
      SymbolTable::setSymbolName(cast<FunctionOpInterface>(newFunc),
                                 (std::string)outfn);
    }
  }
};

} // end anonymous namespace
