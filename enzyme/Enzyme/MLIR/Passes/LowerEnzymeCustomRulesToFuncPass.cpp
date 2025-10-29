//===- LowerEnzymeCustomRulesToFuncPass.cpp - ------------------------------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Dialect/Ops.h"
#include "Passes/Passes.h"
#include "Passes/RemovalUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMECUSTOMRULESTOFUNCPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {
struct LowerEnzymeCustomRulesToFuncPass
    : public enzyme::impl::LowerEnzymeCustomRulesToFuncPassBase<
          LowerEnzymeCustomRulesToFuncPass> {
  using LowerEnzymeCustomRulesToFuncPassBase::
      LowerEnzymeCustomRulesToFuncPassBase;

  void runOnOperation() override;
};
} // end anonymous namespace

static LogicalResult
lowerCustomReverseRuleToFunc(enzyme::CustomReverseRuleOp revRule) {
  SymbolTable symbolTable(SymbolTable::getNearestSymbolTable(revRule));

  Block *bodyDef = &revRule.getBody().front();

  enzyme::AugmentedPrimalOp primal = nullptr;
  enzyme::ReverseOp reverse = nullptr;

  for (Operation &op : *bodyDef) {
    if (auto AP = dyn_cast<enzyme::AugmentedPrimalOp>(op)) {
      if (primal) {
        AP->emitError() << "multiple augmented primal ops in a custom rule";
        return failure();
      }
      primal = AP;
    } else if (auto RO = dyn_cast<enzyme::ReverseOp>(op)) {
      if (reverse) {
        RO->emitError() << "multiple reverse op in a custom rule";
        return failure();
      }
      reverse = RO;
    }
  }

  bool singleBlock =
      primal.getBody().hasOneBlock() && reverse.getBody().hasOneBlock();
  if (!singleBlock) {
    // TODO: caching with non-structured control flow;
    revRule->emitError() << "todo: lowering to func.func is not supported for "
                            "custom rules with more than one block.";
    return failure();
  }

  auto funcType = revRule.getFunctionType();

  SmallVector<mlir::Type> primalArgTypes(funcType.getInputs().begin(),
                                         funcType.getInputs().end());
  SmallVector<mlir::Type> primalResultTypes(funcType.getResults().begin(),
                                            funcType.getResults().end());

  SmallVector<mlir::Type> reverseArgTypes;
  for (auto [retTy, act] :
       llvm::zip_equal(funcType.getResults(), revRule.getRetActivity())) {

    auto iattr = cast<mlir::enzyme::ActivityAttr>(act);
    switch (iattr.getValue()) {
    case mlir::enzyme::Activity::enzyme_active:
    case mlir::enzyme::Activity::enzyme_activenoneed:
      reverseArgTypes.push_back(retTy);
      break;
    case mlir::enzyme::Activity::enzyme_const:
      break;
    default:
      llvm_unreachable("todo");
    }
  }

  SmallVector<mlir::Type> reverseResultTypes;
  for (auto [argTy, act] :
       llvm::zip_equal(funcType.getInputs(), revRule.getActivity())) {

    auto iattr = cast<mlir::enzyme::ActivityAttr>(act);
    switch (iattr.getValue()) {
    case mlir::enzyme::Activity::enzyme_active:
    case mlir::enzyme::Activity::enzyme_activenoneed:
      reverseResultTypes.push_back(argTy);
      break;
    case mlir::enzyme::Activity::enzyme_const:
      break;
    default:
      llvm_unreachable("todo");
    }
  }

  SmallVector<CacheInfo> caches;
  SmallVector<mlir::Type> cacheTypes;
  for (Operation &op : *bodyDef) {
    if (auto init = dyn_cast<enzyme::InitOp>(&op)) {
      auto CT = dyn_cast<enzyme::CacheType>(init.getType());
      if (!CT)
        continue;

      CacheInfo info(init.getResult());
      if (info.pushOp->getBlock() != &primal.getBody().front()) {
        info.pushOp->emitError()
            << "push operation not hoisted to the top level.";
        return failure();
      }

      if (info.popOp->getBlock() != &reverse.getBody().front()) {
        info.popOp->emitError()
            << "pop operation not hoisted to the top level.";
        return failure();
      }

      auto ET = CT.getType();
      cacheTypes.push_back(ET);
      caches.push_back(info);
    }
  }

  primalResultTypes.append(cacheTypes.begin(), cacheTypes.end());
  reverseArgTypes.append(cacheTypes.begin(), cacheTypes.end());

  auto revRuleName = revRule.getName();

  FunctionType primalFuncType = FunctionType::get(
      revRule->getContext(), primalArgTypes, primalResultTypes);

  SmallVector<char> nameBuf;
  Twine primalName = revRuleName + "_primal";
  Twine reverseName = revRuleName + "_reverse";

  auto primalFunc =
      func::FuncOp::create(primal.getLoc(), primalName.toStringRef(nameBuf),
                           primalFuncType, ArrayRef<NamedAttribute>());

  nameBuf.clear();

  FunctionType reverseFuncType = FunctionType::get(
      revRule->getContext(), reverseArgTypes, reverseResultTypes);
  auto reverseFunc =
      func::FuncOp::create(reverse.getLoc(), reverseName.toStringRef(nameBuf),
                           reverseFuncType, ArrayRef<NamedAttribute>());

  primalFunc.getBody().takeBody(primal.getBody());
  for (Block &b : primalFunc.getBody()) {
    Operation *term = b.getTerminator();
    if (isa<enzyme::YieldOp>(term)) {
      OpBuilder builder(term);
      SmallVector<Value> toReturn(term->getOperands().begin(),
                                  term->getOperands().end());
      for (auto &info : caches) {
        toReturn.push_back(info.pushOp.getValue());
        info.pushOp->erase();
      }
      builder.create<func::ReturnOp>(term->getLoc(), toReturn);
      term->erase();
    }
  }

  reverseFunc.getBody().takeBody(reverse.getBody());
  SmallVector<Location> cacheLocs = llvm::map_to_vector(
      caches, [](CacheInfo info) { return info.initOp->getLoc(); });
  for (auto [info, arg] : llvm::zip_equal(
           caches,
           reverseFunc.getBody().front().addArguments(cacheTypes, cacheLocs))) {
    info.popOp.getResult().replaceAllUsesWith(arg);
    info.popOp->erase();
    info.initOp->erase();
  }
  for (Block &b : reverseFunc.getBody()) {
    Operation *term = b.getTerminator();
    if (isa<enzyme::YieldOp>(term)) {
      OpBuilder builder(term);
      builder.create<func::ReturnOp>(term->getLoc(), term->getOperands());
      term->erase();
    }
  }

  symbolTable.insert(primalFunc);
  SymbolTable::setSymbolVisibility(primalFunc,
                                   SymbolTable::Visibility::Private);

  symbolTable.insert(reverseFunc);
  SymbolTable::setSymbolVisibility(reverseFunc,
                                   SymbolTable::Visibility::Private);

  auto uses = SymbolTable::getSymbolUses(
      StringAttr::get(revRule->getContext(), revRuleName), symbolTable.getOp());
  if (!uses) {
    revRule->erase();
    return success();
  }

  SmallVector<Value> tapes;

  SetVector<Operation *> toDelete;

  for (auto use : *uses) {
    Operation *user = use.getUser();
    auto CAP = dyn_cast<enzyme::CallAugmentedPrimalOp>(user);
    if (!CAP)
      continue;

    OpBuilder builder(CAP);
    auto primalCall = builder.create<func::CallOp>(CAP.getLoc(), primalFunc,
                                                   CAP->getOperands());

    auto tape = CAP->getResult(CAP->getNumResults() - 1);
    for (auto tapeUser : tape.getUsers()) {
      if (auto CCR = dyn_cast<enzyme::CallCustomReverseOp>(tapeUser)) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(CCR);
        SmallVector<Value> operands(
            CCR->getOperands().slice(0, CCR->getNumOperands() - 1).begin(),
            CCR->getOperands().slice(0, CCR->getNumOperands() - 1).end());
        operands.append(
            primalCall.getResults()
                .slice(revRule.getFunctionType().getNumResults(), caches.size())
                .begin(),
            primalCall.getResults()
                .slice(revRule.getFunctionType().getNumResults(), caches.size())
                .end());
        auto reverseCall =
            builder.create<func::CallOp>(CCR.getLoc(), reverseFunc, operands);
        for (auto [oldRes, newRes] :
             llvm::zip(CCR.getResults(), reverseCall.getResults())) {
          oldRes.replaceAllUsesWith(newRes);
        }

        toDelete.insert(CAP);
        toDelete.insert(CCR);
      } else {
        tapeUser->emitError()
            << "todo: support tape going through this operation";
        return failure();
      }
    }
  }

  toDelete.insert(revRule);

  auto worklist = toDelete.takeVector();
  while (!worklist.empty()) {
    Operation *op = worklist.back();
    op->erase();
    worklist.pop_back();
  }

  return success();
}

void LowerEnzymeCustomRulesToFuncPass::runOnOperation() {
  bool failed = false;

  getOperation()->walk([&failed](enzyme::CustomReverseRuleOp revRule) {
    failed |= lowerCustomReverseRuleToFunc(revRule).failed();
  });

  if (failed) {
    signalPassFailure();
    return;
  }
}
