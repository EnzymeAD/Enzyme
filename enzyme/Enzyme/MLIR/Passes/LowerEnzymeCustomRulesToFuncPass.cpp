//===- LowerEnzymeCustomRulesToFuncPass.cpp - ------------------------------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a pass to lower enzyme custom rules to the func dialect.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Passes/Passes.h"
#include "Passes/RemovalUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

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

  if (!revRule.getBody().hasOneBlock())
    return revRule->emitError() << "a custom reverse rule needs one body block";

  Block *bodyDef = &revRule.getBody().front();

  enzyme::CustomReverseRuleAugmentedPrimalOp primal = nullptr;
  enzyme::CustomReverseRuleReverseOp reverse = nullptr;

  for (Operation &op : bodyDef->without_terminator()) {
    if (auto AP = dyn_cast<enzyme::CustomReverseRuleAugmentedPrimalOp>(op)) {
      if (primal) {
        AP->emitError() << "multiple augmented primal ops in a custom rule";
        return failure();
      }
      primal = AP;
    } else if (auto RO = dyn_cast<enzyme::CustomReverseRuleReverseOp>(op)) {
      if (reverse) {
        RO->emitError() << "multiple reverse op in a custom rule";
        return failure();
      }
      reverse = RO;
    }
  }

  if (!primal || !reverse)
    return revRule->emitError() << "a custom reverse rule needs one augmented "
                                   "primal and one reverse";

  bool singleBlock =
      primal.getBody().hasOneBlock() && reverse.getBody().hasOneBlock();
  if (!singleBlock) {
    // TODO: caching with non-structured control flow;
    return revRule->emitError()
           << "todo: lowering to func.func is not supported for "
              "custom rules with more than one block.";
  }

  auto funcType = revRule.getFunctionType();
  if (funcType.getNumInputs() != revRule.getActivity().size() ||
      funcType.getNumResults() != revRule.getRetActivity().size())
    return revRule->emitError()
           << "custom rule activities must match its function type";

  SmallVector<mlir::Type> primalArgTypes;
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
    case mlir::enzyme::Activity::enzyme_constnoneed:
      break;
    default:
      return revRule->emitError()
             << "unsupported custom rule return activity " << iattr.getValue();
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
      primalArgTypes.push_back(argTy);
      break;
    case mlir::enzyme::Activity::enzyme_const:
    case mlir::enzyme::Activity::enzyme_constnoneed:
      primalArgTypes.push_back(argTy);
      break;
    case mlir::enzyme::Activity::enzyme_dup:
    case mlir::enzyme::Activity::enzyme_dupnoneed:
      primalArgTypes.push_back(argTy);
      primalArgTypes.push_back(
          cast<AutoDiffTypeInterface>(argTy).getShadowType(/*width*/ 1));
      break;
    default:
      return revRule->emitError()
             << "unsupported custom rule argument activity "
             << iattr.getValue();
    }
  }

  // The caches were settled when the first call named the rule (or are
  // settled now, for a rule no differentiation has used yet): their order is
  // the order of the rule's top-level `enzyme.init` ops, which is the order of
  // the cache values every call site carries.
  if (failed(finalizeCustomReverseRule(revRule)))
    return failure();

  SmallVector<CacheInfo> caches;
  for (enzyme::InitOp init : getCustomReverseRuleCacheInits(revRule))
    caches.push_back(CacheInfo(init.getResult()));
  SmallVector<mlir::Type> cacheTypes = llvm::map_to_vector(
      caches, [](CacheInfo info) { return info.cachedType(); });

  SmallVector<Operation *> toCopyOnBoth;

  for (Operation &op :
       llvm::make_early_inc_range(bodyDef->without_terminator())) {
    if (isa<enzyme::InitOp, enzyme::CustomReverseRuleReverseOp,
            enzyme::CustomReverseRuleAugmentedPrimalOp>(op)) {
      // allowed
      continue;
    }

    if (auto pushOp = dyn_cast<PushOp>(&op)) {
      CacheInfo info(pushOp.getCache());

      if (info.initOp->getBlock() == bodyDef &&
          info.pushOp->getBlock() == bodyDef &&
          info.popOp->getBlock() == bodyDef) {
        info.popOp.getResult().replaceAllUsesWith(info.pushedValue());

        info.pushOp.erase();
        info.popOp.erase();
        info.initOp.erase();
      }

      continue;
    }

    toCopyOnBoth.push_back(&op);
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
      func::ReturnOp::create(builder, term->getLoc(), toReturn);
      term->erase();
    }
  }

  {
    IRMapping mapping;
    OpBuilder builder(&primalFunc.getBody().front(),
                      primalFunc.getBody().front().begin());
    for (Operation *op : toCopyOnBoth) {
      auto newOp = builder.clone(*op, mapping);
      for (auto [newRes, oldRes] :
           llvm::zip_equal(newOp->getResults(), op->getResults())) {
        oldRes.replaceUsesWithIf(newRes, [&](OpOperand &use) {
          return primalFunc->isProperAncestor(use.getOwner());
        });
      }
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
      func::ReturnOp::create(builder, term->getLoc(), term->getOperands());
      term->erase();
    }
  }

  {
    IRMapping mapping;
    OpBuilder builder(&reverseFunc.getBody().front(),
                      reverseFunc.getBody().front().begin());
    for (Operation *op : toCopyOnBoth) {
      auto newOp = builder.clone(*op, mapping);
      for (auto [newRes, oldRes] :
           llvm::zip_equal(newOp->getResults(), op->getResults())) {
        oldRes.replaceUsesWithIf(newRes, [&](OpOperand &use) {
          return reverseFunc->isProperAncestor(use.getOwner());
        });
      }
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

  for (auto use : *uses) {
    Operation *user = use.getUser();
    removeCustomReverseRule(user, "enzyme.custom_rule", revRuleName);
    removeCustomReverseRule(user, "enzyme.derived_rules", revRuleName);
  }

  // Every call carries the rule's cache values explicitly, so each lowers
  // one-to-one: the augmented primal returns (results..., caches...), the
  // reverse takes (cotangents..., caches...).
  SetVector<Operation *> toDelete;
  for (auto use : *uses) {
    Operation *user = use.getUser();
    if (auto CAP = dyn_cast<enzyme::CallAugmentedPrimalOp>(user)) {
      OpBuilder builder(CAP);
      auto primalCall = func::CallOp::create(builder, CAP.getLoc(), primalFunc,
                                             CAP.getInputs());
      CAP->replaceAllUsesWith(primalCall.getResults());
      toDelete.insert(CAP);
    } else if (auto CCR = dyn_cast<enzyme::CallCustomReverseOp>(user)) {
      OpBuilder builder(CCR);
      SmallVector<Value> operands(CCR.getInputs().begin(),
                                  CCR.getInputs().end());
      operands.append(CCR.getCaches().begin(), CCR.getCaches().end());
      auto reverseCall =
          func::CallOp::create(builder, CCR.getLoc(), reverseFunc, operands);
      CCR->replaceAllUsesWith(reverseCall.getResults());
      toDelete.insert(CCR);
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

LogicalResult mlir::enzyme::lowerCustomReverseRulesToFunc(Operation *root) {
  bool failed = false;
  root->walk([&failed](enzyme::CustomReverseRuleOp revRule) {
    failed |= lowerCustomReverseRuleToFunc(revRule).failed();
  });
  return success(!failed);
}

void LowerEnzymeCustomRulesToFuncPass::runOnOperation() {
  if (failed(lowerCustomReverseRulesToFunc(getOperation())))
    signalPassFailure();
}
