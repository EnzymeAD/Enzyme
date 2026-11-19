//===- GradientUtilsReverse.cpp - Utilities for gradient interfaces
//--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interfaces/GradientUtilsReverse.h"
#include "Dialect/Dialect.h"
#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffControlFlowInterface.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// TODO: this shouldn't depend on specific dialects except Enzyme.
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "CloneFunction.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/BreadthFirstIterator.h"

using namespace mlir;
using namespace mlir::enzyme;

mlir::enzyme::MGradientUtilsReverse::MGradientUtilsReverse(
    MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
    FunctionOpInterface oldFunc_, MTypeAnalysis &TA_,
    IRMapping invertedPointers_, const llvm::ArrayRef<bool> returnPrimals,
    const llvm::ArrayRef<bool> returnShadows,
    const SmallPtrSetImpl<mlir::Value> &constantvalues_,
    const SmallPtrSetImpl<mlir::Value> &activevals_,
    ArrayRef<DIFFE_TYPE> ReturnActivity, ArrayRef<DIFFE_TYPE> ArgDiffeTypes_,
    IRMapping &originalToNewFn_,
    std::map<Operation *, Operation *> &originalToNewFnOps_,
    DerivativeMode mode_, unsigned width, bool omp, StringRef postpasses,
    bool verifyPostPasses, bool strongZero)
    : MDiffeGradientUtils(Logic, newFunc_, oldFunc_, TA_, /*MTypeResults*/ {},
                          invertedPointers_, returnPrimals, returnShadows,
                          constantvalues_, activevals_, ReturnActivity,
                          ArgDiffeTypes_, originalToNewFn_, originalToNewFnOps_,
                          mode_, width, omp, postpasses, verifyPostPasses,
                          strongZero) {}

Type mlir::enzyme::MGradientUtilsReverse::getIndexCacheType() {
  Type indexType = getIndexType();
  return getCacheType(indexType);
}

Type mlir::enzyme::MGradientUtilsReverse::getIndexType() {
  return getAutoDiffControlFlow(newFunc.getOperation())
      .getBranchIndexType(newFunc.getContext());
}

Value mlir::enzyme::MGradientUtilsReverse::getDeferredFrees(Type elemType,
                                                            StringRef dealloc) {
  for (auto init : initializationBlock->getOps<enzyme::InitOp>())
    if (auto attr = init->getAttrOfType<StringAttr>(deferredFreeAttrName);
        attr && attr.getValue() == dealloc)
      return init;
  Value cache = insertInit(getCacheType(elemType));
  cache.getDefiningOp()->setAttr(deferredFreeAttrName,
                                 StringAttr::get(cache.getContext(), dealloc));
  return cache;
}

Value mlir::enzyme::MGradientUtilsReverse::insertInit(Type t) {
  OpBuilder builder(initializationBlock, initializationBlock->begin());
  // The block is where the init goes, not where a location comes from: asking
  // its last operation for one reads off the end of an empty block, and the
  // first cache anything asks for is the one that finds it empty.
  Location loc = initializationBlock->empty()
                     ? builder.getUnknownLoc()
                     : initializationBlock->rbegin()->getLoc();
  return enzyme::InitOp::create(builder, loc, t);
}

// Cache
Type mlir::enzyme::MGradientUtilsReverse::getCacheType(Type t) {
  // Likewise the context, which is the type's own and does not need looking up
  // in the block.
  Type cacheType = CacheType::get(t.getContext(), t);
  return cacheType;
}

void MGradientUtilsReverse::registerCacheCreatorHook(
    std::function<std::pair<Value, Value>(Type)> hook) {
  if (hook != nullptr) {
    cacheCreatorHook.push_back(hook);
  }
}

void MGradientUtilsReverse::deregisterCacheCreatorHook(
    std::function<std::pair<Value, Value>(Type)> hook) {
  if (hook != nullptr) {
    cacheCreatorHook.pop_back();
  }
}

std::pair<Value, Value> MGradientUtilsReverse::getNewCache(Type t) {
  if (cacheCreatorHook.empty()) {
    Value cache = insertInit(t);
    return {cache, cache};
  }
  return cacheCreatorHook.back()(t);
}

// We assume that caches will only be written to at one location. The returned
// cache is (might be) "pop only"
Value MGradientUtilsReverse::initAndPushCache(Value v, OpBuilder &builder) {
  if (Logic.optimizeCaches && cacheCreatorHook.empty()) {
    unsigned newCaches = 0;
    FailureOr<ReversePlan> plan = planReverseValue(v, 0, newCaches);
    if (succeeded(plan))
      return commitReversePlan(*plan, v.getType());
  }
  auto ct = getCacheType(v.getType());
  auto [pushCache, popCache] = getNewCache(ct);
  enzyme::PushOp::create(builder, v.getLoc(), pushCache, v);
  return popCache;
}

Value MGradientUtilsReverse::popCache(Value cache, OpBuilder &builder) {
  if (reverseValues.count(cache))
    return materializeReverseValue(cache, builder);
  return enzyme::PopOp::create(
      builder, cache.getLoc(),
      cast<enzyme::CacheType>(cache.getType()).getType(), cache);
}

// Find the inverse mapping of originalToNewFn
Value MGradientUtilsReverse::originalOf(Value v) {
  const auto &map = originalToNewFn.getValueMap();
  auto valid = [&](Value orig) {
    return orig && originalToNewFn.lookupOrNull(orig) == v;
  };
  Value orig = newToOriginal.lookup(v);
  if (map.size() != newToOriginalSize || (orig && !valid(orig))) {
    newToOriginal.clear();
    for (auto [o, now] : map)
      newToOriginal[now] = o;
    newToOriginalSize = map.size();
    orig = newToOriginal.lookup(v);
  }
  return valid(orig) ? orig : Value();
}

// The original value `v` is the shadow of, if it is one.
Value MGradientUtilsReverse::shadowedOriginalOf(Value v) {
  for (auto [orig, shadow] : invertedPointers.getValueMap())
    if (shadow == v)
      return orig;
  return Value();
}

// Check whether block runs at most once.
bool MGradientUtilsReverse::isOnceBeforeReverse(Block *origBlock) {
  if (!onceBeforeReverse) {
    onceBeforeReverse.emplace();
    Region &body = oldFunc.getFunctionBody();
    DominanceInfo dom(oldFunc);
    SmallVector<Block *> returns;
    for (Block &block : body)
      if (isFunctionReturnLike(block.getTerminator()))
        returns.push_back(&block);
    for (Block &block : body) {
      // On a cycle: it reaches itself.
      SmallVector<Block *> todo(block.getSuccessors());
      DenseSet<Block *> seen;
      bool cyclic = false;
      while (!todo.empty() && !cyclic) {
        Block *b = todo.pop_back_val();
        cyclic = b == &block;
        if (seen.insert(b).second)
          llvm::append_range(todo, b->getSuccessors());
      }
      if (!cyclic && llvm::all_of(returns, [&](Block *ret) {
            return dom.dominates(&block, ret);
          }))
        onceBeforeReverse->insert(&block);
    }
  }
  return onceBeforeReverse->contains(origBlock);
}

FailureOr<MGradientUtilsReverse::ReversePlan>
MGradientUtilsReverse::planReverseValue(Value v, unsigned depth,
                                        unsigned &newCaches) {
  // The derivative's own arguments.
  // enzyme.pop %7.
  if (auto arg = dyn_cast<BlockArgument>(v);
      arg && arg.getOwner() == &newFunc.getFunctionBody().front())
    return ReversePlan{ReverseValue::Available, Value(), false, v, nullptr, {}};

  // The original is showdow or not.
  bool shadow = false;
  Value orig = originalOf(v);
  if (!orig) {
    orig = shadowedOriginalOf(v);
    shadow = static_cast<bool>(orig);
  }
  if (!orig || orig.getParentRegion() != &oldFunc.getFunctionBody())
    return failure();
  return planOriginal(orig, shadow, depth, newCaches,
                      loopDepthOf(orig.getParentBlock()));
}

// How many loops of the original function a block is in, by its natural
// loops: a back edge's target dominates its source, and the loop is what
// reaches the source without passing through the target.
unsigned MGradientUtilsReverse::loopDepthOf(Block *origBlock) {
  if (!loopDepths) {
    loopDepths.emplace();
    Region &body = oldFunc.getFunctionBody();
    DominanceInfo dom(oldFunc);
    DenseMap<Block *, DenseSet<Block *>> loops;
    for (Block &source : body)
      for (Block *header : source.getSuccessors()) {
        if (!dom.dominates(header, &source))
          continue;
        DenseSet<Block *> &loop = loops[header];
        loop.insert(header);
        SmallVector<Block *> todo{&source};
        while (!todo.empty()) {
          Block *b = todo.pop_back_val();
          if (!loop.insert(b).second)
            continue;
          for (Block *pred : b->getPredecessors())
            todo.push_back(pred);
        }
      }
    for (auto &[header, loop] : loops)
      for (Block *b : loop)
        ++(*loopDepths)[b];
  }
  return loopDepths->lookup(origBlock);
}

// How the reverse sweep gets `orig` back, or its shadow.
FailureOr<MGradientUtilsReverse::ReversePlan>
MGradientUtilsReverse::planOriginal(Value orig, bool shadow, unsigned depth,
                                    unsigned &newCaches, unsigned loopDepth) {
  if (orig.getParentRegion() != &oldFunc.getFunctionBody())
    return failure();
  if (isOnceBeforeReverse(orig.getParentBlock()))
    return ReversePlan{
        ReverseValue::Available, orig, shadow, Value(), nullptr, {}};

  // A pure op, computed again from its operands, when that takes no more
  // than one more value cached, as often as it would be, than caching it
  // would: one defined in an outer loop is cached less often.
  Operation *def = orig.getDefiningOp();
  auto iface = dyn_cast<AutoDiffTypeInterface>(orig.getType());
  bool cheap = shadow || (iface && iface.isMutable()) || isConstantValue(orig);
  auto view = dyn_cast_if_present<ViewLikeOpInterface>(def);
  Operation *primal = def ? getNewFromOriginal(orig).getDefiningOp() : nullptr;
  if (cheap && primal && depth < 3 && def->getNumResults() == 1 &&
      def->getNumRegions() == 0 && def->getNumOperands() <= 3 &&
      primal->getNumOperands() == def->getNumOperands() &&
      (isMemoryEffectFree(def) || view) && (!shadow || view)) {
    ReversePlan plan{
        ReverseValue::Recompute, Value(), false, Value(), primal, {}};
    unsigned operandCaches = 0;
    bool ok = true;
    for (OpOperand &operand : def->getOpOperands()) {
      bool sourceShadow = shadow && operand.get() == view.getViewSource();
      FailureOr<ReversePlan> sub = planOriginal(
          operand.get(), sourceShadow, depth + 1, operandCaches, loopDepth);
      if (failed(sub)) {
        ok = false;
        break;
      }
      plan.operands.push_back(std::move(*sub));
    }
    if (ok && operandCaches <= 1) {
      newCaches += operandCaches;
      return plan;
    }
  }

  if (!sharedCaches.count({orig, unsigned(shadow)}) &&
      loopDepthOf(orig.getParentBlock()) >= loopDepth)
    ++newCaches;
  return ReversePlan{ReverseValue::Shared, orig, shadow, Value(), nullptr, {}};
}

Value MGradientUtilsReverse::commitReversePlan(const ReversePlan &plan,
                                               Type type) {
  if (plan.kind == ReverseValue::Shared) {
    if (Value cache = sharedCaches.lookup({plan.orig, unsigned(plan.shadow)}))
      return cache;
    // Pushed where the original value is defined: after its op, before
    // which a shadow is made, or at the start of its block.
    Value primal = getNewFromOriginal(plan.orig);
    Value now = plan.shadow ? invertedPointers.lookupOrNull(plan.orig) : primal;
    Value cache = insertInit(getCacheType(now.getType()));
    OpBuilder builder(now.getContext());
    if (auto arg = dyn_cast<BlockArgument>(primal))
      builder.setInsertionPointToStart(arg.getOwner());
    else
      builder.setInsertionPointAfter(primal.getDefiningOp());
    enzyme::PushOp::create(builder, now.getLoc(), cache, now);
    sharedCaches[{plan.orig, unsigned(plan.shadow)}] = cache;
    reverseValues[cache] = {
        ReverseValue::Shared, plan.orig, plan.shadow, Value(), nullptr, {}};
    return cache;
  }
  Value handle = insertInit(getCacheType(type));
  ReverseValue rv{plan.kind, plan.orig, plan.shadow, plan.fwd, plan.def, {}};
  for (auto [operand, sub] : llvm::zip(
           plan.def ? plan.def->getOperands() : ValueRange(), plan.operands))
    rv.operands.push_back(commitReversePlan(sub, operand.getType()));
  reverseValues[handle] = rv;
  return handle;
}

Value MGradientUtilsReverse::materializeReverseValue(Value handle,
                                                     OpBuilder &builder) {
  const ReverseValue &rv = reverseValues.find(handle)->second;
  switch (rv.kind) {
  case ReverseValue::Available:
    if (!rv.orig)
      return rv.fwd;
    return rv.shadow ? invertedPointers.lookupOrNull(rv.orig)
                     : getNewFromOriginal(rv.orig);
  case ReverseValue::Shared:
    return enzyme::GetOp::create(
        builder, handle.getLoc(),
        cast<enzyme::CacheType>(handle.getType()).getType(), handle);
  case ReverseValue::Recompute: {
    SmallVector<Value> operands;
    for (Value sub : rv.operands)
      operands.push_back(materializeReverseValue(sub, builder));
    Operation *clone = builder.clone(*rv.def);
    for (auto [i, operand] : llvm::enumerate(operands))
      clone->setOperand(i, operand);
    return clone->getResult(0);
  }
  }
  llvm_unreachable("unknown reverse value");
}

void MGradientUtilsReverse::recordReverseDefinitions(ValueRange origValues,
                                                     Block *reverseBB) {
  if (!Logic.optimizeCaches)
    return;
  Operation *last = reverseBB->empty() ? nullptr : &reverseBB->back();
  for (Value orig : origValues)
    reverseDefinitions[orig] = {reverseBB, last};
}

void MGradientUtilsReverse::finishReverseValues() {
  // Placeholder is removed, so the definition can go after the use
  for (auto [key, cache] : sharedCaches)
    for (Operation *user : cache.getUsers())
      if (auto push = dyn_cast<enzyme::PushOp>(user))
        if (Operation *def = push.getValue().getDefiningOp();
            def && def->getBlock() == push->getBlock() &&
            push->isBeforeInBlock(def))
          push->moveAfter(def);
  // Replace the dominating value of get as pop
  for (auto [key, cache] : sharedCaches) {
    auto [block, after] = reverseDefinitions.lookup(key.first);
    assert(block && "a shared cache's value was never reached in reverse");
    SmallVector<enzyme::GetOp> reads;
    for (Operation *user : cache.getUsers())
      if (auto get = dyn_cast<enzyme::GetOp>(user))
        reads.push_back(get);
    if (reads.size() == 1 && reads.front()->getBlock() == block &&
        (!after || reads.front()->isBeforeInBlock(after) ||
         reads.front() == after)) {
      OpBuilder builder(reads.front());
      reads.front().replaceAllUsesWith(
          enzyme::PopOp::create(
              builder, cache.getLoc(),
              cast<enzyme::CacheType>(cache.getType()).getType(), cache)
              .getResult());
      reads.front().erase();
      continue;
    }
    OpBuilder builder(cache.getContext());
    if (after)
      builder.setInsertionPointAfter(after);
    else
      builder.setInsertionPointToStart(block);
    enzyme::PopOp::create(builder, cache.getLoc(),
                          cast<enzyme::CacheType>(cache.getType()).getType(),
                          cache);
  }
  for (auto &[handle, rv] : reverseValues)
    if (rv.kind != ReverseValue::Shared && handle.use_empty())
      handle.getDefiningOp()->erase();
  reverseValues.clear();
  sharedCaches.clear();
  reverseDefinitions.clear();
}

Operation *
mlir::enzyme::MGradientUtilsReverse::cloneWithNewOperands(OpBuilder &B,
                                                          Operation *op) {
  IRMapping map;
  for (auto operand : op->getOperands())
    map.map(operand, getNewFromOriginal(operand));
  return B.clone(*op, map);
}

void mlir::enzyme::MGradientUtilsReverse::addToDiffe(Value oldGradient,
                                                     Value addedGradient,
                                                     OpBuilder &builder) {
  assert(!isConstantValue(oldGradient));
  Value operandGradient = diffe(oldGradient, builder);
  auto iface = cast<AutoDiffTypeInterface>(addedGradient.getType());
  auto added = iface.createAddOp(builder, oldGradient.getLoc(), operandGradient,
                                 addedGradient);
  setDiffe(oldGradient, added, builder);
}

void MGradientUtilsReverse::createReverseModeBlocks(Region &oldFunc,
                                                    Region &newFunc) {
  for (auto it = oldFunc.getBlocks().rbegin(); it != oldFunc.getBlocks().rend();
       ++it) {
    Block *block = &*it;
    Block *reverseBlock = new Block();
    newFunc.getBlocks().insert(newFunc.end(), reverseBlock);
    mapReverseModeBlocks.map(block, reverseBlock);
  }
}

MGradientUtilsReverse *MGradientUtilsReverse::CreateFromClone(
    MEnzymeLogic &Logic, DerivativeMode mode_, unsigned width,
    FunctionOpInterface todiff, MTypeAnalysis &TA, MFnTypeInfo &oldTypeInfo,
    const ArrayRef<bool> returnPrimals, const ArrayRef<bool> returnShadows,
    ArrayRef<DIFFE_TYPE> retType, ArrayRef<DIFFE_TYPE> constant_args,
    mlir::Type additionalArg, bool omp, llvm::StringRef postpasses,
    bool verifyPostPasses, bool strongZero) {
  std::string prefix;

  switch (mode_) {
  case DerivativeMode::ForwardMode:
  case DerivativeMode::ForwardModeSplit:
  case DerivativeMode::ForwardModeError:
    assert(false);
    break;
  case DerivativeMode::ReverseModeCombined:
  case DerivativeMode::ReverseModeGradient:
    prefix = "diffe";
    break;
  case DerivativeMode::ReverseModePrimal:
    llvm_unreachable("invalid DerivativeMode: ReverseModePrimal\n");
  }

  if (width > 1)
    prefix += std::to_string(width);

  IRMapping originalToNew;
  std::map<Operation *, Operation *> originalToNewOps;

  SmallPtrSet<mlir::Value, 1> returnvals;
  SmallPtrSet<mlir::Value, 1> constant_values;
  SmallPtrSet<mlir::Value, 1> nonconstant_values;
  IRMapping invertedPointers;
  FunctionOpInterface newFunc = CloneFunctionWithReturns(
      mode_, width, todiff, invertedPointers, constant_args, constant_values,
      nonconstant_values, returnvals, returnPrimals, returnShadows, retType,
      prefix + todiff.getName(), originalToNew, originalToNewOps,
      additionalArg);

  return new MGradientUtilsReverse(
      Logic, newFunc, todiff, TA, invertedPointers, returnPrimals,
      returnShadows, constant_values, nonconstant_values, retType,
      constant_args, originalToNew, originalToNewOps, mode_, width, omp,
      postpasses, verifyPostPasses, strongZero);
}
