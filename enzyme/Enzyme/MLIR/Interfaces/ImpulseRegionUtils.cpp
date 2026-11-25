//===- ImpulseRegionUtils.cpp - Utilities for impulse region-form ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interfaces/ImpulseRegionUtils.h"
#include "Interfaces/ImpulseUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace mlir::impulse;

static void collectSitesInOrder(Region &region,
                                SmallVectorImpl<SampleRegionOp> &sites) {
  region.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto site = dyn_cast<SampleRegionOp>(op)) {
      sites.push_back(site);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
}

static bool holdsSites(Operation *op) {
  return op->walk([](SampleRegionOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

/// Whether `op` may move out of a branch arm that holds sites; see
/// constructLogpdfRegion.
static bool
isHoistableArmOp(Operation &op,
                 function_ref<bool(Operation *)> isHoistableFromBranch) {
  if (op.getNumRegions() != 0 || op.getNumResults() == 0 ||
      !isMemoryEffectFree(&op))
    return false;
  return isa<arith::ConstantOp, SliceOp, ReshapeOp>(op) ||
         isHoistableFromBranch(&op);
}

/// Clones `block` through `map` with every site replaced by its density, and
/// appends the per-site densities to `contributions`. Returns the density
/// total of `block`, or null when it holds no site.
static Value
cloneDensityBlock(OpBuilder &builder, Location loc, Block &block,
                  IRMapping &map, SmallVectorImpl<Value> &contributions,
                  function_ref<bool(Operation *)> isHoistableFromBranch);

/// Rebuilds `ifOp`, whose arms hold sites, with one extra result per site
/// (zero in the arm without that site) and the arm total last. Appends the
/// per-site results to `contributions` and returns the total.
static Value
rebuildSiteIf(OpBuilder &builder, Location loc, IfOp ifOp, IRMapping &map,
              SmallVectorImpl<Value> &contributions,
              function_ref<bool(Operation *)> isHoistableFromBranch) {
  SmallVector<SampleRegionOp> trueSites, falseSites;
  collectSitesInOrder(ifOp.getTrueBranch(), trueSites);
  collectSitesInOrder(ifOp.getFalseBranch(), falseSites);
  SmallVector<Type> types(ifOp.getResultTypes().begin(),
                          ifOp.getResultTypes().end());
  unsigned numOriginal = types.size();
  Type totalType;
  for (auto list : {&trueSites, &falseSites})
    for (SampleRegionOp site : *list) {
      Type t =
          site.getLogpdf().front().getTerminator()->getOperand(0).getType();
      types.push_back(t);
      totalType = t;
    }
  types.push_back(totalType);
  for (Region *arm : {&ifOp.getTrueBranch(), &ifOp.getFalseBranch()})
    for (Operation &armOp : arm->front().without_terminator()) {
      if (!isHoistableArmOp(armOp, isHoistableFromBranch))
        continue;
      bool outside = llvm::all_of(armOp.getOperands(), [&](Value v) {
        Operation *def = v.getDefiningOp();
        bool inArm = def ? arm->isAncestor(def->getParentRegion())
                         : arm->isAncestor(v.getParentRegion());
        return !inArm || map.contains(v);
      });
      if (outside)
        builder.clone(armOp, map);
    }
  auto newIf = IfOp::create(builder, loc, types,
                            map.lookupOrDefault(ifOp.getPredicate()));
  auto zeroOf = [&](OpBuilder &b, Type t) {
    return emitSplatConstant(b, loc, cast<RankedTensorType>(t), 0.0);
  };
  auto buildArm = [&](Region &src, Region &dst, bool isTrue) {
    Block *armBlock = new Block();
    dst.push_back(armBlock);
    OpBuilder armBuilder(armBlock, armBlock->end());
    SmallVector<Value> armContributions;
    Block &srcBlock = src.front();
    Value armTotal = cloneDensityBlock(armBuilder, loc, srcBlock, map,
                                       armContributions, isHoistableFromBranch);
    SmallVector<Value> yields;
    for (Value v : srcBlock.getTerminator()->getOperands())
      yields.push_back(map.lookupOrDefault(v));
    unsigned k = 0;
    for (size_t i = 0, e = trueSites.size(); i < e; ++i)
      yields.push_back(isTrue ? armContributions[k++]
                              : zeroOf(armBuilder, types[yields.size()]));
    for (size_t i = 0, e = falseSites.size(); i < e; ++i)
      yields.push_back(!isTrue ? armContributions[k++]
                               : zeroOf(armBuilder, types[yields.size()]));
    yields.push_back(armTotal ? armTotal : zeroOf(armBuilder, totalType));
    YieldOp::create(armBuilder, loc, yields);
  };
  buildArm(ifOp.getTrueBranch(), newIf.getTrueBranch(), /*isTrue=*/true);
  buildArm(ifOp.getFalseBranch(), newIf.getFalseBranch(), /*isTrue=*/false);
  for (unsigned i = 0; i < numOriginal; ++i)
    map.map(ifOp.getResult(i), newIf.getResult(i));
  unsigned numSites = trueSites.size() + falseSites.size();
  for (unsigned i = 0; i < numSites; ++i)
    contributions.push_back(newIf.getResult(numOriginal + i));
  return newIf.getResult(numOriginal + numSites);
}

static Value
cloneDensityBlock(OpBuilder &builder, Location loc, Block &block,
                  IRMapping &map, SmallVectorImpl<Value> &contributions,
                  function_ref<bool(Operation *)> isHoistableFromBranch) {
  Value total;
  auto addToTotal = [&](Value c) {
    total = total ? Value(arith::AddFOp::create(builder, loc, total, c)) : c;
  };
  for (Operation &op : block.without_terminator()) {
    if (auto site = dyn_cast<SampleRegionOp>(&op)) {
      // Pass the rng through so clones resolve; the dead chain is swept later.
      if (!site.getInputs().empty())
        map.map(site.getResult(0), map.lookupOrDefault(site.getInputs()[0]));
      Block &siteLogpdf = site.getLogpdf().front();
      unsigned numOutputs = site.getNumResults() - 1;
      IRMapping siteMap;
      unsigned argIdx = 0;
      for (unsigned i = 0; i < numOutputs; ++i, ++argIdx)
        siteMap.map(siteLogpdf.getArgument(argIdx),
                    map.lookup(site.getResult(i + 1)));
      for (unsigned i = 1, e = site.getInputs().size(); i < e; ++i, ++argIdx)
        siteMap.map(siteLogpdf.getArgument(argIdx),
                    map.lookupOrDefault(site.getInputs()[i]));
      Value contribution = cloneBlockBody(builder, siteLogpdf, siteMap).front();
      contributions.push_back(contribution);
      addToTotal(contribution);
      continue;
    }
    // Already hoisted above the enclosing rebuilt impulse.if.
    if (op.getNumResults() != 0 &&
        llvm::all_of(op.getResults(), [&](Value r) { return map.contains(r); }))
      continue;
    auto ifOp = dyn_cast<IfOp>(&op);
    if (!ifOp || !holdsSites(ifOp)) {
      builder.clone(op, map);
      continue;
    }
    addToTotal(rebuildSiteIf(builder, loc, ifOp, map, contributions,
                             isHoistableFromBranch));
  }
  return total;
}

bool impulse::constructLogpdfRegion(
    InferRegionOp regionOp,
    function_ref<bool(Operation *)> isHoistableFromBranch) {
  if (!regionOp.getLogpdf().empty())
    return false;
  Region &samplerRegion = regionOp.getSampling();
  if (samplerRegion.empty() || !samplerRegion.hasOneBlock())
    return false;
  ArrayAttr selection = regionOp.getSelectionAttr();
  ArrayAttr allAddresses = regionOp.getAllAddressesAttr();
  if (!selection || selection.empty() || !allAddresses)
    return false;

  Block &samplerEntry = samplerRegion.front();
  Location loc = regionOp.getLoc();

  // Site logpdf args: (<non-RNG outputs>..., <non-RNG inputs>...).
  DenseMap<Attribute, SampleRegionOp> siteByAddress;
  SmallVector<SampleRegionOp> allSites;
  collectSitesInOrder(samplerRegion, allSites);
  for (SampleRegionOp site : allSites) {
    for (Operation *p = site->getParentOp(); p != regionOp.getOperation();
         p = p->getParentOp())
      if (!isa<IfOp>(p))
        return false;
    ArrayAttr address = site.getAddressAttr();
    if (!address || site.getNumResults() < 2 || site.getLogpdf().empty() ||
        !site.getLogpdf().hasOneBlock())
      return false;
    unsigned numOutputs = site.getNumResults() - 1;
    unsigned numParams =
        site.getInputs().empty() ? 0 : site.getInputs().size() - 1;
    if (site.getLogpdf().front().getNumArguments() != numOutputs + numParams)
      return false;
    siteByAddress[address] = site;
  }
  if (siteByAddress.empty())
    return false;

  DenseSet<Attribute> selectedSet;
  for (Attribute addr : selection) {
    if (!siteByAddress.count(addr))
      return false;
    // A selected site's value is a position argument: it must be top level.
    if (siteByAddress.lookup(addr)->getBlock() != &samplerEntry)
      return false;
    selectedSet.insert(addr);
  }
  // Constrained branch sites read the constraint; a trace slot may be stale.
  DenseMap<Attribute, int64_t> constraintOffsetOf;
  if (regionOp.getConstraint()) {
    ArrayAttr constrained = regionOp.getConstrainedAddressesAttr();
    if (!constrained)
      return false;
    int64_t off = 0;
    for (Attribute addr : constrained) {
      SampleRegionOp site = siteByAddress.lookup(addr);
      if (!site)
        return false;
      constraintOffsetOf[addr] = off;
      for (unsigned i = 1, e = site.getNumResults(); i < e; ++i)
        off += cast<RankedTensorType>(site.getResult(i).getType())
                   .getNumElements();
    }
  }
  DenseSet<Attribute> allSet;
  bool hasUnselected = false;
  for (Attribute addr : allAddresses) {
    if (!siteByAddress.count(addr))
      return false;
    allSet.insert(addr);
    if (!selectedSet.contains(addr))
      hasUnselected = true;
  }
  for (auto &entry : siteByAddress)
    if (!allSet.contains(entry.first))
      return false;
  if (hasUnselected && !regionOp.getOriginalTrace())
    return false;

  // Validation passed; construction cannot fail from here on.
  Block *lpBlock = new Block();
  regionOp.getLogpdf().push_back(lpBlock);

  IRMapping map;

  // Position args in selection order (the flat lowering's position layout).
  int64_t positionSize = 0;
  for (Attribute addr : selection) {
    SampleRegionOp site = siteByAddress.lookup(addr);
    for (unsigned i = 1, e = site.getNumResults(); i < e; ++i) {
      Value result = site.getResult(i);
      BlockArgument arg =
          lpBlock->addArgument(result.getType(), result.getLoc());
      map.map(result, arg);
      positionSize += cast<RankedTensorType>(result.getType()).getNumElements();
    }
  }
  int64_t numPositionArgs = lpBlock->getNumArguments();

  // An MH move leaves unselected columns intact, so pins may skip past it.
  if (hasUnselected) {
    OpBuilder above(regionOp);
    Value trace = regionOp.getOriginalTrace();
    auto traceType = cast<RankedTensorType>(trace.getType());
    Type elemType = traceType.getElementType();
    auto pinnedFrom = [&](Attribute addr) {
      Value from = trace;
      while (auto prev = from.getDefiningOp<InferRegionOp>()) {
        Value prevTrace = prev.getOriginalTrace();
        if (prev.getTrace() != from || !prev.getMhConfig() || !prevTrace ||
            prevTrace.getType() != traceType ||
            prev.getAllAddressesAttr() != allAddresses ||
            llvm::is_contained(prev.getSelectionAttr().getValue(), addr))
          break;
        from = prevTrace;
      }
      return from;
    };
    int64_t offset = 0;
    for (Attribute addr : allAddresses) {
      SampleRegionOp site = siteByAddress.lookup(addr);
      bool selected = selectedSet.contains(addr);
      auto cIt = constraintOffsetOf.find(addr);
      bool fromConstraint = !selected && cIt != constraintOffsetOf.end() &&
                            site->getBlock() != &samplerEntry;
      Value siteTrace = selected         ? Value()
                        : fromConstraint ? regionOp.getConstraint()
                                         : pinnedFrom(addr);
      int64_t readOffset = fromConstraint ? cIt->second : offset;
      for (unsigned i = 1, e = site.getNumResults(); i < e; ++i) {
        auto resultType = cast<RankedTensorType>(site.getResult(i).getType());
        int64_t numElements = resultType.getNumElements();
        if (!selected) {
          auto sliceType = RankedTensorType::get({1, numElements}, elemType);
          Value slice = SliceOp::create(
              above, loc, sliceType, siteTrace,
              above.getDenseI64ArrayAttr({0, readOffset}),
              above.getDenseI64ArrayAttr({1, readOffset + numElements}),
              above.getDenseI64ArrayAttr({1, 1}));
          Value value = ReshapeOp::create(above, loc, resultType, slice);
          map.map(site.getResult(i), value);
        }
        offset += numElements;
        readOffset += numElements;
      }
    }
  }

  for (auto [blockArg, input] :
       llvm::zip(samplerEntry.getArguments(), regionOp.getInputs()))
    map.map(blockArg, input);

  OpBuilder builder(lpBlock, lpBlock->end());
  SmallVector<Value> contributions;
  Value total = cloneDensityBlock(builder, loc, samplerEntry, map,
                                  contributions, isHoistableFromBranch);

  // Yields the joint density, then per-site terms in score-tail order.
  SmallVector<Value> yields{total};
  yields.append(contributions.begin(), contributions.end());
  YieldOp::create(builder, loc, yields);

  // random/random_split are effect-free in a density though not marked Pure.
  // Users follow their operands, so one reverse pass erases every dead chain.
  Operation *terminator = lpBlock->getTerminator();
  for (Operation &op : llvm::make_early_inc_range(llvm::reverse(*lpBlock)))
    if (&op != terminator && op.use_empty() &&
        (isMemoryEffectFree(&op) || isa<RandomOp, RandomSplitOp>(op)))
      op.erase();

  regionOp.setNumPositionArgsAttr(builder.getI64IntegerAttr(numPositionArgs));
  regionOp.setPositionSizeAttr(builder.getI64IntegerAttr(positionSize));

  for (auto &entry : siteByAddress) {
    Region &siteLogpdf = entry.second.getLogpdf();
    for (Block &block : siteLogpdf)
      block.dropAllReferences();
    while (!siteLogpdf.empty())
      siteLogpdf.front().erase();
  }

  return true;
}
