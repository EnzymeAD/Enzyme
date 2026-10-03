//===- ImpulseRegionUtils.cpp - Utilities for impulse region-form ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interfaces/ImpulseRegionUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace mlir::impulse;

bool impulse::constructLogpdfRegion(InferRegionOp regionOp) {
  if (!regionOp.getLogpdf().empty())
    return false;
  Region &samplerRegion = regionOp.getSampler();
  if (samplerRegion.empty() || !samplerRegion.hasOneBlock())
    return false;
  ArrayAttr selection = regionOp.getSelectionAttr();
  ArrayAttr allAddresses = regionOp.getAllAddressesAttr();
  if (!selection || selection.empty() || !allAddresses)
    return false;

  Block &samplerEntry = samplerRegion.front();
  Location loc = regionOp.getLoc();

  // Collect top-level sites; every site must be addressable and scoreable,
  // and its per-site logpdf must follow the calling convention
  DenseMap<Attribute, SampleRegionOp> siteByAddress;
  for (Operation &op : samplerEntry) {
    auto site = dyn_cast<SampleRegionOp>(&op);
    if (!site)
      continue;
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
    selectedSet.insert(addr);
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
      Value siteTrace = selected ? Value() : pinnedFrom(addr);
      for (unsigned i = 1, e = site.getNumResults(); i < e; ++i) {
        auto resultType = cast<RankedTensorType>(site.getResult(i).getType());
        int64_t numElements = resultType.getNumElements();
        if (!selected) {
          auto sliceType = RankedTensorType::get({1, numElements}, elemType);
          Value slice = SliceOp::create(
              above, loc, sliceType, siteTrace,
              above.getDenseI64ArrayAttr({0, offset}),
              above.getDenseI64ArrayAttr({1, offset + numElements}),
              above.getDenseI64ArrayAttr({1, 1}));
          Value value = ReshapeOp::create(above, loc, resultType, slice);
          map.map(site.getResult(i), value);
        }
        offset += numElements;
      }
    }
  }

  for (auto [blockArg, input] :
       llvm::zip(samplerEntry.getArguments(), regionOp.getInputs()))
    map.map(blockArg, input);

  //    per-site logpdf, evaluated at the site's (position or pinned) value.
  OpBuilder builder(lpBlock, lpBlock->end());
  Value total;
  SmallVector<Value> contributions;
  for (Operation &op : samplerEntry.without_terminator()) {
    auto site = dyn_cast<SampleRegionOp>(&op);
    if (!site) {
      builder.clone(op, map);
      continue;
    }

    // Thread the rng chain through so downstream clones resolve; the chain is
    // dead in the density and swept in step 5.
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

    for (Operation &siteOp : siteLogpdf.without_terminator())
      builder.clone(siteOp, siteMap);
    Value contribution =
        siteMap.lookupOrDefault(siteLogpdf.getTerminator()->getOperand(0));
    contributions.push_back(contribution);
    total =
        total ? Value(arith::AddFOp::create(builder, loc, total, contribution))
              : contribution;
  }

  // Yields the joint density, then per-site terms in score-tail order.
  SmallVector<Value> yields{total};
  yields.append(contributions.begin(), contributions.end());
  YieldOp::create(builder, loc, yields);

  // random/randomSplit are effect-free in a density though not marked Pure.
  bool sweptAny = true;
  while (sweptAny) {
    sweptAny = false;
    SmallVector<Operation *> deadOps;
    for (Operation &op : *lpBlock) {
      if (&op == lpBlock->getTerminator())
        continue;
      if (op.use_empty() &&
          (isMemoryEffectFree(&op) || isa<RandomOp, RandomSplitOp>(op)))
        deadOps.push_back(&op);
    }
    for (Operation *op : llvm::reverse(deadOps)) {
      op->erase();
      sweptAny = true;
    }
  }

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
