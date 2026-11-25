//===- LogpdfRegion.cpp - Build the logpdf region of infer_region ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interfaces/LogpdfRegion.h"
#include "Interfaces/ImpulseUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace mlir::impulse;

/// Appends the sample sites of `region` that no other site holds to `sites`,
/// in the order they appear.
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

/// True if `op` holds a sample site.
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
  return op.hasTrait<OpTrait::ConstantLike>() || isHoistableFromBranch(&op);
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
      Type densityType =
          site.getLogpdf().front().getTerminator()->getOperand(0).getType();
      types.push_back(densityType);
      totalType = densityType;
    }
  types.push_back(totalType);
  for (Region *arm : {&ifOp.getTrueBranch(), &ifOp.getFalseBranch()})
    for (Operation &armOp : arm->front().without_terminator()) {
      if (!isHoistableArmOp(armOp, isHoistableFromBranch))
        continue;
      bool outside = llvm::all_of(armOp.getOperands(), [&](Value operand) {
        Operation *defOp = operand.getDefiningOp();
        bool inArm = defOp ? arm->isAncestor(defOp->getParentRegion())
                           : arm->isAncestor(operand.getParentRegion());
        return !inArm || map.contains(operand);
      });
      if (outside)
        builder.clone(armOp, map);
    }
  auto newIf = IfOp::create(builder, loc, types,
                            map.lookupOrDefault(ifOp.getPredicate()));
  auto zeroOf = [&](OpBuilder &armBuilder, Type type) {
    return emitSplatConstant(armBuilder, loc, cast<RankedTensorType>(type),
                             0.0);
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
    for (Value value : srcBlock.getTerminator()->getOperands())
      yields.push_back(map.lookupOrDefault(value));
    unsigned next = 0;
    for (size_t i = 0, e = trueSites.size(); i < e; ++i)
      yields.push_back(isTrue ? armContributions[next++]
                              : zeroOf(armBuilder, types[yields.size()]));
    for (size_t i = 0, e = falseSites.size(); i < e; ++i)
      yields.push_back(!isTrue ? armContributions[next++]
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
  auto addToTotal = [&](Value density) {
    total = total ? Value(arith::AddFOp::create(builder, loc, total, density))
                  : density;
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

namespace {
/// The sample sites of the `sampling` region of an infer_region, by address,
/// and where the logpdf region reads the value of each.
struct SiteTable {
  DenseMap<Attribute, SampleRegionOp> byAddress;
  /// The addresses of `selection`.
  DenseSet<Attribute> selected;
  /// The first column of each site of `constrained_addresses` in
  /// `constraint`.
  DenseMap<Attribute, int64_t> constraintOffset;
  /// True if a site of `all_addresses` is not in `selection`.
  bool hasUnselected = false;
};
} // namespace

/// The sites of `regionOp`, or failure when its logpdf region cannot be built
/// from them.
static FailureOr<SiteTable> collectSites(InferRegionOp regionOp) {
  ArrayAttr selection = regionOp.getSelectionAttr();
  ArrayAttr allAddresses = regionOp.getAllAddressesAttr();
  if (!selection || selection.empty() || !allAddresses)
    return failure();

  Block &samplerEntry = regionOp.getSampling().front();
  SiteTable sites;
  SmallVector<SampleRegionOp> allSites;
  collectSitesInOrder(regionOp.getSampling(), allSites);
  for (SampleRegionOp site : allSites) {
    for (Operation *parent = site->getParentOp();
         parent != regionOp.getOperation(); parent = parent->getParentOp())
      if (!isa<IfOp>(parent))
        return failure();
    ArrayAttr address = site.getAddressAttr();
    if (!address || site.getNumResults() < 2 || site.getLogpdf().empty())
      return failure();
    // Site logpdf args: (<non-RNG outputs>..., <non-RNG inputs>...).
    unsigned numOutputs = site.getNumResults() - 1;
    unsigned numParams =
        site.getInputs().empty() ? 0 : site.getInputs().size() - 1;
    if (site.getLogpdf().front().getNumArguments() != numOutputs + numParams)
      return failure();
    sites.byAddress[address] = site;
  }
  if (sites.byAddress.empty())
    return failure();

  for (Attribute address : selection) {
    SampleRegionOp site = sites.byAddress.lookup(address);
    // A selected site's value is a position argument: it must be top level.
    if (!site || site->getBlock() != &samplerEntry)
      return failure();
    sites.selected.insert(address);
  }
  if (regionOp.getConstraint()) {
    ArrayAttr constrained = regionOp.getConstrainedAddressesAttr();
    if (!constrained)
      return failure();
    int64_t offset = 0;
    for (Attribute address : constrained) {
      SampleRegionOp site = sites.byAddress.lookup(address);
      if (!site)
        return failure();
      sites.constraintOffset[address] = offset;
      for (Value value : llvm::drop_begin(site.getResults()))
        offset += cast<RankedTensorType>(value.getType()).getNumElements();
    }
  }
  DenseSet<Attribute> allSet;
  for (Attribute address : allAddresses) {
    if (!sites.byAddress.count(address))
      return failure();
    allSet.insert(address);
    if (!sites.selected.contains(address))
      sites.hasUnselected = true;
  }
  for (Attribute address : llvm::make_first_range(sites.byAddress))
    if (!allSet.contains(address))
      return failure();
  if (sites.hasUnselected && !regionOp.getOriginalTrace())
    return failure();
  return sites;
}

/// Adds to `logpdfBlock` one argument per non-RNG result of each site of
/// `selection`, in that order, and maps the result to it.
static void addPositionArgs(Block &logpdfBlock, ArrayAttr selection,
                            const SiteTable &sites, IRMapping &map) {
  for (Attribute address : selection) {
    SampleRegionOp site = sites.byAddress.lookup(address);
    for (Value result : llvm::drop_begin(site.getResults()))
      map.map(result,
              logpdfBlock.addArgument(result.getType(), result.getLoc()));
  }
}

/// The trace that `trace` derives from through a chain of MH infer_region ops
/// with the same `all_addresses` that do not select `address`, or `trace`. An
/// MH step keeps the columns of the sites it does not select, so the value at
/// `address` is the same in both.
static Value findColumnOrigin(Value trace, Attribute address,
                              ArrayAttr allAddresses) {
  Type traceType = trace.getType();
  Value origin = trace;
  while (auto prev = origin.getDefiningOp<InferRegionOp>()) {
    Value prevTrace = prev.getOriginalTrace();
    if (prev.getTrace() != origin || !prev.getMhConfig() || !prevTrace ||
        prevTrace.getType() != traceType ||
        prev.getAllAddressesAttr() != allAddresses ||
        llvm::is_contained(prev.getSelectionAttr().getValue(), address))
      break;
    origin = prevTrace;
  }
  return origin;
}

/// Maps each non-RNG result of a site that is not in `selection` to its value,
/// read above `regionOp` from `original_trace`, or from `constraint` for a
/// constrained site in a branch.
static void bindUnselectedColumns(InferRegionOp regionOp,
                                  const SiteTable &sites, IRMapping &map) {
  OpBuilder above(regionOp);
  Location loc = regionOp.getLoc();
  Block &samplerEntry = regionOp.getSampling().front();
  ArrayAttr allAddresses = regionOp.getAllAddressesAttr();
  Value trace = regionOp.getOriginalTrace();
  Type elemType = cast<RankedTensorType>(trace.getType()).getElementType();
  int64_t offset = 0;
  for (Attribute address : allAddresses) {
    SampleRegionOp site = sites.byAddress.lookup(address);
    bool selected = sites.selected.contains(address);
    // A branch that did not run leaves a stale value in the trace.
    auto constraintIt = sites.constraintOffset.find(address);
    bool fromConstraint = !selected &&
                          constraintIt != sites.constraintOffset.end() &&
                          site->getBlock() != &samplerEntry;
    Value siteTrace;
    if (fromConstraint)
      siteTrace = regionOp.getConstraint();
    else if (!selected)
      siteTrace = findColumnOrigin(trace, address, allAddresses);
    int64_t readOffset = fromConstraint ? constraintIt->second : offset;
    for (Value result : llvm::drop_begin(site.getResults())) {
      auto resultType = cast<RankedTensorType>(result.getType());
      int64_t numElements = resultType.getNumElements();
      if (!selected) {
        auto sliceType = RankedTensorType::get({1, numElements}, elemType);
        Value slice = SliceOp::create(
            above, loc, sliceType, siteTrace,
            above.getDenseI64ArrayAttr({0, readOffset}),
            above.getDenseI64ArrayAttr({1, readOffset + numElements}),
            above.getDenseI64ArrayAttr({1, 1}));
        map.map(result, ReshapeOp::create(above, loc, resultType, slice));
      }
      offset += numElements;
      readOffset += numElements;
    }
  }
}

/// Erases the unused ops of `block` that have no effect in a density.
static void eraseDeadOps(Block &block) {
  // random/random_split are effect-free in a density though not marked Pure.
  // Users follow their operands, so one reverse pass erases every dead chain.
  Operation *terminator = block.getTerminator();
  for (Operation &op : llvm::make_early_inc_range(llvm::reverse(block)))
    if (&op != terminator && op.use_empty() &&
        (isMemoryEffectFree(&op) || isa<RandomOp, RandomSplitOp>(op)))
      op.erase();
}

/// Empties the logpdf region of every site of `sites`.
static void dropSiteLogpdfs(const SiteTable &sites) {
  for (SampleRegionOp site : llvm::make_second_range(sites.byAddress)) {
    Region &siteLogpdf = site.getLogpdf();
    for (Block &block : siteLogpdf)
      block.dropAllReferences();
    while (!siteLogpdf.empty())
      siteLogpdf.front().erase();
  }
}

bool impulse::constructLogpdfRegion(
    InferRegionOp regionOp,
    function_ref<bool(Operation *)> isHoistableFromBranch) {
  if (!regionOp.getLogpdf().empty() || regionOp.getSampling().empty())
    return false;
  FailureOr<SiteTable> sites = collectSites(regionOp);
  if (failed(sites))
    return false;

  Block *logpdfBlock = new Block();
  regionOp.getLogpdf().push_back(logpdfBlock);
  IRMapping map;
  addPositionArgs(*logpdfBlock, regionOp.getSelectionAttr(), *sites, map);
  if (sites->hasUnselected)
    bindUnselectedColumns(regionOp, *sites, map);
  Block &samplerEntry = regionOp.getSampling().front();
  for (auto [blockArg, input] :
       llvm::zip(samplerEntry.getArguments(), regionOp.getInputs()))
    map.map(blockArg, input);

  Location loc = regionOp.getLoc();
  OpBuilder builder(logpdfBlock, logpdfBlock->end());
  SmallVector<Value> contributions;
  Value total = cloneDensityBlock(builder, loc, samplerEntry, map,
                                  contributions, isHoistableFromBranch);

  // Yields the joint density, then per-site terms in score-tail order.
  SmallVector<Value> yields{total};
  yields.append(contributions.begin(), contributions.end());
  YieldOp::create(builder, loc, yields);

  eraseDeadOps(*logpdfBlock);
  dropSiteLogpdfs(*sites);
  return true;
}
