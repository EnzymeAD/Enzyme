//===- ExpandImpulseMHGibbs.cpp - Lower MH and Gibbs inference ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers `impulse.infer` and `impulse.infer_region` with an MH config to
// repeated MH steps, batching runs of chained single-site moves, and
// `impulse.infer_region` with a Gibbs config to enumerative Gibbs sweeps.
//
//===----------------------------------------------------------------------===//

#include "ExpandImpulse.h"

#include "Dialect/Impulse/Impulse.h"
#include "Interfaces/ImpulseUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/BitVector.h"

#include <type_traits>

using namespace mlir;
using namespace mlir::impulse;

/// Packs the rank-0 `scalars` into a new vector of type `vecType`, in order.
static Value packScalars(OpBuilder &builder, Location loc,
                         RankedTensorType vecType, ArrayRef<Value> scalars) {
  auto cellType = RankedTensorType::get({1}, vecType.getElementType());
  Value vec = emitSplatConstant(builder, loc, vecType, 0.0);
  for (auto [index, scalar] : llvm::enumerate(scalars)) {
    Value cell = impulse::ReshapeOp::create(builder, loc, cellType, scalar);
    vec = impulse::DynamicUpdateSliceOp::create(
        builder, loc, vecType, vec, cell,
        ValueRange{emitI64ScalarConstant(builder, loc, index)});
  }
  return vec;
}

/// Returns the distance between consecutive `columns`, at least two, when it
/// is the same positive one throughout, else nothing.
static std::optional<int64_t> getEvenStride(ArrayRef<int64_t> columns) {
  int64_t stride = columns[1] - columns[0];
  if (stride < 1)
    return std::nullopt;
  for (size_t k = 2; k < columns.size(); ++k)
    if (columns[k] - columns[k - 1] != stride)
      return std::nullopt;
  return stride;
}

/// Gathers row 0 of the rank-2 `trace` at `columns` into a vector of type
/// `vecType`: with one strided slice when the columns are evenly spaced, else
/// one cell per column.
static Value gatherColumns(OpBuilder &b, Location loc, Value trace,
                           ArrayRef<int64_t> columns,
                           RankedTensorType vecType) {
  Type elemType = cast<RankedTensorType>(trace.getType()).getElementType();
  if (std::optional<int64_t> stride = getEvenStride(columns)) {
    auto rowType =
        RankedTensorType::get({1, (int64_t)columns.size()}, elemType);
    return impulse::ReshapeOp::create(
        b, loc, vecType,
        emitColumnSlice(b, loc, rowType, trace, columns.front(), *stride));
  }
  auto cellType = RankedTensorType::get({1, 1}, elemType);
  SmallVector<Value> cells;
  for (int64_t column : columns)
    cells.push_back(emitColumnSlice(b, loc, cellType, trace, column));
  return packScalars(b, loc, vecType, cells);
}

/// Writes entry `k` of the `1 x n` `row` into row 0 of the rank-2 `trace` at
/// `columns[k]`, one cell at a time.
static Value scatterRow(OpBuilder &b, Location loc, Value trace, Value row,
                        ArrayRef<int64_t> columns) {
  auto cellType = RankedTensorType::get(
      {1, 1}, cast<RankedTensorType>(row.getType()).getElementType());
  for (auto [k, column] : llvm::enumerate(columns))
    trace = updateColumns(b, loc, trace,
                          emitColumnSlice(b, loc, cellType, row, k), column);
  return trace;
}

/// Merges half-open [start, end) windows.
static ColumnWindows mergeColumnWindows(ColumnWindows windows) {
  llvm::sort(windows);
  ColumnWindows merged;
  for (auto w : windows) {
    if (w.first >= w.second)
      continue;
    if (!merged.empty() && w.first <= merged.back().second)
      merged.back().second = std::max(merged.back().second, w.second);
    else
      merged.push_back(w);
  }
  return merged;
}

/// Selects `proposed` over `trace` where `accepted`, window by window, so the
/// columns outside `windows` keep the values of `trace` and stay
/// loop-invariant.
static Value writeBackMHWindows(OpBuilder &b, Location loc, Value trace,
                                Value proposed, Value accepted,
                                ArrayRef<std::pair<int64_t, int64_t>> windows) {
  auto traceType = cast<RankedTensorType>(trace.getType());
  Value out = trace;
  for (const auto &window : windows) {
    int64_t start = window.first, end = window.second;
    auto winType =
        RankedTensorType::get({1, end - start}, traceType.getElementType());
    Value propWin = emitColumnSlice(b, loc, winType, proposed, start);
    Value curWin = emitColumnSlice(b, loc, winType, trace, start);
    Value upd =
        impulse::SelectOp::create(b, loc, winType, accepted, propWin, curWin);
    out = updateColumns(b, loc, out, upd, start);
  }
  return out;
}

ColumnWindows impulse::mhChangedColumnsFlat(const FlatTraceLayout &layout,
                                            ArrayAttr regenerateAddresses,
                                            int64_t width) {
  ColumnWindows windows;
  for (const FlatTraceLayout::Entry &entry : layout.entries) {
    impulse::SampleOp site = layout.siteBySymbol.lookup(entry.address[0]);
    bool changed =
        regenerateAddresses &&
        llvm::is_contained(regenerateAddresses.getValue(), entry.address);
    // Constrained sites outside branches read the old trace (readsConstraint).
    if (!changed && width == layout.positionSize)
      changed = (!site.getDistAttr() && !site.getLogpdfFnAttr()) ||
                entry.address.size() > 1 ||
                site->getParentOfType<impulse::IfOp>();
    if (changed)
      windows.push_back({entry.column, entry.column + entry.size});
  }
  if (width > layout.positionSize)
    windows.push_back({layout.positionSize, width});
  return mergeColumnWindows(windows);
}

namespace {
/// The MH accept test of a proposal: whether it is accepted, the weight it
/// carries when accepted, and the rng after the uniform draw.
struct MHDecision {
  Value accepted;
  Value acceptWeight;
  Value rng;
};
} // namespace

/// Emits the MH accept test of `proposal` against the current `weight`: the
/// proposal is accepted when the log of a uniform draw from its rng falls
/// below log alpha. With `hasScoreTail`, `proposal.weight` is the change of
/// the joint density rather than the new joint density.
static MHDecision emitMHDecision(OpBuilder &builder, Location loc, Value weight,
                                 const MHProposal &proposal,
                                 bool hasScoreTail) {
  Value jointRatio =
      hasScoreTail
          ? proposal.weight
          : arith::SubFOp::create(builder, loc, proposal.weight, weight)
                .getResult();
  Value logAlpha =
      proposal.logRatio
          ? proposal.logRatio
          : arith::SubFOp::create(builder, loc, jointRatio, proposal.correction)
                .getResult();
  Value acceptWeight =
      hasScoreTail
          ? arith::AddFOp::create(builder, loc, weight, proposal.weight)
                .getResult()
          : proposal.weight;
  Value rng = proposal.rng;
  Value uniform = emitUniformDraw(builder, loc, rng,
                                  cast<RankedTensorType>(weight.getType()));
  Value logRand = math::LogOp::create(builder, loc, uniform);
  Value accepted = arith::CmpFOp::create(
      builder, loc, arith::CmpFPredicate::OLT, logRand, logAlpha);
  return {accepted, acceptWeight, rng};
}

MHStep impulse::emitMHStep(OpBuilder &builder, Location loc, Value trace,
                           Value weight, const MHProposal &proposal,
                           bool hasScoreTail) {
  MHDecision decision =
      emitMHDecision(builder, loc, weight, proposal, hasScoreTail);
  Value newTrace = writeBackMHWindows(builder, loc, trace, proposal.trace,
                                      decision.accepted, proposal.changed);
  Value newWeight = arith::SelectOp::create(builder, loc, decision.accepted,
                                            decision.acceptWeight, weight);
  return {newTrace, newWeight, decision.rng, decision.accepted};
}

/// Returns, for each per-site yield of `lpBlock` (terminator operands 1..),
/// whether it depends on `seeds`.
static SmallVector<bool> taintedSiteYields(Block &lpBlock, ValueRange seeds) {
  DenseSet<Value> tainted(seeds.begin(), seeds.end());
  for (Operation &op : lpBlock.without_terminator())
    if (usesAnyOf(&op, tainted))
      tainted.insert(op.result_begin(), op.result_end());
  SmallVector<bool> dirty;
  for (Value v : llvm::drop_begin(lpBlock.getTerminator()->getOperands()))
    dirty.push_back(tainted.contains(v));
  return dirty;
}

namespace {
/// Maps values to the selected sites they depend on, as bit vectors over
/// `numSites`. An `impulse.if` result depends on its predicate only when its
/// two yielded values differ.
struct SiteDependence {
  unsigned numSites;
  DenseMap<Value, llvm::BitVector> sites;

  explicit SiteDependence(unsigned numSites) : numSites(numSites) {}

  llvm::BitVector get(Value v) const {
    auto it = sites.find(v);
    return it == sites.end() ? llvm::BitVector(numSites) : it->second;
  }
  void seed(Value v, unsigned k) {
    llvm::BitVector s(numSites);
    s.set(k);
    sites[v] = s;
  }
  void seed(Value v, const llvm::BitVector &s) {
    if (s.any())
      sites[v] = s;
  }
  void run(Block &block) {
    for (Operation &op : block.without_terminator())
      visit(op);
  }
  void visit(Operation &op) {
    // Sample sites are seeded (selected) or fixed by the trace (unselected).
    if (isa<impulse::SampleRegionOp>(op))
      return;
    if (auto ifOp = dyn_cast<impulse::IfOp>(op)) {
      llvm::BitVector predicateSites = get(ifOp.getPredicate());
      Block &trueBlock = ifOp.getTrueBranch().front();
      Block &falseBlock = ifOp.getFalseBranch().front();
      run(trueBlock);
      run(falseBlock);
      Operation *trueYield = trueBlock.getTerminator();
      Operation *falseYield = falseBlock.getTerminator();
      for (auto [i, result] : llvm::enumerate(ifOp.getResults())) {
        Value trueValue = trueYield->getOperand(i);
        Value falseValue = falseYield->getOperand(i);
        llvm::BitVector resultSites = get(trueValue);
        resultSites |= get(falseValue);
        if (trueValue != falseValue)
          resultSites |= predicateSites;
        if (resultSites.any())
          sites[result] = resultSites;
      }
      return;
    }
    llvm::BitVector s(numSites);
    for (Value v : op.getOperands())
      s |= get(v);
    if (op.getNumRegions() != 0) {
      SetVector<Value> captured;
      getUsedValuesDefinedAbove(op.getRegions(), captured);
      for (Value v : captured)
        s |= get(v);
    }
    if (s.any())
      for (Value r : op.getResults())
        sites[r] = s;
  }
};
} // namespace

namespace {
/// The sampler of a region-form MH move, cloned with fresh values for the
/// selected sites.
struct SamplerProposal {
  /// The trace with the fresh values in their columns.
  Value trace;
  /// The key after the last site.
  Value rng;
  /// The fresh values of each selected address.
  DenseMap<Attribute, SmallVector<Value>> valuesOf;
  /// The program-order index of each selected site, which is also the index
  /// of its per-site yield and of its score-tail column.
  SmallVector<int64_t> selectedOrdinals;
  /// The columns of the fresh values.
  ColumnWindows valueColumns;
};
} // namespace

/// Clones the sampler of `op` from key `rng`. The selected sites draw fresh
/// values, which go into `trace` at their columns of `layout`; the other sites
/// read theirs from `trace`. A draw of a selected site for which
/// `drawOverride` returns a value takes that value instead of being cloned.
static SamplerProposal
cloneSamplerWithProposal(OpBuilder &b, Location loc, impulse::InferRegionOp op,
                         const RegionTraceLayout &layout, Value trace,
                         Value rng,
                         function_ref<Value(Operation *)> drawOverride) {
  ArrayAttr selection = op.getSelectionAttr();
  DenseSet<Attribute> selectedSet(selection.begin(), selection.end());
  Block &samplerBlock = op.getSampling().front();
  IRMapping map;
  map.map(samplerBlock.getArgument(0), rng);
  for (unsigned i = 1, e = samplerBlock.getNumArguments(); i < e; ++i)
    map.map(samplerBlock.getArgument(i), op.getInputs()[i]);

  SamplerProposal proposal;
  proposal.trace = trace;
  proposal.rng = rng;
  int64_t siteOrdinal = -1;
  for (Operation &bodyOp : samplerBlock.without_terminator()) {
    auto site = dyn_cast<impulse::SampleRegionOp>(&bodyOp);
    if (!site) {
      b.clone(bodyOp, map);
      continue;
    }
    ++siteOrdinal;
    Attribute addr = site.getAddressAttr();
    int64_t siteColumn = layout.columnOf.lookup(addr);
    if (!selectedSet.contains(addr)) {
      map.map(site.getResult(0), proposal.rng);
      map.map(site.getResults().drop_front(),
              sliceTraceValues(b, loc, trace, siteColumn,
                               TypeRange(site->getResultTypes()).drop_front()));
      continue;
    }
    proposal.selectedOrdinals.push_back(siteOrdinal);
    proposal.valueColumns.push_back(
        {siteColumn, siteColumn + sampleRegionSize(site)});
    Block &sub = site.getSampling().front();
    IRMapping subMap;
    subMap.map(sub.getArgument(0), proposal.rng);
    for (unsigned j = 1, e = site.getInputs().size(); j < e; ++j)
      subMap.map(sub.getArgument(j), map.lookupOrDefault(site.getInputs()[j]));
    for (Operation &subOp : sub.without_terminator()) {
      Value drawn = drawOverride ? drawOverride(&subOp) : Value();
      if (!drawn) {
        b.clone(subOp, subMap);
        continue;
      }
      subMap.map(subOp.getResult(0),
                 subMap.lookupOrDefault(subOp.getOperand(0)));
      subMap.map(subOp.getResult(1), drawn);
    }
    Operation *samplerYield = sub.getTerminator();
    proposal.rng = subMap.lookupOrDefault(samplerYield->getOperand(0));
    map.map(site.getResult(0), proposal.rng);
    SmallVector<Value> values;
    int64_t column = siteColumn;
    for (unsigned k = 1, e = site.getNumResults(); k < e; ++k) {
      Value v = subMap.lookupOrDefault(samplerYield->getOperand(k));
      map.map(site.getResult(k), v);
      values.push_back(v);
      proposal.trace = writeTraceColumns(b, loc, proposal.trace, v, column);
      column += cast<RankedTensorType>(v.getType()).getNumElements();
    }
    proposal.valuesOf[addr] = values;
  }
  return proposal;
}

/// Returns the Hastings correction of the selected sites of `sampled`: the
/// sum of their new minus old per-site scores, the new ones in `newYields`.
/// The old ones come from the score tail of `trace`, or on a trace that holds
/// exactly the positions of `layout`, from the logpdf region of `op` at the
/// old positions.
static Value emitProposalCorrection(OpBuilder &b, Location loc,
                                    impulse::InferRegionOp op,
                                    const RegionTraceLayout &layout,
                                    Value trace, SamplerProposal &sampled,
                                    ArrayRef<Value> newYields) {
  int64_t positionSize = layout.positionSize;
  bool exactWidth =
      cast<RankedTensorType>(trace.getType()).getShape()[1] == positionSize;
  SmallVector<Value> oldYields;
  if (exactWidth) {
    SmallVector<Value> oldPositions;
    for (Attribute addr : op.getSelectionAttr())
      llvm::append_range(
          oldPositions,
          sliceTraceValues(b, loc, trace, layout.columnOf.lookup(addr),
                           ValueRange(sampled.valuesOf[addr]).getTypes()));
    oldYields = cloneBlockBody(b, op.getLogpdf().front(), oldPositions);
  }
  Value correction;
  for (int64_t j : sampled.selectedOrdinals) {
    Value oldScore = exactWidth
                         ? oldYields[1 + j]
                         : scalarAt(b, loc, trace, {0, positionSize + j});
    accumulate(b, loc, correction,
               arith::SubFOp::create(b, loc, newYields[1 + j], oldScore));
  }
  return correction;
}

/// Returns the sum of the changes of the `dirty` per-site scores, from the
/// score tail of `trace` at `positionSize` to `newYields`. With `splitRatio`,
/// also returns log alpha: the same sum without the terms of the selected
/// sites at `selectedOrdinals`, which cancel against the proposal density.
static std::pair<Value, Value>
emitDeltaWeight(OpBuilder &b, Location loc, Value trace, int64_t positionSize,
                ArrayRef<Value> newYields, ArrayRef<bool> dirty,
                ArrayRef<int64_t> selectedOrdinals, bool splitRatio) {
  llvm::SmallDenseSet<int64_t> ownTerms(selectedOrdinals.begin(),
                                        selectedOrdinals.end());
  Value sum;
  Value ratio;
  for (auto [j, isDirty] : llvm::enumerate(dirty)) {
    if (!isDirty)
      continue;
    Value oldScore = scalarAt(b, loc, trace, {0, positionSize + (int64_t)j});
    Value change = arith::SubFOp::create(b, loc, newYields[1 + j], oldScore);
    accumulate(b, loc, sum, change);
    if (splitRatio && !ownTerms.contains(j))
      accumulate(b, loc, ratio, change);
  }
  auto scalarType = RankedTensorType::get({}, b.getF64Type());
  Value logRatio = splitRatio ? sumOrZero(b, loc, ratio, scalarType) : Value();
  return {sumOrZero(b, loc, sum, scalarType), logRatio};
}

/// Proposes fresh values for the selected sites of `op` from its sampler,
/// reads the other sites from `trace` and scores the result with the logpdf
/// region; `drawOverride` is as for `cloneSamplerWithProposal`. With `delta`,
/// the weight is the sum of new-minus-old scores over the terms the move can
/// change, else the new joint density; with `withLogRatio` too, a move of
/// several sites on a score-tail trace also returns log alpha.
static MHProposal
proposeRegionMove(OpBuilder &b, Location loc, impulse::InferRegionOp op,
                  const RegionTraceLayout &layout, Value trace, Value rng,
                  bool delta, bool withLogRatio,
                  function_ref<Value(Operation *)> drawOverride = nullptr) {
  SamplerProposal sampled =
      cloneSamplerWithProposal(b, loc, op, layout, trace, rng, drawOverride);
  int64_t positionSize = layout.positionSize;
  int64_t width = cast<RankedTensorType>(trace.getType()).getShape()[1];
  bool hasScoreTail = width > positionSize;

  Block &lpBlock = op.getLogpdf().front();
  SmallVector<Value> newPositions;
  for (Attribute addr : op.getSelectionAttr())
    llvm::append_range(newPositions, sampled.valuesOf[addr]);
  SmallVector<Value> newYields = cloneBlockBody(b, lpBlock, newPositions);
  SmallVector<bool> dirty =
      delta ? taintedSiteYields(lpBlock, lpBlock.getArguments())
            : SmallVector<bool>(newYields.size() - 1, true);

  MHProposal proposal;
  proposal.trace = sampled.trace;
  proposal.weight = newYields[0];
  proposal.rng = sampled.rng;
  if (hasScoreTail) {
    Value row0 = emitI64ScalarConstant(b, loc, 0);
    SmallVector<std::pair<int64_t, Value>> cells;
    for (auto [j, isDirty] : llvm::enumerate(dirty))
      if (isDirty)
        cells.push_back({positionSize + (int64_t)j, newYields[1 + j]});
    proposal.trace = writeScoreCells(b, loc, proposal.trace, row0, cells);
  }
  proposal.correction =
      emitProposalCorrection(b, loc, op, layout, trace, sampled, newYields);
  if (delta)
    std::tie(proposal.weight, proposal.logRatio) = emitDeltaWeight(
        b, loc, trace, positionSize, newYields, dirty, sampled.selectedOrdinals,
        withLogRatio && sampled.selectedOrdinals.size() >= 2 && hasScoreTail);

  // Only dirty score columns can change; write back just those.
  proposal.changed = std::move(sampled.valueColumns);
  if (delta) {
    for (auto [j, isDirty] : llvm::enumerate(dirty))
      if (isDirty)
        proposal.changed.push_back(
            {positionSize + (int64_t)j, positionSize + (int64_t)j + 1});
  } else if (hasScoreTail) {
    proposal.changed.push_back({positionSize, width});
  }
  proposal.changed = mergeColumnWindows(std::move(proposal.changed));
  return proposal;
}

template <typename OpT>
MHProposal impulse::proposeFlatMove(OpBuilder &b, Location loc, OpT op,
                                    ValueRange inputs, Value trace,
                                    StringAttr name, TypeRange fnResultTypes) {
  auto weightType = RankedTensorType::get({2}, b.getF64Type());
  SmallVector<Type> regenResultTypes{trace.getType(), weightType};
  llvm::append_range(regenResultTypes, fnResultTypes);
  auto regen = impulse::RegenerateOp::create(
      b, loc, regenResultTypes, op.getFnAttr(), inputs, trace,
      op.getConstraint(), /*all_addresses=*/op.getAllAddressesAttr(),
      /*selection=*/op.getSelectionAttr(), op.getConstrainedAddressesAttr(),
      name);
  regen->setAttr(kMHHastingsAttr, b.getUnitAttr());
  auto [weight, correction] = unpackHastingsWeight(b, loc, regen.getWeight());
  return {regen.getNewTrace(), weight, correction, regen.getOutputs()[0],
          /*logRatio=*/Value()};
}

template MHProposal
impulse::proposeFlatMove<impulse::MHOp>(OpBuilder &, Location, impulse::MHOp,
                                        ValueRange, Value, StringAttr,
                                        TypeRange);
template MHProposal
impulse::proposeFlatMove<impulse::InferOp>(OpBuilder &, Location,
                                           impulse::InferOp, ValueRange, Value,
                                           StringAttr, TypeRange);

/// Emits `step` `count` times, each time on the values the previous step
/// returned, and returns the values of the last step. A single step runs in
/// place on `start`. More steps run in an `impulse.for` that carries the values
/// `loopInit` creates after the loop bounds, or `start` without `loopInit`.
static SmallVector<Value>
emitRepeated(OpBuilder &builder, Location loc, int64_t count, ValueRange start,
             function_ref<SmallVector<Value>()> loopInit,
             function_ref<SmallVector<Value>(ValueRange)> step) {
  if (count <= 1)
    return step(start);
  Value lowerBound = emitI64ScalarConstant(builder, loc, 0);
  Value upperBound = emitI64ScalarConstant(builder, loc, count);
  Value one = emitI64ScalarConstant(builder, loc, 1);
  SmallVector<Value> init = loopInit ? loopInit() : SmallVector<Value>(start);
  auto forOp = impulse::ForOp::create(builder, loc, ValueRange(init).getTypes(),
                                      lowerBound, upperBound, one, init);
  Block *body = createForBody(builder, forOp);
  impulse::YieldOp::create(builder, loc,
                           step(body->getArguments().drop_front()));
  builder.setInsertionPointAfter(forOp);
  return SmallVector<Value>(forOp.getResults());
}

//===----------------------------------------------------------------------===//
// Batched runs of single-site MH moves
//
// A run of chained single-site MH moves becomes one draw of all proposals and
// one vector accept test. That is sound only when each per-site yield depends
// on at most one site of the run, no site's parameters depend on another,
// written columns are disjoint and all proposals are one matching draw.
//===----------------------------------------------------------------------===//

namespace {
/// A single-site MH move that can join a batched run: an
/// `impulse.infer_region` regenerating the site at `address`, whose sampler
/// takes its value from the single `draw`.
struct MHMove {
  impulse::InferRegionOp op;
  Attribute address;
  impulse::SampleRegionOp site;
  impulse::RandomOp draw;
  /// The value columns of the site and the score columns the move can change.
  ColumnWindows windows;
  /// The score of per-site yield `j` sits at column `layout.positionSize + j`.
  RegionTraceLayout layout;
  /// Index of the site's per-site yield, in program order.
  int64_t ordinal = -1;
};

/// The ops between the first and the last move of a run that read the traces
/// of the moves before the last.
struct InterleavedOps {
  /// The trace of move `i`, for every move but the last.
  DenseMap<Value, unsigned> intermediate;
  /// Ops that read an intermediate trace directly or through another reader,
  /// in program order.
  SmallVector<Operation *> readers;
  DenseSet<Operation *> readerSet;
  /// The moves whose written columns each reader result can depend on.
  DenseMap<Value, llvm::BitVector> moveDependence;
  /// Reader results used after the run, recomputed on the final trace.
  SetVector<Value> liveOut;
};
} // namespace

/// Returns `op` as a move that can join a batched run: a single-site MH move
/// on a score-tail trace whose site draws one uniform or normal value with
/// parameters defined outside `op`. Returns nothing for any other op.
static std::optional<MHMove> analyzeMHMove(impulse::InferRegionOp op) {
  if (!op.getMhConfig().has_value() || op.getSampling().empty() ||
      op.getLogpdf().empty() || op.getNumSamples() > 1 ||
      op.getStateIn().size() != 1 || op.getInputs().empty() ||
      op.getConstraint() || !op.getOriginalTrace())
    return std::nullopt;
  ArrayAttr selection = op.getSelectionAttr();
  ArrayAttr allAddresses = op.getAllAddressesAttr();
  if (!selection || selection.size() != 1 || !allAddresses)
    return std::nullopt;
  Block &lpBlock = op.getLogpdf().front();
  SmallVector<impulse::SampleRegionOp> sites = llvm::to_vector(
      op.getSampling().front().getOps<impulse::SampleRegionOp>());
  Operation *lpTerm = lpBlock.getTerminator();
  if (lpTerm->getNumOperands() != 1 + sites.size() ||
      lpBlock.getNumArguments() != 1)
    return std::nullopt;
  FailureOr<RegionTraceLayout> layout =
      getRegionTraceLayout(op, /*topLevelOnly=*/true);
  if (failed(layout))
    return std::nullopt;
  Attribute address = selection[0];
  impulse::SampleRegionOp site = layout->siteByAddress.lookup(address);
  if (!site || site.getNumResults() != 2)
    return std::nullopt;
  int64_t positionSize = layout->positionSize;
  auto traceType = cast<RankedTensorType>(op.getOriginalTrace().getType());
  if (traceType.getRank() != 2 || traceType.getShape()[0] != 1 ||
      traceType.getShape()[1] <= positionSize)
    return std::nullopt;
  Block &sub = site.getSampling().front();
  if (sub.getNumArguments() == 0)
    return std::nullopt;
  impulse::RandomOp draw;
  for (Operation &o : sub.without_terminator()) {
    if (o.getNumRegions() != 0)
      return std::nullopt;
    if (auto r = dyn_cast<impulse::RandomOp>(&o)) {
      if (draw)
        return std::nullopt;
      draw = r;
    }
  }
  if (!draw || draw.getRngState() != sub.getArgument(0) ||
      !sub.getArgument(0).hasOneUse() ||
      !draw.getOutputRngState().hasOneUse() ||
      sub.getTerminator()->getOperand(0) != draw.getOutputRngState())
    return std::nullopt;
  impulse::RngDistribution dist = draw.getRngDistribution();
  if (dist != impulse::RngDistribution::UNIFORM &&
      dist != impulse::RngDistribution::NORMAL)
    return std::nullopt;
  // Parameters must dominate the batched draw, which precedes every move.
  auto definedOutside = [&](Value v) {
    return !op->isAncestor(v.getParentRegion()->getParentOp());
  };
  if (!isa<RankedTensorType>(draw.getResult().getType()) ||
      !definedOutside(draw.getA()) || !definedOutside(draw.getB()))
    return std::nullopt;
  MHMove m;
  m.op = op;
  m.address = address;
  m.site = site;
  m.draw = draw;
  m.ordinal = llvm::find(sites, site) - sites.begin();
  int64_t column = layout->columnOf.lookup(address);
  m.windows.push_back({column, column + sampleRegionSize(site)});
  SmallVector<bool> dirty = taintedSiteYields(lpBlock, lpBlock.getArguments());
  for (auto [j, d] : llvm::enumerate(dirty))
    if (d)
      m.windows.push_back(
          {positionSize + (int64_t)j, positionSize + (int64_t)j + 1});
  m.windows = mergeColumnWindows(m.windows);
  m.layout = std::move(*layout);
  return m;
}

/// Returns whether `next` continues a run that ends with `prev`: it is the next
/// op of the block that reads the trace, key and weight only `prev` produces,
/// with the same other inputs and addresses.
static bool continuesRun(const MHMove &prev, const MHMove &next) {
  impulse::InferRegionOp p = prev.op, n = next.op;
  if (p->getBlock() != n->getBlock() || !p->isBeforeInBlock(n))
    return false;
  if (n.getOriginalTrace() != p.getTrace() ||
      n.getInputs()[0] != p.getOutputRngState() ||
      n.getStateIn()[0] != p.getStateOut()[0])
    return false;
  if (n.getInputs().size() != p.getInputs().size() ||
      n.getAllAddressesAttr() != p.getAllAddressesAttr() ||
      n.getOriginalTrace().getType() != p.getOriginalTrace().getType())
    return false;
  for (unsigned i = 1, e = n.getInputs().size(); i < e; ++i)
    if (n.getInputs()[i] != p.getInputs()[i])
      return false;
  return p.getOutputRngState().hasOneUse() && p.getStateOut()[0].hasOneUse();
}

/// Returns whether `a` and `b` are one value or equal constants of one type.
static bool sameValueOrConstant(Value a, Value b) {
  if (a == b)
    return true;
  Attribute ca, cb;
  return matchPattern(a, m_Constant(&ca)) && matchPattern(b, m_Constant(&cb)) &&
         ca == cb && a.getType() == b.getType();
}

/// Collects the ops between the moves of `moves` that read intermediate
/// traces. Returns nothing when such an op has memory effects or feeds a move,
/// an intermediate trace has another reader, or a reader result used after the
/// run would differ on the final trace.
static std::optional<InterleavedOps>
analyzeInterleavedOps(MutableArrayRef<MHMove> moves) {
  InterleavedOps interleaved;
  unsigned numMoves = moves.size();
  Block *block = moves.front().op->getBlock();
  Operation *first = moves.front().op, *last = moves.back().op;
  DenseSet<Operation *> moveOps;
  for (MHMove &m : moves)
    moveOps.insert(m.op);
  for (unsigned i = 0; i + 1 < numMoves; ++i)
    interleaved.intermediate[moves[i].op.getTrace()] = i;
  auto writtenBy = [&](int64_t c, unsigned from, unsigned to) {
    llvm::BitVector s(numMoves);
    for (unsigned i = from; i < to; ++i)
      for (auto [windowStart, windowEnd] : moves[i].windows)
        if (c >= windowStart && c < windowEnd)
          s.set(i);
    return s;
  };
  // Same value on the final trace as on the intermediate traces it reads.
  DenseMap<Operation *, bool> finalSafe;
  for (Operation *o = first->getNextNode(); o && o != last;
       o = o->getNextNode()) {
    if (moveOps.contains(o))
      continue;
    bool fromIntermediate = false, safe = true;
    llvm::BitVector s(numMoves);
    auto addValue = [&](Value v) {
      if (interleaved.intermediate.count(v)) {
        fromIntermediate = true;
        safe = false;
        s.set();
      } else if (Operation *d = v.getDefiningOp();
                 d && interleaved.readerSet.contains(d)) {
        fromIntermediate = true;
        safe &= finalSafe.lookup(d);
        s |= interleaved.moveDependence.lookup(v);
      }
    };
    auto slice = dyn_cast<impulse::SliceOp>(o);
    if (slice && interleaved.intermediate.count(slice.getOperand())) {
      ArrayRef<int64_t> starts = slice.getStartIndices(),
                        limits = slice.getLimitIndices(),
                        strides = slice.getStrides();
      if (starts.size() != 2)
        return std::nullopt;
      unsigned i = interleaved.intermediate.lookup(slice.getOperand());
      fromIntermediate = true;
      for (int64_t c = starts[1]; c < limits[1]; c += strides[1]) {
        s |= writtenBy(c, 0, numMoves);
        safe &= writtenBy(c, i + 1, numMoves).none();
      }
    } else {
      for (Value v : o->getOperands())
        addValue(v);
      if (o->getNumRegions() != 0) {
        SetVector<Value> captured;
        getUsedValuesDefinedAbove(o->getRegions(), captured);
        for (Value v : captured)
          addValue(v);
      }
    }
    if (!fromIntermediate)
      continue;
    if (!isMemoryEffectFree(o))
      return std::nullopt;
    interleaved.readers.push_back(o);
    interleaved.readerSet.insert(o);
    finalSafe[o] = safe;
    for (Value r : o->getResults())
      interleaved.moveDependence[r] = s;
  }
  for (unsigned i = 0; i + 1 < numMoves; ++i)
    for (OpOperand &u : moves[i].op.getTrace().getUses()) {
      if (u.getOwner() == moves[i + 1].op.getOperation() &&
          u.getOperandNumber() ==
              moves[i + 1].op.getOriginalTraceMutable()[0].getOperandNumber())
        continue;
      Operation *anc = block->findAncestorOpInBlock(*u.getOwner());
      if (!anc || !interleaved.readerSet.contains(anc))
        return std::nullopt;
    }
  for (Operation *o : interleaved.readers)
    for (Value r : o->getResults())
      for (OpOperand &u : r.getUses()) {
        Operation *anc = block->findAncestorOpInBlock(*u.getOwner());
        // A reader result may be captured by a move's regions, not be its
        // operand.
        if (!anc || (moveOps.contains(anc) && anc == u.getOwner()))
          return std::nullopt;
        if (last->isBeforeInBlock(anc)) {
          if (!finalSafe.lookup(o))
            return std::nullopt;
          interleaved.liveOut.insert(r);
        }
      }
  return interleaved;
}

/// Returns the ops between the moves of the run `moves` when the run can be
/// emitted as one batched move, else nothing.
static std::optional<InterleavedOps>
analyzeBatch(MutableArrayRef<MHMove> moves) {
  unsigned numMoves = moves.size();
  Block *block = moves.front().op->getBlock();
  Operation *last = moves.back().op;
  impulse::RandomOp firstDraw = moves.front().draw;
  for (MHMove &m : moves) {
    if (m.draw.getRngDistribution() != firstDraw.getRngDistribution() ||
        m.draw.getResult().getType() != firstDraw.getResult().getType() ||
        !sameValueOrConstant(m.draw.getA(), firstDraw.getA()) ||
        !sameValueOrConstant(m.draw.getB(), firstDraw.getB()))
      return std::nullopt;
  }
  ColumnWindows all;
  for (MHMove &m : moves)
    for (auto w : m.windows)
      all.push_back(w);
  llvm::sort(all);
  for (unsigned i = 1; i < all.size(); ++i)
    if (all[i].first < all[i - 1].second)
      return std::nullopt;
  for (MHMove &m : moves)
    for (Value v : {m.op.getDiagnostics(), m.op.getLogDensities()})
      for (OpOperand &u : v.getUses()) {
        Operation *anc = block->findAncestorOpInBlock(*u.getOwner());
        if (!anc || !last->isBeforeInBlock(anc))
          return std::nullopt;
      }
  std::optional<InterleavedOps> interleaved = analyzeInterleavedOps(moves);
  if (!interleaved)
    return std::nullopt;
  DenseMap<Value, unsigned> &intermediate = interleaved->intermediate;
  DenseMap<Value, llvm::BitVector> &moveDependence =
      interleaved->moveDependence;
  for (unsigned k = 0; k < numMoves; ++k) {
    impulse::InferRegionOp op = moves[k].op;
    auto seedCaptured = [&](SiteDependence &dep, Region &r) {
      SetVector<Value> captured;
      getUsedValuesDefinedAbove(r, captured);
      for (Value v : captured) {
        if (intermediate.count(v))
          return false;
        auto it = moveDependence.find(v);
        if (it != moveDependence.end())
          dep.seed(v, it->second);
      }
      return true;
    };
    SiteDependence lpDep(numMoves);
    if (!seedCaptured(lpDep, op.getLogpdf()))
      return std::nullopt;
    Block &lpBlock = op.getLogpdf().front();
    lpDep.seed(lpBlock.getArgument(0), k);
    lpDep.run(lpBlock);
    for (Value y : llvm::drop_begin(lpBlock.getTerminator()->getOperands()))
      if (lpDep.get(y).count() > 1)
        return std::nullopt;
    SiteDependence sDep(numMoves);
    if (!seedCaptured(sDep, op.getSampling()))
      return std::nullopt;
    Block &sBlock = op.getSampling().front();
    for (Operation &o : sBlock)
      if (auto s = dyn_cast<impulse::SampleRegionOp>(&o))
        for (unsigned j = 0; j < numMoves; ++j)
          if (s.getAddressAttr() == moves[j].address)
            for (Value r : llvm::drop_begin(s.getResults()))
              sDep.seed(r, j);
    sDep.run(sBlock);
    for (Value param : llvm::drop_begin(moves[k].site->getOperands())) {
      llvm::BitVector p = sDep.get(param);
      p.reset(k);
      if (p.any())
        return std::nullopt;
    }
  }
  return interleaved;
}

namespace {
/// A batchable run and the ops between its moves.
struct MHBatch {
  SmallVector<MHMove> moves;
  InterleavedOps interleaved;
};

/// The proposals of a batched run.
struct BatchedProposals {
  /// The key after the draw of all proposals and the accept uniforms.
  Value rngOut;
  /// The logs of the accept uniforms, one per move.
  Value logRand;
  SmallVector<MHProposal> moves;
};

/// The columns the moves of a batched run change, when they line up: slot
/// `s` holds a value column in every move or a score column in every move,
/// and the score of each move's own site sits at slot `ownSlot`.
struct SlotLayout {
  /// The columns of each move, in slot order.
  SmallVector<SmallVector<int64_t>> columns;
  int64_t ownSlot = -1;
  /// The proposed and the current value of each slot, one entry per move.
  SmallVector<Value> proposedAt;
  SmallVector<Value> currentAt;
};

/// The slots of a `SlotLayout` grouped into windows of one row per move, where
/// each move's columns are the previous move's shifted by `moveStride`.
struct StrideGroups {
  int64_t moveStride;
  /// The first column of each window in move 0 and the slots in the window.
  SmallVector<std::pair<int64_t, SmallVector<size_t>>> groups;
};
} // namespace

/// Emits one draw of the proposals of all `moves` from key `rng0`, the
/// uniforms of their accept tests as a `vecType` vector, and each move's
/// proposal from `trace0` with its draw taken from the batched one.
static BatchedProposals proposeAll(OpBuilder &b, Location loc,
                                   MutableArrayRef<MHMove> moves, Value trace0,
                                   Value rng0, RankedTensorType vecType) {
  impulse::RandomOp firstDraw = moves.front().draw;
  auto drawType = cast<RankedTensorType>(firstDraw.getResult().getType());
  SmallVector<int64_t> batchedShape{(int64_t)moves.size()};
  llvm::append_range(batchedShape, drawType.getShape());
  auto draws = impulse::RandomOp::create(
      b, loc,
      TypeRange{rng0.getType(),
                RankedTensorType::get(batchedShape, drawType.getElementType())},
      rng0, firstDraw.getA(), firstDraw.getB(),
      firstDraw.getRngDistributionAttr());
  BatchedProposals batched;
  batched.rngOut = draws.getOutputRngState();
  Value uniform = emitUniformDraw(b, loc, batched.rngOut, vecType);
  batched.logRand = math::LogOp::create(b, loc, uniform);
  for (auto [k, move] : llvm::enumerate(moves)) {
    Operation *draw = move.draw;
    Value drawn = emitLeadingEntry(b, loc, draws.getResult(), k);
    batched.moves.push_back(proposeRegionMove(
        b, loc, move.op, move.layout, trace0, rng0, /*delta=*/true,
        /*withLogRatio=*/false,
        [&](Operation *o) { return o == draw ? drawn : Value(); }));
  }
  return batched;
}

/// Returns how the columns the `proposals` of `moves` change line up in
/// slots, or nothing when they do not.
static std::optional<SlotLayout>
findSlotLayout(MutableArrayRef<MHMove> moves, ArrayRef<MHProposal> proposals) {
  SlotLayout slots;
  for (const MHProposal &proposal : proposals) {
    SmallVector<int64_t> &columns = slots.columns.emplace_back();
    for (auto [windowStart, windowEnd] : proposal.changed)
      for (int64_t c = windowStart; c < windowEnd; ++c)
        columns.push_back(c);
  }
  ArrayRef<int64_t> firstColumns = slots.columns.front();
  int64_t firstPositionSize = moves.front().layout.positionSize;
  for (auto [k, move] : llvm::enumerate(moves)) {
    ArrayRef<int64_t> columns = slots.columns[k];
    if (columns.size() != firstColumns.size())
      return std::nullopt;
    int64_t positionSize = move.layout.positionSize;
    int64_t own = -1;
    for (auto [s, column] : llvm::enumerate(columns)) {
      if ((column >= positionSize) != (firstColumns[s] >= firstPositionSize))
        return std::nullopt;
      if (column == positionSize + move.ordinal)
        own = s;
    }
    if (own < 0 || (k != 0 && own != slots.ownSlot))
      return std::nullopt;
    slots.ownSlot = own;
  }
  return slots;
}

/// Gathers the proposed and the current value of each slot of `slots` across
/// the moves into `vecType` vectors, from the proposals' traces and `trace0`.
static void emitSlotVectors(OpBuilder &b, Location loc, Value trace0,
                            ArrayRef<MHProposal> proposals, SlotLayout &slots,
                            RankedTensorType vecType) {
  auto cellType = RankedTensorType::get(
      {1, 1}, cast<RankedTensorType>(trace0.getType()).getElementType());
  ArrayRef<SmallVector<int64_t>> columns = slots.columns;
  for (size_t s = 0, e = columns.front().size(); s < e; ++s) {
    SmallVector<Value> proposed;
    SmallVector<int64_t> slotColumns;
    for (auto [proposal, moveColumns] : llvm::zip(proposals, columns)) {
      proposed.push_back(
          emitColumnSlice(b, loc, cellType, proposal.trace, moveColumns[s]));
      slotColumns.push_back(moveColumns[s]);
    }
    slots.proposedAt.push_back(packScalars(b, loc, vecType, proposed));
    slots.currentAt.push_back(
        gatherColumns(b, loc, trace0, slotColumns, vecType));
  }
}

/// Emits the weight changes and the log alphas of the batched `proposals` as
/// `vecType` vectors: from the slot vectors when the columns line up in
/// `slots`, which this fills, else from each proposal's own values.
static std::pair<Value, Value> emitLogAlpha(OpBuilder &b, Location loc,
                                            Value trace0, int64_t positionSize,
                                            ArrayRef<MHProposal> proposals,
                                            std::optional<SlotLayout> &slots,
                                            RankedTensorType vecType) {
  if (!slots) {
    SmallVector<Value> deltas, corrections;
    for (const MHProposal &proposal : proposals) {
      deltas.push_back(proposal.weight);
      corrections.push_back(proposal.correction);
    }
    Value deltaVec = packScalars(b, loc, vecType, deltas);
    return {deltaVec,
            arith::SubFOp::create(b, loc, deltaVec,
                                  packScalars(b, loc, vecType, corrections))};
  }
  emitSlotVectors(b, loc, trace0, proposals, *slots, vecType);
  Value deltaVec;
  for (auto [s, column] : llvm::enumerate(slots->columns.front()))
    if (column >= positionSize)
      accumulate(b, loc, deltaVec,
                 arith::SubFOp::create(b, loc, slots->proposedAt[s],
                                       slots->currentAt[s]));
  int64_t own = slots->ownSlot;
  Value correctionVec = arith::SubFOp::create(b, loc, slots->proposedAt[own],
                                              slots->currentAt[own]);
  return {deltaVec, arith::SubFOp::create(b, loc, deltaVec, correctionVec)};
}

/// Groups the slots of `slots` into windows of one row per move, when each
/// move's columns are the previous move's shifted by one stride and the
/// windows fit in a `width`-wide trace without overlapping.
static std::optional<StrideGroups> findStrideGroups(const SlotLayout &slots,
                                                    int64_t width) {
  ArrayRef<SmallVector<int64_t>> columns = slots.columns;
  int64_t numMoves = columns.size();
  size_t numSlots = columns.front().size();
  int64_t moveStride = columns[1][0] - columns[0][0];
  if (moveStride < 1)
    return std::nullopt;
  for (size_t s = 0; s < numSlots; ++s)
    for (int64_t k = 0; k < numMoves; ++k)
      if (columns[k][s] != columns[0][s] + k * moveStride)
        return std::nullopt;
  SmallVector<size_t> order(numSlots);
  for (size_t i = 0; i < numSlots; ++i)
    order[i] = i;
  llvm::sort(order,
             [&](size_t x, size_t y) { return columns[0][x] < columns[0][y]; });
  SmallVector<std::pair<int64_t, SmallVector<size_t>>> groups;
  for (size_t s : order) {
    if (!groups.empty() && columns[0][s] < groups.back().first + moveStride)
      groups.back().second.push_back(s);
    else
      groups.push_back({columns[0][s], {s}});
  }
  for (size_t i = 0; i < groups.size(); ++i) {
    int64_t end = groups[i].first + numMoves * moveStride;
    if (end > width || (i + 1 != groups.size() && end > groups[i + 1].first))
      return std::nullopt;
  }
  return StrideGroups{moveStride, std::move(groups)};
}

/// Writes the values of the accepted proposals into `trace0`: window by
/// window per move without `slots`, else per slot vector, through blocks of
/// one row per move where the moves' columns are evenly strided.
static Value writeBackAccepted(OpBuilder &b, Location loc, Value trace0,
                               ArrayRef<MHProposal> proposals, Value accepted,
                               ArrayRef<Value> acceptedOf,
                               const std::optional<SlotLayout> &slots) {
  Value trace = trace0;
  if (!slots) {
    for (auto [proposal, acc] : llvm::zip(proposals, acceptedOf))
      trace = writeBackMHWindows(b, loc, trace, proposal.trace, acc,
                                 proposal.changed);
    return trace;
  }
  int64_t numMoves = proposals.size();
  auto traceType = cast<RankedTensorType>(trace0.getType());
  Type elemType = traceType.getElementType();
  SmallVector<Value> resultAt;
  for (auto [proposed, current] :
       llvm::zip(slots->proposedAt, slots->currentAt))
    resultAt.push_back(
        arith::SelectOp::create(b, loc, accepted, proposed, current));
  ArrayRef<SmallVector<int64_t>> columns = slots->columns;
  std::optional<StrideGroups> strided =
      findStrideGroups(*slots, traceType.getShape()[1]);
  if (!strided) {
    auto rowType = RankedTensorType::get({1, numMoves}, elemType);
    for (size_t s = 0, e = resultAt.size(); s < e; ++s) {
      Value row = impulse::ReshapeOp::create(b, loc, rowType, resultAt[s]);
      SmallVector<int64_t> slotColumns =
          llvm::map_to_vector(columns, [&](ArrayRef<int64_t> moveColumns) {
            return moveColumns[s];
          });
      trace = scatterRow(b, loc, trace, row, slotColumns);
    }
    return trace;
  }
  int64_t moveStride = strided->moveStride;
  auto blockType = RankedTensorType::get({numMoves, moveStride}, elemType);
  auto blockColType = RankedTensorType::get({numMoves, 1}, elemType);
  auto winType = RankedTensorType::get({1, numMoves * moveStride}, elemType);
  for (auto &[groupStart, groupSlots] : strided->groups) {
    Value block = impulse::ReshapeOp::create(
        b, loc, blockType,
        emitColumnSlice(b, loc, winType, trace0, groupStart));
    for (size_t s : groupSlots) {
      Value slotCol =
          impulse::ReshapeOp::create(b, loc, blockColType, resultAt[s]);
      block = updateColumns(b, loc, block, slotCol, columns[0][s] - groupStart);
    }
    Value blockWin = impulse::ReshapeOp::create(b, loc, winType, block);
    trace = updateColumns(b, loc, trace, blockWin, groupStart);
  }
  return trace;
}

/// Replaces the results of the moves of `batch` with the batched `trace`,
/// `weight`, key `rngOut` and accept flags `acceptedOf`, recomputes the
/// interleaved readers used after the run on the final trace, and erases the
/// moves. `readsIntermediate` lists the operands of each reader that read an
/// intermediate trace.
static void replaceRun(
    OpBuilder &b, Location loc, MHBatch &batch,
    const DenseMap<Operation *, SmallVector<unsigned>> &readsIntermediate,
    Value trace, Value weight, Value rngOut, ArrayRef<Value> acceptedOf) {
  const InterleavedOps &interleaved = batch.interleaved;
  impulse::InferRegionOp last = batch.moves.back().op;
  if (!interleaved.liveOut.empty()) {
    IRMapping cloneMap;
    for (Operation *o : interleaved.readers) {
      Operation *c = b.clone(*o, cloneMap);
      for (unsigned i : readsIntermediate.lookup(o))
        c->setOperand(i, trace);
    }
    for (Value v : interleaved.liveOut) {
      Value clonedValue = cloneMap.lookup(v);
      v.replaceUsesWithIf(clonedValue, [&](OpOperand &u) {
        Operation *anc = last->getBlock()->findAncestorOpInBlock(*u.getOwner());
        return anc && last->isBeforeInBlock(anc);
      });
    }
  }

  for (auto [move, acc] : llvm::zip(batch.moves, acceptedOf)) {
    impulse::InferRegionOp op = move.op;
    op.getDiagnostics().replaceAllUsesWith(acc);
    auto ldType = cast<ShapedType>(op.getLogDensities().getType());
    if (!op.getLogDensities().use_empty())
      op.getLogDensities().replaceAllUsesWith(
          arith::ConstantOp::create(b, loc, b.getZeroAttr(ldType)));
  }
  last.getTrace().replaceAllUsesWith(trace);
  last.getOutputRngState().replaceAllUsesWith(rngOut);
  last.getStateOut()[0].replaceAllUsesWith(weight);
  for (MHMove &move : llvm::reverse(batch.moves))
    move.op->erase();
}

/// Replaces the run of `batch` with one draw of all proposals, a vector accept
/// test and the write-back of the accepted columns.
static void emitBatchedRun(MHBatch &batch) {
  MutableArrayRef<MHMove> moves = batch.moves;
  unsigned numMoves = moves.size();
  impulse::InferRegionOp first = moves.front().op, last = moves.back().op;
  Location loc = last.getLoc();
  Value trace0 = first.getOriginalTrace();
  Value weight0 = first.getStateIn()[0];
  Value rng0 = first.getInputs()[0];
  DenseMap<Operation *, SmallVector<unsigned>> readsIntermediate;
  for (Operation *o : batch.interleaved.readers)
    for (OpOperand &u : o->getOpOperands())
      if (batch.interleaved.intermediate.count(u.get()))
        readsIntermediate[o].push_back(u.getOperandNumber());
  for (unsigned i = 0; i + 1 < numMoves; ++i)
    moves[i].op.getTrace().replaceUsesWithIf(trace0, [&](OpOperand &u) {
      return !(
          u.getOwner() == moves[i + 1].op.getOperation() &&
          u.getOperandNumber() ==
              moves[i + 1].op.getOriginalTraceMutable()[0].getOperandNumber());
    });

  OpBuilder b(last);
  auto vecType = RankedTensorType::get(
      {(int64_t)numMoves},
      cast<RankedTensorType>(weight0.getType()).getElementType());
  BatchedProposals proposals = proposeAll(b, loc, moves, trace0, rng0, vecType);
  std::optional<SlotLayout> slots = findSlotLayout(moves, proposals.moves);
  auto [deltaVec, logAlphaVec] =
      emitLogAlpha(b, loc, trace0, moves.front().layout.positionSize,
                   proposals.moves, slots, vecType);
  Value accepted = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                         proposals.logRand, logAlphaVec);
  Value weight = weight0;
  SmallVector<Value> acceptedOf;
  // In sweep order, matching the sequential weight updates.
  for (unsigned k = 0; k < numMoves; ++k) {
    Value acc = emitLeadingEntry(b, loc, accepted, k);
    acceptedOf.push_back(acc);
    Value acceptWeight = arith::AddFOp::create(
        b, loc, weight, emitLeadingEntry(b, loc, deltaVec, k));
    weight = arith::SelectOp::create(b, loc, acc, acceptWeight, weight);
  }
  Value trace = writeBackAccepted(b, loc, trace0, proposals.moves, accepted,
                                  acceptedOf, slots);
  replaceRun(b, loc, batch, readsIntermediate, trace, weight, proposals.rngOut,
             acceptedOf);
}

void impulse::batchMHRuns(Operation *root) {
  SmallVector<SmallVector<MHMove>> runs;
  root->walk([&](Block *block) {
    SmallVector<MHMove> run;
    for (Operation &o : *block) {
      auto op = dyn_cast<impulse::InferRegionOp>(&o);
      if (!op)
        continue;
      std::optional<MHMove> m = analyzeMHMove(op);
      if (!m)
        continue;
      if (!run.empty() && !continuesRun(run.back(), *m)) {
        if (run.size() >= 2)
          runs.push_back(run);
        run.clear();
      }
      run.push_back(*m);
    }
    if (run.size() >= 2)
      runs.push_back(run);
  });
  // Longest piece first: a shorter one can fail where a longer one holds.
  // Pieces are capped so the search for the longest one stays bounded.
  constexpr size_t kMaxBatchSize = 256;
  SmallVector<MHBatch, 0> batches;
  for (SmallVector<MHMove> &run : runs) {
    size_t i = 0;
    while (i < run.size()) {
      size_t j = std::min(run.size(), i + kMaxBatchSize) - 1;
      std::optional<InterleavedOps> interleaved;
      for (; j > i; --j)
        if ((interleaved = analyzeBatch(
                 MutableArrayRef<MHMove>(run).slice(i, j + 1 - i))))
          break;
      if (j > i)
        batches.push_back(
            {SmallVector<MHMove>(run.begin() + i, run.begin() + j + 1),
             std::move(*interleaved)});
      i = j + 1;
    }
  }
  for (MHBatch &batch : batches)
    emitBatchedRun(batch);
}

//===----------------------------------------------------------------------===//
// Enumerative Gibbs over scalar BERNOULLI sites
//===----------------------------------------------------------------------===//

namespace {
/// The values a Gibbs sweep carries.
struct GibbsState {
  Value trace;
  Value weight;
  Value rng;
};

/// A constrained site in a branch: its trace columns and where its observed
/// values start in the constraint.
struct ObservedColumns {
  int64_t traceColumn;
  int64_t constraintOffset;
  int64_t size;
};

/// Emits the sweeps of an enumerative Gibbs `impulse.infer_region` over scalar
/// BERNOULLI sites: a selected site becomes 1 when a uniform draw falls below
/// logistic(s1 - s0), with s_v the log density of the terms the site affects
/// at value v. The batched sweep updates all selected sites at once; it is
/// sound only when each per-site yield depends on at most one selected site
/// and no site's parameters depend on another.
class GibbsSweepEmitter {
public:
  /// Checks the regions, trace and sites of `op` and analyses its logpdf
  /// region. Fails after emitting on `op`.
  static FailureOr<GibbsSweepEmitter> analyze(impulse::InferRegionOp op);
  /// Emits one sweep over the selected sites from `state`.
  GibbsState emitSweep(OpBuilder &b, GibbsState state) {
    return batched ? emitBatched(b, state) : emitSequential(b, state);
  }

private:
  explicit GibbsSweepEmitter(impulse::InferRegionOp op)
      : op(op), loc(op.getLoc()) {}

  /// Finds the selected site each per-site yield depends on and decides
  /// whether the sweep can be batched.
  void analyzeBatching(const RegionTraceLayout &layout);
  GibbsState emitBatched(OpBuilder &b, GibbsState state);
  GibbsState emitSequential(OpBuilder &b, GibbsState state);
  /// Writes `value`, flattened, into row 0 of `trace` from `column` on.
  Value writeValue(OpBuilder &b, Value trace, int64_t column, Value value);
  /// Writes the observed values of the constrained sites in branches into
  /// `trace`, whose readers take those sites from it.
  Value writeObserved(OpBuilder &b, Value trace);

  impulse::InferRegionOp op;
  Location loc;
  RankedTensorType traceType;
  RankedTensorType weightType;
  int64_t positionSize = 0;
  bool hasScoreTail = false;
  unsigned numYields = 0;
  /// The trace column of each selected site.
  SmallVector<int64_t> selectedCols;
  SmallVector<ObservedColumns> observed;
  /// For each selected site, whether each per-site yield depends on it.
  SmallVector<SmallVector<bool>> dirtyOf;
  /// The selected site each per-site yield depends on, or -1 for none.
  SmallVector<int> siteOfYield;
  bool batched = false;
};
} // namespace

FailureOr<GibbsSweepEmitter>
GibbsSweepEmitter::analyze(impulse::InferRegionOp op) {
  if (op.getSampling().empty() || op.getLogpdf().empty())
    return op.emitError("Impulse Gibbs: infer_region requires sampler and "
                        "constructed logpdf regions (run sicm first)");
  if (op.getStateIn().size() != 1 || op.getInputs().empty() ||
      !op.getOriginalTrace())
    return op.emitError("Impulse Gibbs: expects the rng input, a trace and a "
                        "single `weight` state_in slot");
  GibbsSweepEmitter emitter(op);
  Block &samplerEntry = op.getSampling().front();
  Block &lpBlock = op.getLogpdf().front();
  emitter.numYields = lpBlock.getTerminator()->getNumOperands() - 1;
  ArrayAttr selection = op.getSelectionAttr();

  FailureOr<RegionTraceLayout> layout =
      getRegionTraceLayout(op, /*topLevelOnly=*/false);
  if (failed(layout))
    return op.emitError("Impulse Gibbs: all_addresses entry without a site");
  emitter.positionSize = layout->positionSize;
  emitter.traceType = cast<RankedTensorType>(op.getOriginalTrace().getType());
  emitter.weightType = cast<RankedTensorType>(op.getStateIn()[0].getType());
  int64_t width = emitter.traceType.getShape()[1];
  emitter.hasScoreTail = width > emitter.positionSize;
  if (emitter.hasScoreTail &&
      width != emitter.positionSize + (int64_t)emitter.numYields)
    return op.emitError("Impulse Gibbs: score-tail width does not match the "
                        "logpdf region's per-site yields");
  if (lpBlock.getNumArguments() != selection.size())
    return op.emitError("Impulse Gibbs: one position argument per selected "
                        "site expected");
  for (auto [k, addr] : llvm::enumerate(selection)) {
    impulse::SampleRegionOp site = layout->siteByAddress.lookup(addr);
    impulse::DistAttr dist = site ? site.getDistAttr() : nullptr;
    if (!site || site->getBlock() != &samplerEntry || !dist ||
        dist.getBase() != impulse::DistBaseKind::BERNOULLI ||
        (dist.getTransforms() && !dist.getTransforms().empty()) ||
        site.getNumResults() != 2 ||
        cast<RankedTensorType>(lpBlock.getArgument(k).getType())
                .getNumElements() != 1)
      return op.emitError("Impulse Gibbs: every selected site must be a "
                          "top-level scalar BERNOULLI site");
    emitter.selectedCols.push_back(layout->columnOf.lookup(addr));
  }
  if (op.getConstraint()) {
    int64_t offset = 0;
    for (Attribute addr : op.getConstrainedAddressesAttr()) {
      impulse::SampleRegionOp site = layout->siteByAddress.lookup(addr);
      if (!site)
        return op.emitError(
            "Impulse Gibbs: constrained address without a site");
      if (site->getBlock() != &samplerEntry)
        emitter.observed.push_back(
            {layout->columnOf.lookup(addr), offset, sampleRegionSize(site)});
      offset += sampleRegionSize(site);
    }
  }
  for (unsigned k = 0; k < selection.size(); ++k)
    emitter.dirtyOf.push_back(
        taintedSiteYields(lpBlock, ValueRange{lpBlock.getArgument(k)}));
  emitter.analyzeBatching(*layout);
  return emitter;
}

void GibbsSweepEmitter::analyzeBatching(const RegionTraceLayout &layout) {
  Block &lpBlock = op.getLogpdf().front();
  Operation *lpTerm = lpBlock.getTerminator();
  ArrayAttr selection = op.getSelectionAttr();
  unsigned numSites = selection.size();
  siteOfYield.assign(numYields, -1);
  batched = numSites >= 2;
  if (!batched)
    return;
  SiteDependence lpDep(numSites);
  for (unsigned k = 0; k < numSites; ++k)
    lpDep.seed(lpBlock.getArgument(k), k);
  lpDep.run(lpBlock);
  for (unsigned j = 0; j < numYields; ++j) {
    llvm::BitVector s = lpDep.get(lpTerm->getOperand(1 + j));
    if (s.count() > 1) {
      batched = false;
      return;
    }
    if (s.any())
      siteOfYield[j] = s.find_first();
  }
  SiteDependence samplerDep(numSites);
  SmallVector<impulse::SampleRegionOp> selectedSites;
  for (auto [k, addr] : llvm::enumerate(selection)) {
    impulse::SampleRegionOp site = layout.siteByAddress.lookup(addr);
    selectedSites.push_back(site);
    samplerDep.seed(site.getResult(1), k);
  }
  samplerDep.run(op.getSampling().front());
  for (impulse::SampleRegionOp site : selectedSites)
    for (Value param : llvm::drop_begin(site->getOperands()))
      if (samplerDep.get(param).any())
        batched = false;
}

GibbsState GibbsSweepEmitter::emitBatched(OpBuilder &b, GibbsState state) {
  Block &lpBlock = op.getLogpdf().front();
  int64_t n = selectedCols.size();
  Type elemType = traceType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);
  auto vecType = RankedTensorType::get({n}, elemType);
  auto rowType = RankedTensorType::get({1, n}, elemType);
  std::optional<int64_t> stride = getEvenStride(selectedCols);
  Value curVec;
  if (stride) {
    curVec = gatherColumns(b, loc, state.trace, selectedCols, vecType);
  } else {
    SmallVector<Value> cur;
    for (int64_t col : selectedCols)
      cur.push_back(scalarAt(b, loc, state.trace, {0, col}));
    curVec = packScalars(b, loc, vecType, cur);
  }

  // All sites at v at once: each yield sees at most one selected site.
  SmallVector<Value> yieldsAt[2];
  for (int v = 0; v < 2; ++v) {
    SmallVector<Value> positions;
    for (BlockArgument arg : lpBlock.getArguments())
      positions.push_back(emitSplatConstant(
          b, loc, cast<RankedTensorType>(arg.getType()), (double)v));
    llvm::append_range(yieldsAt[v],
                       llvm::drop_begin(cloneBlockBody(b, lpBlock, positions)));
  }
  SmallVector<Value> logOdds(n);
  for (unsigned j = 0; j < numYields; ++j) {
    int k = siteOfYield[j];
    if (k < 0)
      continue;
    accumulate(b, loc, logOdds[k],
               arith::SubFOp::create(b, loc, yieldsAt[1][j], yieldsAt[0][j]));
  }
  for (Value &l : logOdds)
    l = sumOrZero(b, loc, l, scalarType);
  Value lVec = packScalars(b, loc, vecType, logOdds);

  Value p1 = impulse::LogisticOp::create(b, loc, vecType, lVec);
  Value uniform = emitUniformDraw(b, loc, state.rng, vecType);
  Value take1 =
      arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, uniform, p1);
  Value curIs1 =
      arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT, curVec,
                            emitSplatConstant(b, loc, vecType, 0.5));
  Value zeros = emitSplatConstant(b, loc, vecType, 0.0);
  Value newVals = arith::SelectOp::create(
      b, loc, take1, emitSplatConstant(b, loc, vecType, 1.0), zeros);

  // +L_k on a 0 -> 1 flip, -L_k on 1 -> 0, else 0.
  Value negL = arith::NegFOp::create(b, loc, lVec);
  Value toOne = arith::SelectOp::create(b, loc, curIs1, zeros, lVec);
  Value toZero = arith::SelectOp::create(b, loc, curIs1, negL, zeros);
  Value delta = arith::SelectOp::create(b, loc, take1, toOne, toZero);
  for (int64_t k = 0; k < n; ++k)
    state.weight = arith::AddFOp::create(b, loc, state.weight,
                                         emitLeadingEntry(b, loc, delta, k));

  Value newRow = impulse::ReshapeOp::create(b, loc, rowType, newVals);
  if (stride == 1)
    state.trace = updateColumns(b, loc, state.trace, newRow, selectedCols[0]);
  else
    state.trace = scatterRow(b, loc, state.trace, newRow, selectedCols);
  if (hasScoreTail)
    for (unsigned j = 0; j < numYields; ++j)
      if (siteOfYield[j] >= 0)
        state.trace = writeValue(
            b, state.trace, positionSize + j,
            arith::SelectOp::create(
                b, loc, emitLeadingEntry(b, loc, take1, siteOfYield[j]),
                yieldsAt[1][j], yieldsAt[0][j]));
  state.trace = writeObserved(b, state.trace);
  return state;
}

GibbsState GibbsSweepEmitter::emitSequential(OpBuilder &b, GibbsState state) {
  Block &lpBlock = op.getLogpdf().front();
  SmallVector<Value> cur;
  for (auto [k, col] : llvm::enumerate(selectedCols))
    cur.push_back(
        emitRowSlice(b, loc, state.trace, col,
                     cast<RankedTensorType>(lpBlock.getArgument(k).getType())));
  auto scalarType = RankedTensorType::get({}, traceType.getElementType());
  for (unsigned k = 0; k < selectedCols.size(); ++k) {
    auto argType = cast<RankedTensorType>(lpBlock.getArgument(k).getType());
    SmallVector<Value> yieldsAt[2];
    Value scoreAt[2];
    for (int v = 0; v < 2; ++v) {
      SmallVector<Value> positions(cur);
      positions[k] = emitSplatConstant(b, loc, argType, (double)v);
      SmallVector<Value> yields = cloneBlockBody(b, lpBlock, positions);
      for (unsigned j = 0; j < numYields; ++j) {
        Value y = yields[1 + j];
        yieldsAt[v].push_back(y);
        if (dirtyOf[k][j])
          accumulate(b, loc, scoreAt[v], y);
      }
      scoreAt[v] = sumOrZero(b, loc, scoreAt[v], scalarType);
    }
    Value diff = arith::SubFOp::create(b, loc, scoreAt[1], scoreAt[0]);
    Value p1 = impulse::LogisticOp::create(b, loc, diff.getType(), diff);
    Value uniform = emitUniformDraw(b, loc, state.rng, weightType);
    Value take1 =
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, uniform, p1);
    Value curScalar = impulse::ReshapeOp::create(b, loc, scalarType, cur[k]);
    Value curIs1 =
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT, curScalar,
                              emitSplatConstant(b, loc, scalarType, 0.5));
    Value sNew = arith::SelectOp::create(b, loc, take1, scoreAt[1], scoreAt[0]);
    Value sCur =
        arith::SelectOp::create(b, loc, curIs1, scoreAt[1], scoreAt[0]);
    state.weight = arith::AddFOp::create(
        b, loc, state.weight, arith::SubFOp::create(b, loc, sNew, sCur));
    Value oneArg = emitSplatConstant(b, loc, argType, 1.0);
    Value zeroArg = emitSplatConstant(b, loc, argType, 0.0);
    Value newVal = arith::SelectOp::create(b, loc, take1, oneArg, zeroArg);
    cur[k] = newVal;
    state.trace = writeValue(b, state.trace, selectedCols[k], newVal);
    if (hasScoreTail)
      for (unsigned j = 0; j < numYields; ++j)
        if (dirtyOf[k][j])
          state.trace =
              writeValue(b, state.trace, positionSize + j,
                         arith::SelectOp::create(b, loc, take1, yieldsAt[1][j],
                                                 yieldsAt[0][j]));
  }
  state.trace = writeObserved(b, state.trace);
  return state;
}

Value GibbsSweepEmitter::writeValue(OpBuilder &b, Value trace, int64_t column,
                                    Value value) {
  auto valueType = cast<RankedTensorType>(value.getType());
  auto flatType = RankedTensorType::get({1, valueType.getNumElements()},
                                        traceType.getElementType());
  Value flat = impulse::ReshapeOp::create(b, loc, flatType, value);
  return updateColumns(b, loc, trace, flat, column);
}

Value GibbsSweepEmitter::writeObserved(OpBuilder &b, Value trace) {
  for (const ObservedColumns &columns : observed) {
    Value observedValues = emitColumnSlice(
        b, loc,
        RankedTensorType::get({1, columns.size}, traceType.getElementType()),
        op.getConstraint(), columns.constraintOffset);
    trace = writeValue(b, trace, columns.traceColumn, observedValues);
  }
  return trace;
}

namespace {

/// Lowers an MH `impulse.infer` or `impulse.infer_region` to `num_samples`
/// MH steps, each a proposal for the `selection` and an accept test.
template <typename OpT>
struct LowerMHInferPattern : public mlir::OpRewritePattern<OpT> {
  static constexpr bool isRegionForm =
      std::is_same_v<OpT, impulse::InferRegionOp>;
  using mlir::OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    if (!op.getMhConfig().has_value())
      return rewriter.notifyMatchFailure(op, "no mh_config; not an MH op");

    Location loc = op.getLoc();
    auto inputs = op.getInputs();
    if (inputs.empty()) {
      op.emitError("Impulse MH: requires at least the rng_state input");
      return failure();
    }
    if (op.getStateIn().size() != 1) {
      op.emitError("Impulse MH: expects a single `weight` state_in slot");
      return failure();
    }
    if constexpr (isRegionForm) {
      if (op.getSampling().empty() || op.getLogpdf().empty()) {
        op.emitError("Impulse MH: infer_region requires sampler and "
                     "constructed logpdf regions (run sicm first)");
        return failure();
      }
      unsigned numSites = 0;
      for (Operation &o : op.getSampling().front())
        numSites += isa<impulse::SampleRegionOp>(&o);
      if (op.getLogpdf().front().getTerminator()->getNumOperands() !=
          1 + numSites) {
        op.emitError("Impulse MH: infer_region logpdf region must yield the "
                     "joint density followed by one density per site");
        return failure();
      }
    }

    Value initTrace = op.getOriginalTrace();
    Value initWeight = op.getStateIn()[0];
    Value initRng = inputs[0];
    auto traceType = cast<RankedTensorType>(initTrace.getType());

    ArrayAttr selection = op.getSelectionAttr();
    ArrayAttr allAddresses = op.getAllAddressesAttr();

    SymbolTableCollection symbolTable;
    FunctionOpInterface fn;
    if constexpr (!isRegionForm) {
      fn = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
          op, op.getFnAttr());
      if (!fn || fn.getFunctionBody().empty()) {
        op.emitError("Impulse MH: `fn` must reference a non-empty funcOp");
        return failure();
      }
    }

    int64_t fullPositionSize = 0;
    RegionTraceLayout layout;
    ColumnWindows flatChangedColumns;
    if constexpr (isRegionForm) {
      FailureOr<RegionTraceLayout> regionLayout =
          getRegionTraceLayout(op, /*topLevelOnly=*/true);
      if (failed(regionLayout)) {
        op.emitError("Impulse MH: all_addresses entry without a top-level "
                     "site");
        return failure();
      }
      layout = std::move(*regionLayout);
      for (auto site :
           op.getSampling().front().template getOps<impulse::SampleRegionOp>())
        fullPositionSize += sampleRegionSize(site);
    } else {
      FailureOr<FlatTraceLayout> flatLayout =
          getFlatTraceLayout(op, fn, allAddresses, symbolTable);
      if (failed(flatLayout))
        return failure();
      fullPositionSize = flatLayout->positionSize;
      flatChangedColumns =
          mhChangedColumnsFlat(*flatLayout, selection, traceType.getShape()[1]);
    }
    bool hasScoreTail = traceType.getShape()[1] > fullPositionSize;

    // Carries the trace, the weight, the key and whether the step accepted.
    auto oneStep = [&](ValueRange carried) -> SmallVector<Value> {
      Value trace = carried[0], rng = carried[2];
      MHProposal proposal;
      if constexpr (isRegionForm) {
        proposal = proposeRegionMove(rewriter, loc, op, layout, trace, rng,
                                     hasScoreTail, /*withLogRatio=*/true);
      } else {
        SmallVector<Value> inputs(op.getInputs());
        inputs[0] = rng;
        proposal =
            proposeFlatMove(rewriter, loc, op, inputs, trace,
                            rewriter.getStringAttr(""), fn.getResultTypes());
        proposal.changed = flatChangedColumns;
      }
      MHStep step =
          emitMHStep(rewriter, loc, trace, carried[1], proposal, hasScoreTail);
      return {step.trace, step.weight, step.rng, step.accepted};
    };
    SmallVector<Value> out = emitRepeated(
        rewriter, loc, op.getNumSamples(), {initTrace, initWeight, initRng},
        [&]() -> SmallVector<Value> {
          Value notAccepted = emitI1ScalarConstant(rewriter, loc, false);
          return {initTrace, initWeight, initRng, notAccepted};
        },
        oneStep);

    // MH reports no per-sample log density; `log_densities` is zero.
    Value outLogDensities = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(op.getLogDensities().getType()));
    rewriter.replaceOp(op, {out[0], out[3], outLogDensities, out[2], out[1]});
    return success();
  }
};

/// Lowers a Gibbs `impulse.infer_region` to `num_samples` sweeps of a
/// `GibbsSweepEmitter`.
struct LowerGibbsRegionPattern
    : public mlir::OpRewritePattern<impulse::InferRegionOp> {
  using mlir::OpRewritePattern<impulse::InferRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(impulse::InferRegionOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getGibbsConfig().has_value())
      return rewriter.notifyMatchFailure(op, "no gibbs_config");
    FailureOr<GibbsSweepEmitter> emitter = GibbsSweepEmitter::analyze(op);
    if (failed(emitter))
      return failure();
    Location loc = op.getLoc();
    SmallVector<Value> out = emitRepeated(
        rewriter, loc, op.getNumSamples(),
        {op.getOriginalTrace(), op.getStateIn()[0], op.getInputs()[0]},
        /*loopInit=*/nullptr, [&](ValueRange carried) {
          GibbsState state = emitter->emitSweep(
              rewriter, {carried[0], carried[1], carried[2]});
          return SmallVector<Value>{state.trace, state.weight, state.rng};
        });

    // Gibbs always accepts; `log_densities` is a zero placeholder.
    auto diagType = cast<ShapedType>(op.getDiagnostics().getType());
    Value diagnostics = arith::ConstantOp::create(
        rewriter, loc, diagType,
        DenseElementsAttr::get(diagType, rewriter.getBoolAttr(true)));
    Value logDensities = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(op.getLogDensities().getType()));
    rewriter.replaceOp(op, {out[0], diagnostics, logDensities, out[2], out[1]});
    return success();
  }
};

/// Rejects a Gibbs `impulse.infer`: only the region form has a lowering.
struct RejectFlatGibbsPattern
    : public mlir::OpRewritePattern<impulse::InferOp> {
  using mlir::OpRewritePattern<impulse::InferOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(impulse::InferOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getGibbsConfig().has_value())
      return failure();
    op.emitError("Impulse Gibbs: requires the region pipeline "
                 "(inline-infer-regions, sicm); this model was not admitted "
                 "to region form");
    return failure();
  }
};

} // namespace

void impulse::populateExpandImpulseMHGibbsPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<LowerMHInferPattern<impulse::InferOp>,
               LowerMHInferPattern<impulse::InferRegionOp>>(context);
  patterns.add<LowerGibbsRegionPattern, RejectFlatGibbsPattern>(context);
}
