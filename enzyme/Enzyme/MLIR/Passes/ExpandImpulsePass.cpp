//===- ExpandImpulsePass.cpp - Lower the Impulse inference ops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the expand-impulse pass, which lowers the Impulse
// inference ops into calls, arith and math ops and lower-level Impulse ops. It
// lowers untraced_call, simulate, generate, regenerate and mh to calls of
// clones of the model; the patterns for infer and infer_region are in the
// ExpandImpulse*.cpp files next to it.
//
//===----------------------------------------------------------------------===//

#include "ExpandImpulse.h"

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"
#include "Interfaces/DistBuilder.h"
#include "Interfaces/ImpulseUtils.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/StringSet.h"

#define DEBUG_TYPE "expand-impulse"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::impulse;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_EXPANDIMPULSEPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

/// Returns the function `fnAttr` that `op` calls, or fails after emitting on
/// `op` when its body is empty: a distribution site needs a `logpdf_fn`, or
/// `opName` would recurse into its sampler.
static FailureOr<FunctionOpInterface>
lookupModelBody(Operation *op, SymbolRefAttr fnAttr, StringRef opName,
                SymbolTableCollection &symbolTable) {
  auto fn =
      symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(op, fnAttr);
  if (!fn.getFunctionBody().empty())
    return fn;
  return op->emitError("Impulse: calling `")
         << opName
         << "` on an empty function; if this is a distribution function, its "
            "sample op should have a `logpdf_fn` attribute to avoid recursive `"
         << opName << "` calls which is intended for generative functions";
}

/// Maps the symbol of every sample site of `fn`, nested ones included, to the
/// site.
static SampleOpMap buildSampleOpMap(FunctionOpInterface fn) {
  SampleOpMap map;
  fn.walk([&](impulse::SampleOp sampleOp) {
    if (auto symbol = sampleOp.getSymbolAttr())
      map[symbol] = sampleOp;
  });
  return map;
}

/// Returns the number of trace columns the values of `sampleOp` take, or -1
/// after emitting on `op` when a value is not a statically shaped tensor.
static int64_t computeSampleElementCount(Operation *op,
                                         impulse::SampleOp sampleOp) {
  int64_t totalCount = 0;
  for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
    auto resultType = sampleOp.getResult(i).getType();
    auto tensorType = dyn_cast<RankedTensorType>(resultType);
    if (!tensorType) {
      op->emitError("Expected ranked tensor type for sample result");
      return -1;
    }
    if (!tensorType.hasStaticShape()) {
      op->emitError("Dynamic tensor dimensions not supported");
      return -1;
    }
    totalCount += tensorType.getNumElements();
  }
  return totalCount;
}

/// Adds the number of trace columns the site at `address` takes to
/// `positionSize`, looking each symbol after the first up in the generative
/// function of the site before it. Returns false when a symbol has no site, and
/// after emitting on `op` when the address continues past a site without a
/// non-empty generative function or a value is dynamically shaped.
static bool computePositionSizeForAddress(Operation *op,
                                          const SampleOpMap &sampleMap,
                                          ArrayRef<Attribute> address,
                                          SymbolTableCollection &symbolTable,
                                          int64_t &positionSize) {
  if (address.empty())
    return false;

  impulse::SampleOp sampleOp = sampleMap.lookup(address[0]);
  if (!sampleOp)
    return false;

  if (address.size() > 1) {
    if (sampleOp.getLogpdfFnAttr()) {
      op->emitError("Cannot select nested address in distribution function");
      return false;
    }

    auto genFn = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
        sampleOp, sampleOp.getFnAttr());
    if (genFn.getFunctionBody().empty()) {
      op->emitError("Cannot find generative function for nested address");
      return false;
    }

    auto nestedMap = buildSampleOpMap(genFn);
    return computePositionSizeForAddress(op, nestedMap, address.drop_front(),
                                         symbolTable, positionSize);
  }

  int64_t elemCount = computeSampleElementCount(op, sampleOp);
  if (elemCount < 0)
    return false;

  positionSize += elemCount;
  return true;
}

/// Returns the number of trace columns the sites of `fn` at the addresses in
/// `selection` take, or -1 after emitting on `op` when one of them has none.
static int64_t
computePositionSizeForSelection(Operation *op, FunctionOpInterface fn,
                                ArrayAttr selection,
                                SymbolTableCollection &symbolTable) {
  auto sampleMap = buildSampleOpMap(fn);
  int64_t positionSize = 0;

  for (auto addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (address.empty()) {
      op->emitError("Empty address in selection");
      return -1;
    }

    SmallVector<Attribute> tailAddresses(address.begin(), address.end());
    if (!computePositionSizeForAddress(op, sampleMap, tailAddresses,
                                       symbolTable, positionSize)) {
      op->emitError("Could not find sample with symbol in address chain");
      return -1;
    }
  }

  return positionSize;
}

int64_t impulse::computeOffsetForSampleInSelection(
    Operation *op, FunctionOpInterface fn, ArrayAttr selection,
    Attribute targetSymbol, SymbolTableCollection &symbolTable) {
  auto sampleMap = buildSampleOpMap(fn);
  int64_t offset = 0;

  for (auto addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (address.empty())
      continue;

    auto firstSymbol = address[0];

    if (firstSymbol == targetSymbol) {
      return offset;
    }

    SmallVector<Attribute> tailAddresses(address.begin(), address.end());
    if (!computePositionSizeForAddress(op, sampleMap, tailAddresses,
                                       symbolTable, offset)) {
      return -1;
    }
  }

  return -1;
}

/// Returns the number of trace columns the site at `address` takes, looking
/// the first symbol up in `sites` and each later one in the generative
/// function of the site before it. Fails after emitting on `op` when a symbol
/// has no site, the address continues past a site with a `logpdf_fn` or into
/// an empty function, or a value of the site is not a statically shaped
/// tensor.
static FailureOr<int64_t> getAddressSize(Operation *op,
                                         const SampleOpMap &sites,
                                         ArrayAttr address,
                                         SymbolTableCollection &symbolTable) {
  SampleOpMap nestedSites;
  const SampleOpMap *scope = &sites;
  for (auto [depth, symbol] : llvm::enumerate(address.getValue())) {
    impulse::SampleOp site = scope->lookup(symbol);
    if (!site)
      return op->emitError(
          "Could not find sample with symbol in address chain");
    if (depth + 1 == address.size()) {
      int64_t size = computeSampleElementCount(op, site);
      if (size < 0)
        return failure();
      return size;
    }
    if (site.getLogpdfFnAttr())
      return op->emitError(
          "Cannot select nested address in distribution function");
    auto genFn = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
        site, site.getFnAttr());
    if (genFn.getFunctionBody().empty())
      return op->emitError(
          "Cannot find generative function for nested address");
    nestedSites = buildSampleOpMap(genFn);
    scope = &nestedSites;
  }
  llvm_unreachable("an empty address has no site");
}

FailureOr<FlatTraceLayout>
impulse::getFlatTraceLayout(Operation *op, FunctionOpInterface fn,
                            ArrayAttr addresses,
                            SymbolTableCollection &symbolTable) {
  FlatTraceLayout layout;
  layout.siteBySymbol = buildSampleOpMap(fn);
  for (Attribute attr : addresses) {
    auto address = cast<ArrayAttr>(attr);
    if (address.empty())
      return op->emitError("Empty address in selection");
    FailureOr<int64_t> size =
        getAddressSize(op, layout.siteBySymbol, address, symbolTable);
    if (failed(size))
      return failure();
    layout.entries.push_back({address, layout.positionSize, *size});
    layout.columnOf.try_emplace(address[0], layout.positionSize);
    if (address.size() > 1)
      layout.nestedSizeOf[address[0]] += *size;
    layout.positionSize += *size;
  }
  return layout;
}

int64_t impulse::sampleRegionSize(impulse::SampleRegionOp site) {
  int64_t size = 0;
  for (Value result : llvm::drop_begin(site->getResults()))
    size += cast<RankedTensorType>(result.getType()).getNumElements();
  return size;
}

FailureOr<RegionTraceLayout>
impulse::getRegionTraceLayout(impulse::InferRegionOp regionOp,
                              bool topLevelOnly) {
  RegionTraceLayout layout;
  auto addSite = [&](impulse::SampleRegionOp site) {
    if (Attribute address = site.getAddressAttr())
      layout.siteByAddress[address] = site;
  };
  if (topLevelOnly)
    for (auto site :
         regionOp.getSampling().front().getOps<impulse::SampleRegionOp>())
      addSite(site);
  else
    regionOp.getSampling().walk(addSite);
  for (Attribute address : regionOp.getAllAddressesAttr()) {
    impulse::SampleRegionOp site = layout.siteByAddress.lookup(address);
    if (!site)
      return failure();
    layout.columnOf[address] = layout.positionSize;
    layout.positionSize += sampleRegionSize(site);
  }
  return layout;
}

/// Returns the addresses of `selection` that continue past the symbol
/// `targetSymbol`, without that first symbol.
static ArrayAttr buildSubSelection(Builder &builder, ArrayAttr selection,
                                   Attribute targetSymbol) {
  SmallVector<Attribute> subAddresses;
  for (auto addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (address.empty())
      continue;
    if (address[0] == targetSymbol && address.size() > 1) {
      SmallVector<Attribute> tail(address.begin() + 1, address.end());
      subAddresses.push_back(builder.getArrayAttr(tail));
    }
  }
  return builder.getArrayAttr(subAddresses);
}

namespace {
/// The trace, weight and Hastings correction carried from one sample site to
/// the next while a function body is lowered. `offset` is the next trace
/// column; it advances over both `impulse.if` branches, which take disjoint
/// columns. `hastingsCorrection` is set only while a Hastings regenerate is
/// lowered.
struct SiteCarry {
  Value trace;
  Value weight;
  int64_t offset;
  Value hastingsCorrection = nullptr;
};
} // namespace

/// Lowers one sample site of a function body, given the carry at the site.
using ProcessSampleFn =
    llvm::function_ref<LogicalResult(impulse::SampleOp, SiteCarry &)>;

/// Array of top-level addresses asking an `impulse.regenerate` for a `2 x N`
/// weight: column `k` holds 1 and the log density of the site at address `k`
/// where that site runs, else zeros.
static constexpr llvm::StringLiteral kMHSiteScoresAttr =
    "impulse.mh_site_scores";

Value impulse::scalarAt(OpBuilder &b, Location loc, Value tensor,
                        ArrayRef<int64_t> index) {
  auto type = cast<RankedTensorType>(tensor.getType());
  SmallVector<int64_t> limit, strides(index.size(), 1), ones(index.size(), 1);
  for (int64_t i : index)
    limit.push_back(i + 1);
  Value slice = impulse::SliceOp::create(
      b, loc, RankedTensorType::get(ones, type.getElementType()), tensor,
      b.getDenseI64ArrayAttr(index), b.getDenseI64ArrayAttr(limit),
      b.getDenseI64ArrayAttr(strides));
  return impulse::ReshapeOp::create(
      b, loc, RankedTensorType::get({}, type.getElementType()), slice);
}

/// Lowers the sample sites of `block` in program order with `processSample`,
/// including those nested in the regions of its ops.
static LogicalResult processImpulseBlock(Block &block,
                                         PatternRewriter &rewriter,
                                         SiteCarry &carry,
                                         ProcessSampleFn processSample);

/// Lowers the sample sites of both branches of `ifOp` with `processSample` and
/// replaces `ifOp` with an `impulse.if` that also yields the trace, weight and
/// Hastings correction of the branch taken, which become `carry`. Both
/// branches start from the trace and weight of `carry`, so the columns of the
/// sites of the branch not taken keep their values from before `ifOp`.
static LogicalResult processImpulseIf(impulse::IfOp ifOp,
                                      PatternRewriter &rewriter,
                                      SiteCarry &carry,
                                      ProcessSampleFn processSample) {
  Location loc = ifOp.getLoc();
  SiteCarry before = carry;

  Block &trueBlock = ifOp.getTrueBranch().front();
  if (failed(processImpulseBlock(trueBlock, rewriter, carry, processSample)))
    return failure();
  SiteCarry afterTrue = carry;

  Block &falseBlock = ifOp.getFalseBranch().front();
  carry.trace = before.trace;
  carry.weight = before.weight;
  carry.hastingsCorrection = before.hastingsCorrection;
  if (failed(processImpulseBlock(falseBlock, rewriter, carry, processSample)))
    return failure();

  auto extendYield = [&](Block &block, const SiteCarry &branch) {
    auto yield = cast<impulse::YieldOp>(block.getTerminator());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(yield);
    SmallVector<Value> operands(yield.getOperands().begin(),
                                yield.getOperands().end());
    operands.push_back(branch.trace);
    operands.push_back(branch.weight);
    if (branch.hastingsCorrection)
      operands.push_back(branch.hastingsCorrection);
    impulse::YieldOp::create(rewriter, yield.getLoc(), operands);
    rewriter.eraseOp(yield);
  };
  extendYield(trueBlock, afterTrue);
  extendYield(falseBlock, carry);

  SmallVector<Type> newResultTypes(ifOp.getResultTypes().begin(),
                                   ifOp.getResultTypes().end());
  unsigned oldNumResults = ifOp.getNumResults();
  newResultTypes.push_back(before.trace.getType());
  // Not always scalar: an MH site-scores regenerate carries a vector.
  newResultTypes.push_back(before.weight.getType());
  if (before.hastingsCorrection)
    newResultTypes.push_back(before.hastingsCorrection.getType());

  rewriter.setInsertionPoint(ifOp);
  auto newIf =
      impulse::IfOp::create(rewriter, loc, newResultTypes, ifOp.getPredicate());
  rewriter.inlineRegionBefore(ifOp.getTrueBranch(), newIf.getTrueBranch(),
                              newIf.getTrueBranch().end());
  rewriter.inlineRegionBefore(ifOp.getFalseBranch(), newIf.getFalseBranch(),
                              newIf.getFalseBranch().end());

  carry.trace = newIf.getResult(oldNumResults);
  carry.weight = newIf.getResult(oldNumResults + 1);
  if (before.hastingsCorrection)
    carry.hastingsCorrection = newIf.getResult(oldNumResults + 2);

  rewriter.replaceOp(ifOp, newIf.getResults().take_front(oldNumResults));
  return success();
}

static LogicalResult processImpulseBlock(Block &block,
                                         PatternRewriter &rewriter,
                                         SiteCarry &carry,
                                         ProcessSampleFn processSample) {
  SmallVector<Operation *> ops;
  for (Operation &op : block)
    ops.push_back(&op);

  for (Operation *op : ops) {
    if (auto sampleOp = dyn_cast<impulse::SampleOp>(op)) {
      if (failed(processSample(sampleOp, carry)))
        return failure();
    } else if (auto ifOp = dyn_cast<impulse::IfOp>(op)) {
      bool hasSample = false;
      ifOp->walk([&](impulse::SampleOp) { hasSample = true; });
      if (hasSample &&
          failed(processImpulseIf(ifOp, rewriter, carry, processSample)))
        return failure();
    } else {
      for (Region &region : op->getRegions())
        for (Block &nested : region.getBlocks())
          if (failed(
                  processImpulseBlock(nested, rewriter, carry, processSample)))
            return failure();
    }
  }
  return success();
}

Value impulse::updateColumns(OpBuilder &builder, Location loc, Value tensor,
                             Value update, int64_t column) {
  Value row = emitI64ScalarConstant(builder, loc, 0);
  Value col = emitI64ScalarConstant(builder, loc, column);
  return impulse::DynamicUpdateSliceOp::create(
      builder, loc, tensor.getType(), tensor, update, ValueRange{row, col});
}

Value impulse::writeTraceColumns(OpBuilder &builder, Location loc, Value trace,
                                 Value value, int64_t column) {
  auto valueType = cast<RankedTensorType>(value.getType());
  auto rowType = RankedTensorType::get({1, valueType.getNumElements()},
                                       valueType.getElementType());
  Value row = impulse::ReshapeOp::create(builder, loc, rowType, value);
  return updateColumns(builder, loc, trace, row, column);
}

/// Writes the `values` of `site` into `trace` one after another from column
/// `offset` on, advancing `offset` past them. With `write` unset only `offset`
/// advances. Fails on a dynamically shaped value.
static LogicalResult appendSiteValues(OpBuilder &builder,
                                      impulse::SampleOp site,
                                      ArrayRef<Value> values, Value &trace,
                                      int64_t &offset, bool write = true) {
  for (Value value : values) {
    auto type = cast<RankedTensorType>(value.getType());
    if (!type.hasStaticShape())
      return site.emitError("Impulse: dynamic tensor dimensions not supported");
    if (write)
      trace = writeTraceColumns(builder, site.getLoc(), trace, value, offset);
    offset += type.getNumElements();
  }
  return success();
}

/// Returns the parameters of `site`: its inputs after the rng.
static ValueRange siteParams(impulse::SampleOp site) {
  ValueRange inputs = site.getInputs();
  return inputs.empty() ? inputs : inputs.drop_front();
}

/// Draws fresh values for `site`: inline from its distribution when that has a
/// loop-free sampler, else by calling its `fn`. Returns replacements for all
/// results of `site`, the rng first.
static SmallVector<Value> emitSiteDraw(OpBuilder &builder,
                                       SymbolTableCollection &symbolTable,
                                       impulse::SampleOp site) {
  Location loc = site.getLoc();
  if (impulse::DistAttr dist = site.getDistAttr();
      dist && dist.hasLoopFreeSampler()) {
    Value rng = site.getInputs().front();
    Value sample = impulse::emitDistSampler(
        builder, loc, dist, rng, siteParams(site),
        cast<RankedTensorType>(site.getResult(1).getType()));
    return {rng, sample};
  }
  auto fn = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      site, site.getFnAttr());
  auto call = func::CallOp::create(builder, loc, fn.getName(),
                                   fn.getResultTypes(), site.getInputs());
  return SmallVector<Value>(call.getResults());
}

/// Returns the `logpdf_fn` function of `site`, or null for a site without
/// one, after checking that it takes the `numValues` values of the site
/// followed by its parameters. `call` names the call in the diagnostic.
static FailureOr<FunctionOpInterface>
verifySiteLogpdf(SymbolTableCollection &symbolTable, impulse::SampleOp site,
                 size_t numValues, StringRef call = "logpdf call") {
  if (!site.getLogpdfFnAttr())
    return FunctionOpInterface();
  auto logpdfFn = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      site, site.getLogpdfFnAttr());
  if (numValues + siteParams(site).size() == logpdfFn.getNumArguments())
    return logpdfFn;
  return site.emitError("Impulse: failed to construct ")
         << call << "; logpdf function has wrong number of arguments";
}

/// Emits the log density of `values` (the site's results after the rng) under
/// `site`: from its distribution, or by calling `logpdfFn`, the function
/// `verifySiteLogpdf` returned, on the values followed by the site's
/// parameters.
static Value emitSiteLogpdf(OpBuilder &builder, impulse::SampleOp site,
                            FunctionOpInterface logpdfFn, ValueRange values) {
  Location loc = site.getLoc();
  if (impulse::DistAttr dist = site.getDistAttr())
    return impulse::emitDistLogpdf(builder, loc, dist, values.front(),
                                   siteParams(site));
  SmallVector<Value> operands(values);
  llvm::append_range(operands, siteParams(site));
  return func::CallOp::create(builder, loc, logpdfFn.getName(),
                              logpdfFn.getResultTypes(), operands)
      .getResult(0);
}

SmallVector<Value> impulse::sliceTraceValues(OpBuilder &builder, Location loc,
                                             Value trace, int64_t offset,
                                             TypeRange types) {
  SmallVector<Value> values;
  for (Type type : types) {
    auto tensorType = cast<RankedTensorType>(type);
    values.push_back(emitRowSlice(builder, loc, trace, offset, tensorType));
    offset += tensorType.getNumElements();
  }
  return values;
}

/// Returns replacements for the results of `site`: its rng input, then its
/// values read from row 0 of `source` starting at column `offset`. Fails on a
/// dynamically shaped value.
static FailureOr<SmallVector<Value>> readSiteValues(OpBuilder &builder,
                                                    impulse::SampleOp site,
                                                    Value source,
                                                    int64_t offset) {
  SmallVector<Value> values{site.getInputs().front()};
  for (Value result : llvm::drop_begin(site->getResults())) {
    auto type = cast<RankedTensorType>(result.getType());
    if (!type.hasStaticShape()) {
      site.emitError("Impulse: dynamic tensor dimensions not supported");
      return failure();
    }
    values.push_back(
        emitRowSlice(builder, site.getLoc(), source, offset, type));
    offset += type.getNumElements();
  }
  return values;
}

Value impulse::writeScoreCells(OpBuilder &builder, Location loc, Value trace,
                               Value row0,
                               ArrayRef<std::pair<int64_t, Value>> cells) {
  auto traceType = cast<RankedTensorType>(trace.getType());
  auto cellType = RankedTensorType::get({1, 1}, traceType.getElementType());
  for (auto [column, score] : cells) {
    Value cell = impulse::ReshapeOp::create(builder, loc, cellType, score);
    Value columnIndex = emitI64ScalarConstant(builder, loc, column);
    trace = impulse::DynamicUpdateSliceOp::create(
        builder, loc, traceType, trace, cell, ValueRange{row0, columnIndex});
  }
  return trace;
}

/// Returns a `1 x width` trace whose leading columns are `values` and whose
/// following columns hold `siteScores`, one score per column.
static Value appendScoreTail(OpBuilder &builder, Location loc, Value values,
                             ArrayRef<Value> siteScores, int64_t width) {
  auto fullType = RankedTensorType::get({1, width}, builder.getF64Type());
  Value zeroFull = emitSplatConstant(builder, loc, fullType, 0.0);
  Value c0 = emitI64ScalarConstant(builder, loc, 0);
  Value trace = impulse::DynamicUpdateSliceOp::create(
      builder, loc, fullType, zeroFull, values, ValueRange{c0, c0});
  int64_t firstScoreColumn =
      cast<RankedTensorType>(values.getType()).getShape()[1];
  SmallVector<std::pair<int64_t, Value>> cells;
  for (auto [i, score] : llvm::enumerate(siteScores))
    cells.push_back({firstScoreColumn + (int64_t)i, score});
  return writeScoreCells(builder, loc, trace, c0, cells);
}

/// Packs `weight` and the Hastings correction `correction` into the weight
/// vector of a Hastings regenerate.
static Value packHastingsWeight(OpBuilder &b, Location loc, Value weight,
                                Value correction) {
  auto vecType = RankedTensorType::get({2}, b.getF64Type());
  auto oneType = RankedTensorType::get({1}, b.getF64Type());
  Value acc = emitSplatConstant(b, loc, vecType, 0.0);
  // Named array: a ValueRange over a braced list would dangle in the loop.
  Value parts[] = {weight, correction};
  for (auto [i, v] : llvm::enumerate(parts)) {
    Value idx = emitI64ScalarConstant(b, loc, i);
    Value flat = impulse::ReshapeOp::create(b, loc, oneType, v);
    acc = impulse::DynamicUpdateSliceOp::create(b, loc, vecType, acc, flat,
                                                ValueRange{idx});
  }
  return acc;
}

std::pair<Value, Value>
impulse::unpackHastingsWeight(OpBuilder &b, Location loc, Value packed) {
  return {scalarAt(b, loc, packed, {0}), scalarAt(b, loc, packed, {1})};
}

bool impulse::usesAnyOf(Operation *op, const DenseSet<Value> &values) {
  auto inValues = [&](Value v) { return values.contains(v); };
  if (llvm::any_of(op->getOperands(), inValues))
    return true;
  if (op->getNumRegions() == 0)
    return false;
  return op
      ->walk([&](Operation *inner) {
        return llvm::any_of(inner->getOperands(), inValues)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

/// Returns whether an address in `addresses` starts with the symbol of
/// `sampleOp`.
static bool addressInList(impulse::SampleOp sampleOp, ArrayAttr addresses) {
  for (auto addr : addresses) {
    auto address = cast<ArrayAttr>(addr);
    if (!address.empty() && address[0] == sampleOp.getSymbolAttr())
      return true;
  }
  return false;
}

/// Fails with a diagnostic naming `role` when the first address in
/// `addresses` that starts with the symbol of `sampleOp` is nested.
static LogicalResult verifyTopLevelAddress(impulse::SampleOp sampleOp,
                                           ArrayAttr addresses,
                                           StringRef role) {
  for (Attribute attr : addresses) {
    auto address = cast<ArrayAttr>(attr);
    if (address.empty() || address[0] != sampleOp.getSymbolAttr())
      continue;
    if (address.size() == 1)
      return success();
    return sampleOp.emitError(
               "Impulse: distribution function cannot have composite ")
           << role << " address";
  }
  return success();
}

/// Returns the values in `fn` that regenerating `regenerateAddresses` can
/// change. Taint stops at a site that is not regenerated, whose value comes
/// from the trace; an op with regions is tainted when anything nested in it
/// reads a tainted value.
static DenseSet<Value> computeRegenerateTaint(FunctionOpInterface fn,
                                              ArrayAttr regenerateAddresses) {
  DenseSet<Value> tainted;
  bool changed = true;
  while (changed) {
    changed = false;
    auto taint = [&](auto values) {
      for (Value v : values)
        changed |= tainted.insert(v).second;
    };
    fn.getFunctionBody().walk([&](Operation *op) {
      if (auto sampleOp = dyn_cast<impulse::SampleOp>(op)) {
        if (addressInList(sampleOp, regenerateAddresses))
          taint(llvm::drop_begin(op->getResults()));
        return;
      }
      if (!usesAnyOf(op, tainted))
        return;
      taint(op->getResults());
      for (Region &region : op->getRegions())
        for (Block &block : region)
          taint(block.getArguments());
    });
  }
  return tainted;
}

/// Returns whether every site of `fn` runs exactly once, as a score tail with
/// one column per site requires: no `impulse.if`, and only top-level
/// distribution sites.
static bool supportsScoreTail(FunctionOpInterface fn) {
  bool feasible = true;
  fn.getFunctionBody().walk([&](Operation *op) {
    if (isa<impulse::IfOp>(op))
      feasible = false;
    if (auto s = dyn_cast<impulse::SampleOp>(op))
      if ((!s.getDistAttr() && !s.getLogpdfFnAttr()) ||
          s->getParentOp() != fn.getOperation())
        feasible = false;
  });
  return feasible;
}

//===----------------------------------------------------------------------===//
// Traced lowerings of simulate, generate and regenerate
//===----------------------------------------------------------------------===//

namespace {
/// A clone of a model that simulate, generate or regenerate rewrites site by
/// site into a function that also returns the trace of `allAddresses` and the
/// weight.
struct TracedClone {
  FunctionOpInterface fn;
  ArrayAttr allAddresses;
  /// Where the sites at `allAddresses` sit on the trace.
  FlatTraceLayout layout;
  FunctionOpInterface newFn;
  /// The op the body of `newFn` started with; the lowering's constants go
  /// before it.
  Operation *bodyStart = nullptr;
  /// Width of the returned trace, wider than the positions with a score tail.
  int64_t declaredWidth = 0;
  /// The zero weight and the zero position trace at the start of `newFn`.
  Value zeroWeight;
  Value zeroTrace;
  /// What the sites carry, from the zero trace and weight on.
  SiteCarry carry;
  /// The log density of each site in program order, for the score tail.
  SmallVector<Value> siteScores;

  int64_t positionSize() const { return layout.positionSize; }
  bool hasScoreTail() const { return declaredWidth > layout.positionSize; }
};

/// A sub-model site of a traced lowering whose nested trace holds the
/// addresses below it.
struct NestedSite {
  FunctionOpInterface genFn;
  /// The addresses of `all_addresses` below the site, without its symbol.
  ArrayAttr subAllAddresses;
  /// Width of the nested trace and its first column in the caller's trace.
  int64_t subPositionSize = 0;
  int64_t mergeOffset = 0;

  RankedTensorType getTraceType() {
    return RankedTensorType::get({1, subPositionSize},
                                 Float64Type::get(genFn.getContext()));
  }
  /// The result types of the nested op: its trace, a weight of `weightType`
  /// and the results of `genFn`.
  SmallVector<Type> getResultTypes(Type weightType) {
    SmallVector<Type> types{getTraceType(), weightType};
    llvm::append_range(types, genFn.getResultTypes());
    return types;
  }
};
} // namespace

/// Lowers one sample site of a traced clone given the carry at the site, at
/// an insertion point before the site, and returns the replacements for the
/// site's results.
using TracedSiteFn =
    function_ref<FailureOr<SmallVector<Value>>(impulse::SampleOp, SiteCarry &)>;

/// Looks up the model `op` calls, lays out its `allAddresses` and clones it
/// for `mode` with a `declaredWidth`-wide trace and a `constraintSize`-wide
/// constraint (see `cloneForMode`), then creates the zero weight and position
/// trace at the start of the clone. `opName` names `op` in the diagnostics.
/// Fails after emitting on `op` when the model is empty or the addresses take
/// no column.
static FailureOr<TracedClone>
beginTracedClone(Operation *op, SymbolRefAttr fnAttr, ArrayAttr allAddresses,
                 int64_t declaredWidth, ImpulseMode mode,
                 int64_t constraintSize, StringRef opName,
                 SymbolTableCollection &symbolTable) {
  FailureOr<FunctionOpInterface> fn =
      lookupModelBody(op, fnAttr, opName, symbolTable);
  if (failed(fn))
    return failure();
  FailureOr<FlatTraceLayout> layout =
      getFlatTraceLayout(op, *fn, allAddresses, symbolTable);
  if (failed(layout))
    return failure();
  if (layout->positionSize <= 0)
    return op->emitError("Impulse: failed to compute position size for ")
           << opName;

  TracedClone clone;
  clone.fn = *fn;
  clone.allAddresses = allAddresses;
  clone.layout = std::move(*layout);
  clone.declaredWidth = declaredWidth;
  clone.newFn = cloneForMode(*fn, mode, declaredWidth, constraintSize);
  clone.bodyStart = &clone.newFn.getFunctionBody().front().front();
  OpBuilder entryBuilder(clone.bodyStart);
  Location loc = clone.bodyStart->getLoc();
  Type f64 = entryBuilder.getF64Type();
  clone.zeroWeight =
      emitSplatConstant(entryBuilder, loc, RankedTensorType::get({}, f64), 0.0);
  clone.zeroTrace = emitSplatConstant(
      entryBuilder, loc, RankedTensorType::get({1, clone.positionSize()}, f64),
      0.0);
  clone.carry = {clone.zeroTrace, clone.zeroWeight, /*offset=*/0};
  return clone;
}

/// Resolves the sub-model site `site` of `clone`: its generative function and
/// the addresses below it, with `subAllAddresses` empty when there are none.
/// Fails after emitting on `site` when the clone has a score tail or the
/// generative function is empty.
static FailureOr<NestedSite>
resolveNestedSite(Builder &builder, impulse::SampleOp site,
                  const TracedClone &clone,
                  SymbolTableCollection &symbolTable) {
  if (clone.hasScoreTail())
    return site.emitError("Impulse: score-tail traces do not support "
                          "nested generative sites");
  auto genFn = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      site, site.getFnAttr());
  if (genFn.getFunctionBody().empty())
    return site.emitError(
        "Impulse: generative function body is empty; "
        "if this is a distribution, add a `logpdf_fn` attribute");
  Attribute symbol = site.getSymbolAttr();
  NestedSite nested;
  nested.genFn = genFn;
  nested.subAllAddresses =
      buildSubSelection(builder, clone.allAddresses, symbol);
  nested.subPositionSize = clone.layout.nestedSizeOf.lookup(symbol);
  nested.mergeOffset = clone.layout.firstColumn(symbol);
  return nested;
}

/// Adds the log density of `values` under `site` to the weight of `carry` and
/// to the site scores of `clone`, and writes `values` into the trace of `carry`
/// when `site` is at an address of `clone`. `call` names the logpdf call in
/// the diagnostic.
static LogicalResult scoreAndTraceSite(PatternRewriter &rewriter,
                                       impulse::SampleOp site,
                                       ArrayRef<Value> values,
                                       TracedClone &clone, SiteCarry &carry,
                                       SymbolTableCollection &symbolTable,
                                       StringRef call = "logpdf call") {
  FailureOr<FunctionOpInterface> logpdfFn =
      verifySiteLogpdf(symbolTable, site, values.size(), call);
  if (failed(logpdfFn))
    return failure();
  Value logpdf = emitSiteLogpdf(rewriter, site, *logpdfFn, values);
  carry.weight =
      arith::AddFOp::create(rewriter, site.getLoc(), carry.weight, logpdf);
  if (clone.hasScoreTail())
    clone.siteScores.push_back(logpdf);
  if (!addressInList(site, clone.allAddresses))
    return success();
  return appendSiteValues(rewriter, site, values, carry.trace, carry.offset);
}

/// Writes `subTrace`, the trace of `nested`, into the trace `carry` holds and
/// advances its offset past it.
static void mergeNestedTrace(OpBuilder &builder, Location loc,
                             const NestedSite &nested, Value subTrace,
                             SiteCarry &carry) {
  carry.trace =
      updateColumns(builder, loc, carry.trace, subTrace, nested.mergeOffset);
  carry.offset =
      std::max(carry.offset, nested.mergeOffset + nested.subPositionSize);
}

/// Lowers the sample sites of `clone` with `lowerSite` and erases them, then
/// rewrites each return of the clone to also return the trace, with the score
/// tail when the clone has one, and the weight `weightOut` emits at the
/// return, before the model's results. A weight whose type differs from the
/// declared one retypes the clone's results. Replaces `op` by a call of the
/// clone on `callOperands`. Fails after emitting on `op` when a site fails.
static LogicalResult finishTracedClone(PatternRewriter &rewriter, Operation *op,
                                       TracedClone &clone,
                                       TracedSiteFn lowerSite,
                                       function_ref<Value(Location)> weightOut,
                                       ValueRange callOperands) {
  FunctionOpInterface newFn = clone.newFn;
  SmallVector<Operation *> loweredSites;
  auto lowerAndReplace = [&](impulse::SampleOp site,
                             SiteCarry &carry) -> LogicalResult {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(site);
    FailureOr<SmallVector<Value>> replacements = lowerSite(site, carry);
    if (failed(replacements))
      return failure();
    rewriter.replaceAllUsesWith(site->getResults(), *replacements);
    loweredSites.push_back(site);
    return success();
  };
  LogicalResult lowered = processImpulseBlock(
      newFn.getFunctionBody().front(), rewriter, clone.carry, lowerAndReplace);
  for (Operation *site : loweredSites)
    rewriter.eraseOp(site);
  if (failed(lowered))
    return op->emitError("Impulse: failed to walk sample ops");

  SmallVector<Type> returnTypes;
  newFn.walk([&](func::ReturnOp retOp) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(retOp);
    Location loc = retOp.getLoc();
    SmallVector<Value> results{
        clone.hasScoreTail()
            ? appendScoreTail(rewriter, loc, clone.carry.trace,
                              clone.siteScores, clone.declaredWidth)
            : clone.carry.trace};
    results.push_back(weightOut(loc));
    llvm::append_range(results, retOp.getOperands());
    auto newReturn = func::ReturnOp::create(rewriter, loc, results);
    rewriter.eraseOp(retOp);
    returnTypes = llvm::to_vector(newReturn.getOperandTypes());
  });
  if (returnTypes[1] != newFn.getResultTypes()[1])
    newFn.setType(FunctionType::get(op->getContext(), newFn.getArgumentTypes(),
                                    returnTypes));

  rewriter.setInsertionPoint(op);
  auto call = func::CallOp::create(rewriter, op->getLoc(), newFn.getName(),
                                   newFn.getResultTypes(), callOperands);
  rewriter.replaceOp(op, call.getResults());
  return success();
}

namespace {
/// Lowers the sites of a clone for one `impulse.regenerate`: selected sites
/// are drawn afresh, the others read from the old trace or the constraint.
/// The weight is the joint log density, or on a score-tail trace the change
/// of the terms the move can change. `kMHHastingsAttr` adds the Hastings
/// correction of the move to it, and `kMHSiteScoresAttr` replaces it with the
/// liveness and log density of the listed sites.
class RegenerateLowering {
public:
  RegenerateLowering(PatternRewriter &rewriter, impulse::RegenerateOp op,
                     TracedClone &clone, SymbolTableCollection &symbolTable);

  /// Checks a score-tail model, computes what the move can change and emits
  /// the starting trace and weight of the clone, and for a Hastings move the
  /// scores of the old sites it cannot evaluate inline. Fails after emitting
  /// on the op.
  LogicalResult prepare();
  /// Lowers one site of the clone; see `TracedSiteFn`.
  FailureOr<SmallVector<Value>> lowerSite(impulse::SampleOp site,
                                          SiteCarry &carry);
  /// Emits the weight the clone returns at `loc`.
  Value emitWeightOut(Location loc);

private:
  /// A distribution site being lowered.
  struct DistSite {
    impulse::SampleOp op;
    /// Its first column on the old trace, or -1 when it has none.
    int64_t column;
    bool isSelected;
    /// Its column of `oldSiteScores` when its liveness can flip, else -1.
    int64_t flipIdx;
    /// Whether it ran before the move, set when its liveness can flip.
    Value oldLive;
  };

  /// Emits the regenerate that scores the old sites of a Hastings move whose
  /// density cannot be evaluated inline at the new state.
  void emitOldSiteScores();
  FailureOr<SmallVector<Value>> lowerDistSite(impulse::SampleOp op,
                                              SiteCarry &carry);
  FailureOr<SmallVector<Value>> lowerCallSite(impulse::SampleOp op,
                                              SiteCarry &carry);
  /// Returns the values of the site that is not regenerated: read from the
  /// old trace or the constraint, and for a site whose liveness can flip,
  /// redrawn unless it ran before.
  FailureOr<SmallVector<Value>> readOrRedrawValues(DistSite &site);
  /// Adds the term of `site` at `values` to the weight and the Hastings
  /// correction of `carry`, and records its score for the score tail.
  /// `termDirty` says whether the term can change under the move.
  void accumulateSiteWeight(const DistSite &site, FunctionOpInterface logpdfFn,
                            ArrayRef<Value> values, bool termDirty,
                            SiteCarry &carry);
  /// The `impulse.if` ops around `site` in the clone, innermost first.
  SmallVector<impulse::IfOp> enclosingIfs(Operation *site);
  /// Whether `site` sits in a branch whose predicate the move can change.
  bool inFlippableBranch(Operation *site);
  /// Whether `site` reads its value from the constraint when constrained. In
  /// MH, constraint entries outside branches may be latent initial values.
  bool readsConstraint(impulse::SampleOp site);
  /// The `oldSiteScores` column of a latent whose liveness can flip: kept if
  /// it ran before the move, else redrawn. -1 for other sites.
  int64_t flipLatentIndex(impulse::SampleOp site, bool isSelected);
  /// Emits whether the site at `oldSiteScores` column `k` ran before the move.
  Value emitOldLive(Location loc, int64_t k);
  /// Records `score` as the log density of the site at column `k` of the site
  /// scores `acc`, and marks the site live.
  Value recordSiteScore(Location loc, Value acc, int64_t k, Value score);

  PatternRewriter &rewriter;
  impulse::RegenerateOp op;
  TracedClone &clone;
  SymbolTableCollection &symbolTable;
  ArrayAttr selection;
  ArrayAttr constrainedAddresses;
  /// The old trace and the constraint, arguments of the clone.
  Value prevTrace;
  Value constraint;
  bool hastings;
  bool scoring;
  bool hasScoreTail;
  /// Values the move can change, for the terms of a score-tail weight.
  DenseSet<Value> scoreTailTaint;
  /// Values a Hastings move can change, for the liveness of branches.
  DenseSet<Value> hastingsTaint;
  /// Site symbol -> column of the site scores weight or of `oldSiteScores`.
  DenseMap<Attribute, int64_t> siteIndex;
  /// `[liveness; old log density]` of the sites a Hastings move cannot score
  /// inline, one column per site.
  Value oldSiteScores;
  /// Program-order index of the next distribution site, its score column.
  int64_t siteOrdinal = 0;
};
} // namespace

RegenerateLowering::RegenerateLowering(PatternRewriter &rewriter,
                                       impulse::RegenerateOp op,
                                       TracedClone &clone,
                                       SymbolTableCollection &symbolTable)
    : rewriter(rewriter), op(op), clone(clone), symbolTable(symbolTable),
      selection(op.getSelectionAttr()),
      constrainedAddresses(op.getConstrainedAddressesAttr()),
      prevTrace(clone.newFn.getArgument(0)),
      constraint(op.getConstraint() ? clone.newFn.getArgument(1) : Value()),
      hastings(op->hasAttr(kMHHastingsAttr)),
      scoring(
          static_cast<bool>(op->getAttrOfType<ArrayAttr>(kMHSiteScoresAttr))),
      hasScoreTail(clone.hasScoreTail()) {}

LogicalResult RegenerateLowering::prepare() {
  OpBuilder entryBuilder(clone.bodyStart);
  Location entryLoc = clone.bodyStart->getLoc();
  Value initTrace = clone.zeroTrace;
  Value initWeight = clone.zeroWeight;
  // On a score-tail trace the weight is the sum over dirty terms (regenerated
  // or dependent on one) of new minus old log density; clean terms cancel in
  // the MH ratio. Taint is computed up front; fresh draws are added as they
  // are created.
  if (hasScoreTail) {
    if (!supportsScoreTail(clone.newFn))
      return op.emitError("Impulse: regenerate over a score-tail trace "
                          "requires a straight-line body of top-level "
                          "distribution sites");
    scoreTailTaint = computeRegenerateTaint(clone.newFn, selection);
    // Non-regenerated values are unchanged: start from the previous trace.
    initTrace = emitColumnSlice(
        entryBuilder, entryLoc,
        cast<RankedTensorType>(clone.zeroTrace.getType()), prevTrace,
        /*start=*/0);
  }
  if (scoring) {
    auto siteScoresAttr = op->getAttrOfType<ArrayAttr>(kMHSiteScoresAttr);
    for (auto [k, addr] : llvm::enumerate(siteScoresAttr))
      siteIndex[cast<ArrayAttr>(addr)[0]] = k;
    auto scoresType = RankedTensorType::get({2, (int64_t)siteScoresAttr.size()},
                                            entryBuilder.getF64Type());
    initWeight = emitSplatConstant(entryBuilder, entryLoc, scoresType, 0.0);
  } else if (hastings && !hasScoreTail) {
    emitOldSiteScores();
  }
  clone.carry = {initTrace, initWeight, /*offset=*/0,
                 hastings ? clone.zeroWeight : Value()};
  return success();
}

void RegenerateLowering::emitOldSiteScores() {
  hastingsTaint = computeRegenerateTaint(clone.newFn, selection);
  SmallVector<Attribute> oldScoreAddrs;
  clone.newFn.getFunctionBody().walk([&](impulse::SampleOp s) {
    if (!s.getDistAttr() && !s.getLogpdfFnAttr())
      return;
    bool needsOld;
    if (addressInList(s, selection))
      needsOld = !enclosingIfs(s).empty() ||
                 llvm::any_of(llvm::drop_begin(s.getInputs()), [&](Value v) {
                   return hastingsTaint.contains(v);
                 });
    else
      needsOld = !(constraint && constrainedAddresses &&
                   addressInList(s, constrainedAddresses)) &&
                 inFlippableBranch(s);
    if (!needsOld)
      return;
    siteIndex[s.getSymbolAttr()] = oldScoreAddrs.size();
    oldScoreAddrs.push_back(rewriter.getArrayAttr({s.getSymbolAttr()}));
  });
  if (oldScoreAddrs.empty())
    return;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&clone.newFn.getFunctionBody().front());
  auto scoresType = RankedTensorType::get({2, (int64_t)oldScoreAddrs.size()},
                                          rewriter.getF64Type());
  SmallVector<Type> scoreTypes{prevTrace.getType(), scoresType};
  llvm::append_range(scoreTypes, clone.fn.getResultTypes());
  SmallVector<Value> scoreInputs(
      clone.newFn.getArguments().drop_front(constraint ? 2 : 1));
  auto scoreOp = impulse::RegenerateOp::create(
      rewriter, op.getLoc(), scoreTypes, op.getFnAttr(), scoreInputs, prevTrace,
      constraint, clone.allAddresses, rewriter.getArrayAttr({}),
      constraint ? constrainedAddresses : ArrayAttr(),
      rewriter.getStringAttr(""));
  scoreOp->setAttr(kMHSiteScoresAttr, rewriter.getArrayAttr(oldScoreAddrs));
  oldSiteScores = scoreOp.getWeight();
}

FailureOr<SmallVector<Value>>
RegenerateLowering::lowerSite(impulse::SampleOp site, SiteCarry &carry) {
  if (site.getDistAttr() || site.getLogpdfFnAttr())
    return lowerDistSite(site, carry);
  return lowerCallSite(site, carry);
}

FailureOr<SmallVector<Value>>
RegenerateLowering::lowerDistSite(impulse::SampleOp op, SiteCarry &carry) {
  if (failed(verifyTopLevelAddress(op, selection, "selected")))
    return failure();
  DistSite site{op, clone.layout.firstColumn(op.getSymbolAttr()),
                addressInList(op, selection), /*flipIdx=*/-1, Value()};
  site.flipIdx = flipLatentIndex(op, site.isSelected);
  FailureOr<SmallVector<Value>> sampledValues =
      site.isSelected ? emitSiteDraw(rewriter, symbolTable, op)
                      : readOrRedrawValues(site);
  if (failed(sampledValues))
    return failure();
  ArrayRef<Value> values = ArrayRef<Value>(*sampledValues).drop_front();
  if (hastings && !hasScoreTail && (site.isSelected || site.flipIdx >= 0))
    hastingsTaint.insert(values.begin(), values.end());
  FailureOr<FunctionOpInterface> logpdfFn =
      verifySiteLogpdf(symbolTable, op, values.size());
  if (failed(logpdfFn))
    return failure();

  bool termDirty = site.isSelected ||
                   (hasScoreTail && llvm::any_of(siteParams(op), [&](Value v) {
                      return scoreTailTaint.contains(v);
                    }));
  if (hasScoreTail && site.isSelected)
    scoreTailTaint.insert(values.begin(), values.end());
  accumulateSiteWeight(site, *logpdfFn, values, termDirty, carry);
  ++siteOrdinal;

  if (addressInList(op, clone.allAddresses) &&
      failed(appendSiteValues(rewriter, op, values, carry.trace, carry.offset,
                              /*write=*/!hasScoreTail || site.isSelected)))
    return failure();
  return sampledValues;
}

FailureOr<SmallVector<Value>>
RegenerateLowering::readOrRedrawValues(DistSite &site) {
  impulse::SampleOp op = site.op;
  bool isConstrained = false;
  if (readsConstraint(op)) {
    if (failed(verifyTopLevelAddress(op, constrainedAddresses, "constrained")))
      return failure();
    isConstrained = addressInList(op, constrainedAddresses);
  }
  int64_t readOffset = isConstrained
                           ? computeOffsetForSampleInSelection(
                                 this->op, clone.fn, constrainedAddresses,
                                 op.getSymbolAttr(), symbolTable)
                           : site.column;
  FailureOr<SmallVector<Value>> values = readSiteValues(
      rewriter, op, isConstrained ? constraint : prevTrace, readOffset);
  if (failed(values) || site.flipIdx < 0)
    return values;
  Location loc = op.getLoc();
  site.oldLive = emitOldLive(loc, site.flipIdx);
  SmallVector<Value> drawn = emitSiteDraw(rewriter, symbolTable, op);
  for (unsigned i = 1; i < values->size(); ++i)
    (*values)[i] =
        impulse::SelectOp::create(rewriter, loc, (*values)[i].getType(),
                                  site.oldLive, (*values)[i], drawn[i]);
  (*values)[0] = drawn[0];
  return values;
}

void RegenerateLowering::accumulateSiteWeight(const DistSite &site,
                                              FunctionOpInterface logpdfFn,
                                              ArrayRef<Value> values,
                                              bool termDirty,
                                              SiteCarry &carry) {
  impulse::SampleOp op = site.op;
  Location loc = op.getLoc();
  Value oldScore;
  if (hasScoreTail)
    oldScore = scalarAt(rewriter, loc, prevTrace,
                        {0, clone.positionSize() + siteOrdinal});

  if (scoring) {
    auto it = siteIndex.find(op.getSymbolAttr());
    if (it != siteIndex.end()) {
      Value score = emitSiteLogpdf(rewriter, op, logpdfFn, values);
      carry.weight = recordSiteScore(loc, carry.weight, it->second, score);
    }
    return;
  }
  if (hasScoreTail && !termDirty) {
    clone.siteScores.push_back(oldScore);
    return;
  }

  Value logpdf = emitSiteLogpdf(rewriter, op, logpdfFn, values);
  Value contribution =
      hasScoreTail
          ? arith::SubFOp::create(rewriter, loc, logpdf, oldScore).getResult()
          : logpdf;
  carry.weight =
      arith::AddFOp::create(rewriter, loc, carry.weight, contribution);
  if (hasScoreTail)
    clone.siteScores.push_back(logpdf);
  if (hastings && site.isSelected) {
    // A prior draw's proposal density cancels its own term.
    carry.hastingsCorrection = arith::AddFOp::create(
        rewriter, loc, carry.hastingsCorrection, contribution);
    if (!hasScoreTail && !siteIndex.count(op.getSymbolAttr())) {
      // Parameters unchanged by the move: score the old value inline.
      Value oldLogpdf = emitSiteLogpdf(
          rewriter, op, logpdfFn,
          sliceTraceValues(rewriter, loc, prevTrace, site.column,
                           TypeRange(op->getResultTypes()).drop_front()));
      carry.hastingsCorrection = arith::SubFOp::create(
          rewriter, loc, carry.hastingsCorrection, oldLogpdf);
    }
  } else if (site.flipIdx >= 0) {
    // Kept: old density, cancelled via oldSiteScores; fresh: new density.
    Value term = arith::SelectOp::create(
        rewriter, loc, site.oldLive,
        scalarAt(rewriter, loc, oldSiteScores, {1, site.flipIdx}), logpdf);
    carry.hastingsCorrection =
        arith::AddFOp::create(rewriter, loc, carry.hastingsCorrection, term);
  }
}

FailureOr<SmallVector<Value>>
RegenerateLowering::lowerCallSite(impulse::SampleOp op, SiteCarry &carry) {
  FailureOr<NestedSite> nested =
      resolveNestedSite(rewriter, op, clone, symbolTable);
  if (failed(nested))
    return failure();
  if (nested->subAllAddresses.empty())
    return emitSiteDraw(rewriter, symbolTable, op);
  if (nested->subPositionSize <= 0)
    return op.emitError("Impulse: failed to compute sub-position size");
  Location loc = op.getLoc();
  Value subPrevTrace = emitColumnSlice(rewriter, loc, nested->getTraceType(),
                                       prevTrace, nested->mergeOffset);

  Attribute symbol = op.getSymbolAttr();
  ArrayAttr subSelection = buildSubSelection(rewriter, selection, symbol);
  Value subConstraint;
  ArrayAttr subConstrainedForOp;
  if (constraint && constrainedAddresses) {
    ArrayAttr subConstrainedAddrs =
        buildSubSelection(rewriter, constrainedAddresses, symbol);
    if (!subConstrainedAddrs.empty()) {
      int64_t subConstraintSize = computePositionSizeForSelection(
          op, nested->genFn, subConstrainedAddrs, symbolTable);
      int64_t subConstraintOffset = computeOffsetForSampleInSelection(
          op, clone.fn, constrainedAddresses, symbol, symbolTable);
      if (subConstraintSize > 0 && subConstraintOffset >= 0) {
        auto subConstraintType = RankedTensorType::get({1, subConstraintSize},
                                                       rewriter.getF64Type());
        subConstraint = emitColumnSlice(rewriter, loc, subConstraintType,
                                        constraint, subConstraintOffset);
        subConstrainedForOp = subConstrainedAddrs;
      }
    }
  }

  if (hastings) {
    bool genHasIf = false;
    nested->genFn.getFunctionBody().walk(
        [&](impulse::IfOp) { genHasIf = true; });
    bool inputsTainted =
        llvm::any_of(llvm::drop_begin(op.getInputs()),
                     [&](Value v) { return hastingsTaint.contains(v); });
    if (inFlippableBranch(op) ||
        (inputsTainted && (!subSelection.empty() || genHasIf)))
      return op.emitError("Impulse: MH over a generative call whose sites can "
                          "change liveness or parameters under the move is "
                          "not supported");
  }

  Type weightType = hastings ? RankedTensorType::get({2}, rewriter.getF64Type())
                             : clone.zeroWeight.getType();
  auto nestedRegenerate = impulse::RegenerateOp::create(
      rewriter, loc, nested->getResultTypes(weightType), op.getFnAttr(),
      op.getInputs(), subPrevTrace, subConstraint, nested->subAllAddresses,
      subSelection, subConstrainedForOp, rewriter.getStringAttr(""));
  Value subWeight = nestedRegenerate.getWeight();
  if (hastings) {
    nestedRegenerate->setAttr(kMHHastingsAttr, rewriter.getUnitAttr());
    auto [weight, correction] = unpackHastingsWeight(rewriter, loc, subWeight);
    carry.weight = arith::AddFOp::create(rewriter, loc, carry.weight, weight);
    carry.hastingsCorrection = arith::AddFOp::create(
        rewriter, loc, carry.hastingsCorrection, correction);
  } else if (!scoring) {
    carry.weight =
        arith::AddFOp::create(rewriter, loc, carry.weight, subWeight);
  }
  mergeNestedTrace(rewriter, loc, *nested, nestedRegenerate.getNewTrace(),
                   carry);
  return SmallVector<Value>(nestedRegenerate.getOutputs());
}

Value RegenerateLowering::emitWeightOut(Location loc) {
  const SiteCarry &carry = clone.carry;
  if (!hastings)
    return carry.weight;
  Value correction = carry.hastingsCorrection;
  if (oldSiteScores)
    for (int64_t
             k = 0,
             e = cast<RankedTensorType>(oldSiteScores.getType()).getShape()[1];
         k < e; ++k)
      correction =
          arith::SubFOp::create(rewriter, loc, correction,
                                scalarAt(rewriter, loc, oldSiteScores, {1, k}));
  return packHastingsWeight(rewriter, loc, carry.weight, correction);
}

SmallVector<impulse::IfOp> RegenerateLowering::enclosingIfs(Operation *site) {
  SmallVector<impulse::IfOp> ifs;
  for (Operation *p = site->getParentOp(); p && p != clone.newFn.getOperation();
       p = p->getParentOp())
    if (auto ifOp = dyn_cast<impulse::IfOp>(p))
      ifs.push_back(ifOp);
  return ifs;
}

bool RegenerateLowering::inFlippableBranch(Operation *site) {
  return llvm::any_of(enclosingIfs(site), [&](impulse::IfOp ifOp) {
    return hastingsTaint.contains(ifOp.getPredicate());
  });
}

bool RegenerateLowering::readsConstraint(impulse::SampleOp site) {
  if (!constraint || !constrainedAddresses)
    return false;
  return !(hastings || scoring) || !enclosingIfs(site).empty();
}

int64_t RegenerateLowering::flipLatentIndex(impulse::SampleOp site,
                                            bool isSelected) {
  if (!hastings || isSelected)
    return -1;
  auto it = siteIndex.find(site.getSymbolAttr());
  return it == siteIndex.end() ? -1 : it->second;
}

Value RegenerateLowering::emitOldLive(Location loc, int64_t k) {
  Value flag = scalarAt(rewriter, loc, oldSiteScores, {0, k});
  Value half = emitSplatConstant(
      rewriter, loc, cast<RankedTensorType>(clone.zeroWeight.getType()), 0.5);
  return arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGT, flag,
                               half);
}

Value RegenerateLowering::recordSiteScore(Location loc, Value acc, int64_t k,
                                          Value score) {
  auto cellType = RankedTensorType::get({1, 1}, rewriter.getF64Type());
  Value one = emitSplatConstant(rewriter, loc, cellType, 1.0);
  Value scoreCell = impulse::ReshapeOp::create(rewriter, loc, cellType, score);
  acc = impulse::DynamicUpdateSliceOp::create(
      rewriter, loc, acc.getType(), acc, one,
      ValueRange{emitI64ScalarConstant(rewriter, loc, 0),
                 emitI64ScalarConstant(rewriter, loc, k)});
  return impulse::DynamicUpdateSliceOp::create(
      rewriter, loc, acc.getType(), acc, scoreCell,
      ValueRange{emitI64ScalarConstant(rewriter, loc, 1),
                 emitI64ScalarConstant(rewriter, loc, k)});
}

namespace {

/// Lowers the Impulse inference ops. Runs of chained single-site MH moves are
/// batched before the patterns run.
struct ExpandImpulsePass
    : public enzyme::impl::ExpandImpulsePassBase<ExpandImpulsePass> {
  using ExpandImpulsePassBase::ExpandImpulsePassBase;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::OpPassManager pm;
    mlir::LogicalResult result = mlir::parsePassPipeline(postpasses, pm);
    if (!mlir::failed(result)) {
      pm.getDependentDialects(registry);
    }

    registry
        .insert<mlir::arith::ArithDialect, mlir::math::MathDialect,
                mlir::enzyme::EnzymeDialect, mlir::impulse::ImpulseDialect>();
  }
};

/// Lowers `impulse.untraced_call` to a call of a clone of the callee whose
/// sample sites draw fresh values.
struct LowerUntracedCallPattern
    : public mlir::OpRewritePattern<impulse::UntracedCallOp> {
  using mlir::OpRewritePattern<impulse::UntracedCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(impulse::UntracedCallOp op,
                                PatternRewriter &rewriter) const override {
    SymbolTableCollection symbolTable;

    auto fn = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
        op, op.getFnAttr());

    if (fn.getFunctionBody().empty()) {
      op.emitError("Impulse: trying to call an empty function");
      return failure();
    }

    FunctionOpInterface newFn = cloneForMode(fn, ImpulseMode::Call);

    SmallVector<Operation *, 4> toErase;
    newFn.walk([&](impulse::SampleOp sampleOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(sampleOp);

      rewriter.replaceAllUsesWith(
          sampleOp->getResults(),
          emitSiteDraw(rewriter, symbolTable, sampleOp));
      toErase.push_back(sampleOp);
    });

    for (Operation *site : toErase)
      rewriter.eraseOp(site);

    rewriter.setInsertionPoint(op);
    auto call = func::CallOp::create(rewriter, op.getLoc(), newFn.getName(),
                                     newFn.getResultTypes(), op.getOperands());

    rewriter.replaceOp(op, call.getResults());

    return success();
  }
};

/// Lowers `impulse.simulate` to a call of a clone of the model that draws
/// every site and also returns the trace of `all_addresses` and the joint log
/// density.
struct LowerSimulatePattern
    : public mlir::OpRewritePattern<impulse::SimulateOp> {
  using mlir::OpRewritePattern<impulse::SimulateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(impulse::SimulateOp op,
                                PatternRewriter &rewriter) const override {
    SymbolTableCollection symbolTable;
    int64_t declaredWidth =
        cast<RankedTensorType>(op.getTrace().getType()).getShape()[1];
    FailureOr<TracedClone> clone =
        beginTracedClone(op, op.getFnAttr(), op.getAllAddressesAttr(),
                         declaredWidth, ImpulseMode::Simulate,
                         /*constraintSize=*/-1, "simulate", symbolTable);
    if (failed(clone))
      return failure();

    auto lowerSite = [&](impulse::SampleOp sampleOp,
                         SiteCarry &carry) -> FailureOr<SmallVector<Value>> {
      Location loc = sampleOp.getLoc();
      if (sampleOp.getDistAttr() || sampleOp.getLogpdfFnAttr()) {
        SmallVector<Value> sampledValues =
            emitSiteDraw(rewriter, symbolTable, sampleOp);
        if (failed(scoreAndTraceSite(
                rewriter, sampleOp, ArrayRef<Value>(sampledValues).drop_front(),
                *clone, carry, symbolTable)))
          return failure();
        return sampledValues;
      }

      FailureOr<NestedSite> nested =
          resolveNestedSite(rewriter, sampleOp, *clone, symbolTable);
      if (failed(nested))
        return failure();
      if (nested->subAllAddresses.empty())
        return emitSiteDraw(rewriter, symbolTable, sampleOp);
      if (nested->subPositionSize <= 0)
        return sampleOp.emitError(
            "Impulse: failed to compute sub-position size");
      auto nestedSimulate = impulse::SimulateOp::create(
          rewriter, loc, nested->getResultTypes(clone->zeroWeight.getType()),
          sampleOp.getFnAttr(), sampleOp.getInputs(), nested->subAllAddresses);
      carry.weight = arith::AddFOp::create(rewriter, loc, carry.weight,
                                           nestedSimulate.getWeight());
      mergeNestedTrace(rewriter, loc, *nested, nestedSimulate.getTrace(),
                       carry);
      return SmallVector<Value>(nestedSimulate.getOutputs());
    };

    return finishTracedClone(
        rewriter, op, *clone, lowerSite,
        [&](Location) { return clone->carry.weight; }, op.getInputs());
  }
};

/// Lowers `impulse.mh` to a Hastings `impulse.regenerate` of the selection
/// and an accept test.
struct LowerMHPattern : public mlir::OpRewritePattern<impulse::MHOp> {
  using mlir::OpRewritePattern<impulse::MHOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(impulse::MHOp mhOp,
                                PatternRewriter &rewriter) const override {
    SymbolTableCollection symbolTable;
    FailureOr<FunctionOpInterface> fn =
        lookupModelBody(mhOp, mhOp.getFnAttr(), "mh", symbolTable);
    if (failed(fn))
      return failure();

    Location loc = mhOp.getLoc();

    Value oldTrace = mhOp.getOriginalTrace();
    Value oldWeight = mhOp.getOriginalWeight();
    auto allAddresses = mhOp.getAllAddressesAttr();

    Type traceType = oldTrace.getType();

    auto nameAttr = mhOp.getNameAttr();
    if (!nameAttr)
      nameAttr = rewriter.getStringAttr("");

    auto selection = mhOp.getSelectionAttr();

    FailureOr<FlatTraceLayout> layout =
        getFlatTraceLayout(mhOp, *fn, allAddresses, symbolTable);
    if (failed(layout))
      return failure();
    // On a score-tail trace, regenerate returns the dirty-term weight delta.
    int64_t width = cast<RankedTensorType>(traceType).getShape()[1];
    bool hasScoreTail = width > layout->positionSize;

    MHProposal proposal =
        proposeFlatMove(rewriter, loc, mhOp, mhOp.getInputs(), oldTrace,
                        nameAttr, fn->getResultTypes());
    proposal.changed = mhChangedColumnsFlat(*layout, selection, width);
    MHStep step =
        emitMHStep(rewriter, loc, oldTrace, oldWeight, proposal, hasScoreTail);
    rewriter.replaceOp(mhOp,
                       {step.trace, step.weight, step.accepted, step.rng});
    return success();
  }
};

/// Lowers `impulse.generate` to a call of a clone of the model that reads the
/// constrained sites from the constraint, draws the others and also returns
/// the trace of `all_addresses` and the joint log density.
struct LowerGeneratePattern
    : public mlir::OpRewritePattern<impulse::GenerateOp> {
  using mlir::OpRewritePattern<impulse::GenerateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(impulse::GenerateOp op,
                                PatternRewriter &rewriter) const override {
    SymbolTableCollection symbolTable;
    // Operand width, not the packed size: extra columns are simply ignored.
    int64_t constraintSize =
        cast<RankedTensorType>(op.getConstraint().getType()).getShape()[1];
    int64_t declaredWidth =
        cast<RankedTensorType>(op.getTrace().getType()).getShape()[1];
    FailureOr<TracedClone> clone = beginTracedClone(
        op, op.getFnAttr(), op.getAllAddressesAttr(), declaredWidth,
        ImpulseMode::Generate, constraintSize, "generate", symbolTable);
    if (failed(clone))
      return failure();
    ArrayAttr constrainedAddresses = op.getConstrainedAddressesAttr();
    Value constraint = clone->newFn.getArgument(0);

    auto lowerSite = [&](impulse::SampleOp sampleOp,
                         SiteCarry &carry) -> FailureOr<SmallVector<Value>> {
      Location loc = sampleOp.getLoc();
      if (sampleOp.getDistAttr() || sampleOp.getLogpdfFnAttr()) {
        if (failed(verifyTopLevelAddress(sampleOp, constrainedAddresses,
                                         "constrained")))
          return failure();
        bool isConstrained = addressInList(sampleOp, constrainedAddresses);
        SmallVector<Value> sampledValues;
        if (isConstrained) {
          int64_t constrainedOffset = computeOffsetForSampleInSelection(
              op, clone->fn, constrainedAddresses, sampleOp.getSymbolAttr(),
              symbolTable);
          FailureOr<SmallVector<Value>> read =
              readSiteValues(rewriter, sampleOp, constraint, constrainedOffset);
          if (failed(read))
            return failure();
          sampledValues = std::move(*read);
        } else {
          if (impulse::DistAttr dist = sampleOp.getDistAttr();
              dist && !dist.hasLoopFreeSampler())
            return sampleOp.emitError(
                "Impulse: distribution sampler is not "
                "emittable; the site must be constrained");
          sampledValues = emitSiteDraw(rewriter, symbolTable, sampleOp);
        }
        if (failed(scoreAndTraceSite(
                rewriter, sampleOp, ArrayRef<Value>(sampledValues).drop_front(),
                *clone, carry, symbolTable,
                isConstrained ? "logpdf call for constrained sample"
                              : "logpdf call")))
          return failure();
        return sampledValues;
      }

      FailureOr<NestedSite> nested =
          resolveNestedSite(rewriter, sampleOp, *clone, symbolTable);
      if (failed(nested))
        return failure();
      if (nested->subAllAddresses.empty())
        return emitSiteDraw(rewriter, symbolTable, sampleOp);
      Attribute symbol = sampleOp.getSymbolAttr();
      ArrayAttr subConstrainedAddrs =
          buildSubSelection(rewriter, constrainedAddresses, symbol);
      int64_t subConstraintSize = computePositionSizeForSelection(
          sampleOp, nested->genFn, subConstrainedAddrs, symbolTable);
      if (nested->subPositionSize <= 0 || subConstraintSize < 0)
        return sampleOp.emitError("Impulse: failed to compute sub-position "
                                  "or sub-constraint size");

      auto subConstraintType =
          RankedTensorType::get({1, subConstraintSize}, rewriter.getF64Type());
      Value subConstraint;
      if (subConstraintSize > 0) {
        int64_t subConstraintOffset = computeOffsetForSampleInSelection(
            sampleOp, clone->fn, constrainedAddresses, symbol, symbolTable);
        subConstraint = emitColumnSlice(rewriter, loc, subConstraintType,
                                        constraint, subConstraintOffset);
      } else {
        subConstraint =
            emitSplatConstant(rewriter, loc, subConstraintType, 0.0);
      }
      auto nestedGenerate = impulse::GenerateOp::create(
          rewriter, loc, nested->getResultTypes(clone->zeroWeight.getType()),
          sampleOp.getFnAttr(), sampleOp.getInputs(), subConstraint,
          nested->subAllAddresses, subConstrainedAddrs);
      carry.weight = arith::AddFOp::create(rewriter, loc, carry.weight,
                                           nestedGenerate.getWeight());
      mergeNestedTrace(rewriter, loc, *nested, nestedGenerate.getTrace(),
                       carry);
      return SmallVector<Value>(nestedGenerate.getOutputs());
    };

    SmallVector<Value> callOperands{op.getConstraint()};
    llvm::append_range(callOperands, op.getInputs());
    return finishTracedClone(
        rewriter, op, *clone, lowerSite,
        [&](Location) { return clone->carry.weight; }, callOperands);
  }
};

/// Lowers `impulse.regenerate` to a call of a clone of the model that redraws
/// the selected sites, reads the others from the old trace and also returns
/// the new trace and the weight that `RegenerateLowering` describes.
struct LowerRegeneratePattern
    : public mlir::OpRewritePattern<impulse::RegenerateOp> {
  using mlir::OpRewritePattern<impulse::RegenerateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(impulse::RegenerateOp op,
                                PatternRewriter &rewriter) const override {
    SymbolTableCollection symbolTable;
    bool hasConstraint = static_cast<bool>(op.getConstraint());
    // Operand width, not the packed size, as in generate.
    int64_t constraintSize =
        hasConstraint
            ? cast<RankedTensorType>(op.getConstraint().getType()).getShape()[1]
            : -1;
    int64_t declaredWidth =
        cast<RankedTensorType>(op.getNewTrace().getType()).getShape()[1];
    FailureOr<TracedClone> clone = beginTracedClone(
        op, op.getFnAttr(), op.getAllAddressesAttr(), declaredWidth,
        ImpulseMode::Regenerate, constraintSize, "regenerate", symbolTable);
    if (failed(clone))
      return failure();
    RegenerateLowering lowering(rewriter, op, *clone, symbolTable);
    if (failed(lowering.prepare()))
      return failure();

    SmallVector<Value> callOperands{op.getOriginalTrace()};
    if (hasConstraint)
      callOperands.push_back(op.getConstraint());
    llvm::append_range(callOperands, op.getInputs());
    return finishTracedClone(
        rewriter, op, *clone,
        [&](impulse::SampleOp site, SiteCarry &carry) {
          return lowering.lowerSite(site, carry);
        },
        [&](Location loc) { return lowering.emitWeightOut(loc); },
        callOperands);
  }
};

} // end anonymous namespace

/// Applies `patterns` to `root` and fails when they do not converge, a pattern
/// emits an error or an inference op is left. A pattern that fails after an
/// error leaves its op in place, and the driver can visit the op again and
/// repeat the error, so each error is reported once.
static LogicalResult lowerInferenceOps(Operation *root,
                                       RewritePatternSet &&patterns) {
  bool hadError = false;
  llvm::StringSet<> reported;
  ScopedDiagnosticHandler reportOnce(root->getContext(), [&](Diagnostic &diag) {
    if (diag.getSeverity() != DiagnosticSeverity::Error)
      return failure();
    hadError = true;
    std::string key;
    llvm::raw_string_ostream os(key);
    os << diag.getLocation() << ": " << diag;
    return success(!reported.insert(key).second);
  });
  if (failed(applyPatternsGreedily(root, std::move(patterns))) || hadError)
    return failure();
  root->walk([](Operation *op) {
    if (isa<impulse::UntracedCallOp, impulse::SimulateOp, impulse::GenerateOp,
            impulse::RegenerateOp, impulse::MHOp, impulse::InferOp,
            impulse::InferRegionOp>(op))
      op->emitError("Impulse: expand-impulse could not lower this op");
  });
  return failure(hadError);
}

void ExpandImpulsePass::runOnOperation() {
  // Must run before the per-op MH patterns lower these moves.
  batchMHRuns(getOperation());

  RewritePatternSet patterns(&getContext());
  patterns.add<LowerUntracedCallPattern, LowerSimulatePattern,
               LowerGeneratePattern, LowerMHPattern, LowerRegeneratePattern>(
      &getContext());
  populateExpandImpulseHMCVIPatterns(patterns, debugDump);
  populateExpandImpulseMHGibbsPatterns(patterns);

  if (failed(lowerInferenceOps(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  if (!postpasses.empty()) {
    mlir::PassManager pm(getOperation()->getContext());

    if (mlir::failed(mlir::parsePassPipeline(postpasses, pm))) {
      getOperation()->emitError()
          << "Failed to parse expand-impulse post-passes pipeline: "
          << postpasses;
      signalPassFailure();
      return;
    }

    if (mlir::failed(pm.run(getOperation()))) {
      signalPassFailure();
      return;
    }
  }
}
