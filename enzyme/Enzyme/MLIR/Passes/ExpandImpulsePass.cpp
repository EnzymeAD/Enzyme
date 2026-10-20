//===- ExpandImpulsePass.cpp - Expand Impulse region ops ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that expands high-level Impulse region ops
// (simulate, generate, regenerate, mcmc, mh, untraced_call) into lower-level
// Impulse ops (sample, random, etc.) plus arith/math/cf.
//===----------------------------------------------------------------------===//

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"
#include "Interfaces/DistBuilder.h"
#include "Interfaces/HMCUtils.h"
#include "Interfaces/ImpulseUtils.h"
#include "Interfaces/VIUtils.h"
#include "PassDetails.h"
#include "Passes/InferStateLayout.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/BitVector.h"

#include <type_traits>

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

namespace {

static int64_t computeTensorElementCount(RankedTensorType tensorType) {
  int64_t elemCount = 1;
  for (auto dim : tensorType.getShape()) {
    if (dim == ShapedType::kDynamic)
      return -1;
    elemCount *= dim;
  }
  return elemCount;
}

namespace state_slot {
constexpr StringLiteral kPosition = "position";
constexpr StringLiteral kGradient = "gradient";
constexpr StringLiteral kPotentialEnergy = "potential_energy";
constexpr StringLiteral kStepSize = "step_size";
constexpr StringLiteral kInverseMassMatrix = "inverse_mass_matrix";
constexpr StringLiteral kDaLogStepSize = "da_log_step_size";
constexpr StringLiteral kWindowIdx = "window_idx";
constexpr StringLiteral kWarmupOffset = "warmup_offset";
constexpr StringLiteral kRng = "rng";
} // namespace state_slot

template <typename MCMCOpT>
static Value stateInSlot(MCMCOpT op, StringRef name) {
  auto idx = mlir::impulse::slotIndex(op.getStateSchemaAttr(), name);
  if (!idx)
    return Value();
  auto stateIn = op.getStateIn();
  if (*idx >= stateIn.size())
    return Value();
  return stateIn[*idx];
}

using SampleOpMap = DenseMap<Attribute, impulse::SampleOp>;

static SampleOpMap buildSampleOpMap(FunctionOpInterface fn) {
  SampleOpMap map;
  fn.walk([&](impulse::SampleOp sampleOp) {
    if (auto symbol = sampleOp.getSymbolAttr())
      map[symbol] = sampleOp;
  });
  return map;
}

static impulse::SampleOp findSampleBySymbol(const SampleOpMap &map,
                                            Attribute targetSymbol) {
  auto it = map.find(targetSymbol);
  return it != map.end() ? it->second : nullptr;
}

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
    int64_t elemCount = computeTensorElementCount(tensorType);
    if (elemCount < 0) {
      op->emitError("Dynamic tensor dimensions not supported");
      return -1;
    }
    totalCount += elemCount;
  }
  return totalCount;
}

static bool computePositionSizeForAddress(Operation *op,
                                          const SampleOpMap &sampleMap,
                                          ArrayRef<Attribute> address,
                                          SymbolTableCollection &symbolTable,
                                          int64_t &positionSize) {
  if (address.empty())
    return false;

  auto sampleOp = findSampleBySymbol(sampleMap, address[0]);
  if (!sampleOp)
    return false;

  if (address.size() > 1) {
    if (sampleOp.getLogpdfAttr()) {
      op->emitError("Cannot select nested address in distribution function");
      return false;
    }

    auto genFn = cast<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(sampleOp, sampleOp.getFnAttr()));
    if (!genFn || genFn.getFunctionBody().empty()) {
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

static int64_t
computeOffsetForSampleInSelection(Operation *op, FunctionOpInterface fn,
                                  ArrayAttr selection, Attribute targetSymbol,
                                  SymbolTableCollection &symbolTable) {
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

/// Position order: declaration (VI, NumPyro parity) or `selection` as given.
static SmallVector<impulse::SupportInfo>
collectSupportInfoForSelection(Operation *op, FunctionOpInterface fn,
                               ArrayAttr selection, ArrayAttr allAddresses,
                               SymbolTableCollection &symbolTable,
                               bool positionDeclarationOrder) {
  auto sampleMap = buildSampleOpMap(fn);
  SmallVector<impulse::SupportInfo> supports;
  int64_t currentPositionOffset = 0;
  int64_t currentUnconstrainedOffset = 0;

  SmallVector<Attribute> orderedSelected;
  if (positionDeclarationOrder) {
    llvm::SmallDenseSet<Attribute, 4> selectedTopLevel;
    for (auto addr : selection) {
      auto address = cast<ArrayAttr>(addr);
      // TODO: Handle nested cases
      if (address.empty() || address.size() != 1)
        continue;
      selectedTopLevel.insert(address[0]);
    }
    for (auto addr : allAddresses) {
      auto address = cast<ArrayAttr>(addr);
      if (address.empty() || address.size() != 1)
        continue;
      if (!selectedTopLevel.contains(address[0]))
        continue;
      orderedSelected.push_back(address[0]);
    }
  } else {
    for (auto addr : selection) {
      auto address = cast<ArrayAttr>(addr);
      if (address.empty() || address.size() != 1)
        continue;
      orderedSelected.push_back(address[0]);
    }
  }

  for (Attribute targetSymbol : orderedSelected) {
    auto sampleOp = findSampleBySymbol(sampleMap, targetSymbol);
    if (!sampleOp)
      continue;

    auto supportAttr = sampleOp.getSupportAttr();

    int64_t sampleSize = computeSampleElementCount(op, sampleOp);
    if (sampleSize < 0)
      continue;

    int64_t unconstrainedSize =
        transforms::getUnconstrainedSize(sampleSize, supportAttr);

    int64_t traceOffset = computeOffsetForSampleInSelection(
        op, fn, allAddresses, targetSymbol, symbolTable);
    if (traceOffset < 0) {
      op->emitError("Symbol in selection not found in all_addresses - cannot "
                    "determine trace offset for scattered selection");
      return {};
    }

    supports.emplace_back(currentPositionOffset, traceOffset, sampleSize,
                          currentUnconstrainedOffset, unconstrainedSize,
                          supportAttr);
    currentPositionOffset += sampleSize;
    currentUnconstrainedOffset += unconstrainedSize;
  }

  return supports;
}

/// Position offsets follow `selection`, trace offsets `all_addresses`.
static SmallVector<impulse::SupportInfo>
collectSupportInfoForRegion(impulse::InferRegionOp regionOp) {
  SmallVector<impulse::SupportInfo> supports;
  ArrayAttr selection = regionOp.getSelectionAttr();
  ArrayAttr allAddresses = regionOp.getAllAddressesAttr();
  if (!selection || !allAddresses || regionOp.getSampler().empty())
    return supports;

  // Branch-local sites also hold trace columns, so walk the whole sampler.
  DenseMap<Attribute, impulse::SampleRegionOp> siteByAddress;
  regionOp.getSampler().walk([&](impulse::SampleRegionOp site) {
    if (auto address = site.getAddressAttr())
      siteByAddress[address] = site;
  });

  auto siteSize = [](impulse::SampleRegionOp site) -> int64_t {
    int64_t size = 0;
    for (unsigned i = 1, e = site.getNumResults(); i < e; ++i)
      size +=
          cast<RankedTensorType>(site.getResult(i).getType()).getNumElements();
    return size;
  };

  int64_t positionOffset = 0;
  int64_t unconstrainedOffset = 0;
  for (Attribute addr : selection) {
    auto it = siteByAddress.find(addr);
    if (it == siteByAddress.end())
      return {};
    impulse::SampleRegionOp site = it->second;

    int64_t traceOffset = 0;
    bool foundInTrace = false;
    for (Attribute traceAddr : allAddresses) {
      if (traceAddr == addr) {
        foundInTrace = true;
        break;
      }
      auto traceIt = siteByAddress.find(traceAddr);
      if (traceIt == siteByAddress.end())
        return {};
      traceOffset += siteSize(traceIt->second);
    }
    if (!foundInTrace)
      return {};

    int64_t constrainedSize = siteSize(site);
    int64_t unconstrainedSize = transforms::getUnconstrainedSize(
        constrainedSize, site.getSupportAttr());
    supports.emplace_back(positionOffset, traceOffset, constrainedSize,
                          unconstrainedOffset, unconstrainedSize,
                          site.getSupportAttr());
    positionOffset += constrainedSize;
    unconstrainedOffset += unconstrainedSize;
  }
  return supports;
}

/// Declaration-order VI layout, matching NumPyro's per-site RNG draw order.
struct RegionVILayout {
  SmallVector<impulse::SupportInfo> supports;
  SmallVector<std::pair<int64_t, int64_t>> argSlices; // selection order
  int64_t positionSize = 0;
};

static bool collectRegionVILayout(impulse::InferRegionOp regionOp,
                                  RegionVILayout &out) {
  ArrayAttr selection = regionOp.getSelectionAttr();
  ArrayAttr allAddresses = regionOp.getAllAddressesAttr();
  if (!selection || !allAddresses || regionOp.getSampler().empty())
    return false;

  DenseMap<Attribute, impulse::SampleRegionOp> siteByAddress;
  regionOp.getSampler().walk([&](impulse::SampleRegionOp site) {
    if (auto address = site.getAddressAttr())
      siteByAddress[address] = site;
  });

  auto resultSizes = [](impulse::SampleRegionOp site,
                        SmallVectorImpl<int64_t> &sizes) {
    for (unsigned i = 1, e = site.getNumResults(); i < e; ++i)
      sizes.push_back(
          cast<RankedTensorType>(site.getResult(i).getType()).getNumElements());
  };

  DenseSet<Attribute> selectedSet(selection.begin(), selection.end());

  DenseMap<Attribute, int64_t> declPositionOffset;
  int64_t traceOffset = 0, positionOffset = 0, unconstrainedOffset = 0;
  for (Attribute addr : allAddresses) {
    auto it = siteByAddress.find(addr);
    if (it == siteByAddress.end())
      return false;
    impulse::SampleRegionOp site = it->second;
    SmallVector<int64_t> sizes;
    resultSizes(site, sizes);
    int64_t siteSize = 0;
    for (int64_t s : sizes)
      siteSize += s;
    if (selectedSet.contains(addr)) {
      int64_t unconstrainedSize =
          transforms::getUnconstrainedSize(siteSize, site.getSupportAttr());
      out.supports.emplace_back(positionOffset, traceOffset, siteSize,
                                unconstrainedOffset, unconstrainedSize,
                                site.getSupportAttr());
      declPositionOffset[addr] = positionOffset;
      positionOffset += siteSize;
      unconstrainedOffset += unconstrainedSize;
    }
    traceOffset += siteSize;
  }
  out.positionSize = positionOffset;

  for (Attribute addr : selection) {
    auto it = siteByAddress.find(addr);
    if (it == siteByAddress.end())
      return false;
    SmallVector<int64_t> sizes;
    resultSizes(it->second, sizes);
    int64_t within = 0;
    for (int64_t s : sizes) {
      out.argSlices.emplace_back(declPositionOffset[addr] + within, s);
      within += s;
    }
  }
  return true;
}

static ArrayAttr buildSubSelection(OpBuilder &builder, ArrayAttr selection,
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

static int64_t
computeOffsetForNestedSample(Operation *op, FunctionOpInterface fn,
                             ArrayAttr selection, Attribute targetSymbol,
                             SymbolTableCollection &symbolTable) {
  auto sampleMap = buildSampleOpMap(fn);
  int64_t offset = 0;

  for (auto addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (address.empty())
      continue;

    if (address[0] == targetSymbol) {
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

// `offset` advances over both `impulse.if` branches: disjoint trace columns.
struct BranchCarry {
  Value currTrace;
  Value weight;
  int64_t offset;
  Value corr = nullptr;
};

using ProcessSampleFn =
    llvm::function_ref<WalkResult(impulse::SampleOp, BranchCarry &)>;

// kMHHastingsAttr: weight = [weight, log q(fresh) - log q(replaced)].
// kMHSiteScoresAttr: weight = [liveness; masked log density] per address.
static constexpr llvm::StringLiteral kMHHastingsAttr = "impulse.mh_hastings";
static constexpr llvm::StringLiteral kMHSiteScoresAttr =
    "impulse.mh_site_scores";

static Value scalarAt(OpBuilder &b, Location loc, Value tensor,
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

static Value packHastingsWeight(OpBuilder &b, Location loc, Value weight,
                                Value corr) {
  auto vecType = RankedTensorType::get({2}, b.getF64Type());
  auto oneType = RankedTensorType::get({1}, b.getF64Type());
  auto i64S = RankedTensorType::get({}, b.getI64Type());
  Value acc = arith::ConstantOp::create(b, loc, vecType,
                                        DenseElementsAttr::get(vecType, 0.0));
  // Named array: a ValueRange over a braced list would dangle in the loop.
  Value parts[] = {weight, corr};
  for (auto [i, v] : llvm::enumerate(parts)) {
    Value idx = arith::ConstantOp::create(
        b, loc, i64S, DenseElementsAttr::get(i64S, b.getI64IntegerAttr(i)));
    Value flat = impulse::ReshapeOp::create(b, loc, oneType, v);
    acc = impulse::DynamicUpdateSliceOp::create(b, loc, vecType, acc, flat,
                                                ValueRange{idx});
  }
  return acc;
}

static std::pair<Value, Value> unpackHastingsWeight(OpBuilder &b, Location loc,
                                                    Value packed) {
  return {scalarAt(b, loc, packed, {0}), scalarAt(b, loc, packed, {1})};
}

// Per-window DUS, not select(acc, P, T): untouched columns stay T's for LICM.
using ColumnWindows = SmallVector<std::pair<int64_t, int64_t>>;

// Merges half-open [start, end) windows.
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

static Value writeBackMHWindows(OpBuilder &b, Location loc, Value trace,
                                Value proposed, Value accepted,
                                ArrayRef<std::pair<int64_t, int64_t>> windows) {
  auto traceType = cast<RankedTensorType>(trace.getType());
  auto i64S = RankedTensorType::get({}, b.getI64Type());
  auto idx = [&](int64_t v) -> Value {
    return arith::ConstantOp::create(
        b, loc, i64S, DenseElementsAttr::get(i64S, b.getI64IntegerAttr(v)));
  };
  Value out = trace;
  for (const auto &window : windows) {
    int64_t start = window.first, end = window.second;
    auto winType =
        RankedTensorType::get({1, end - start}, traceType.getElementType());
    auto sliceOf = [&](Value v) -> Value {
      return impulse::SliceOp::create(
          b, loc, winType, v, b.getDenseI64ArrayAttr({0, start}),
          b.getDenseI64ArrayAttr({1, end}), b.getDenseI64ArrayAttr({1, 1}));
    };
    Value propWin = sliceOf(proposed);
    Value curWin = sliceOf(trace);
    Value upd =
        impulse::SelectOp::create(b, loc, winType, accepted, propWin, curWin);
    out = impulse::DynamicUpdateSliceOp::create(b, loc, traceType, out, upd,
                                                ValueRange{idx(0), idx(start)});
  }
  return out;
}

// Bare traces also change at branch sites (liveness flips) and sub-calls.
static ColumnWindows mhChangedColumnsFlat(Operation *op, FunctionOpInterface fn,
                                          ArrayAttr layout,
                                          ArrayAttr regenerateAddresses,
                                          ArrayAttr constrainedAddresses,
                                          bool hasConstraint, int64_t width,
                                          SymbolTableCollection &symbolTable) {
  auto sampleMap = buildSampleOpMap(fn);
  int64_t positionSize =
      computePositionSizeForSelection(op, fn, layout, symbolTable);
  bool bare = width == positionSize;
  auto inList = [](ArrayAttr list, ArrayAttr address) {
    return list && llvm::is_contained(list.getValue(), address);
  };
  ColumnWindows windows;
  int64_t offset = 0;
  for (Attribute a : layout) {
    auto address = cast<ArrayAttr>(a);
    int64_t size = computePositionSizeForSelection(
        op, fn, ArrayAttr::get(op->getContext(), {address}), symbolTable);
    impulse::SampleOp site = findSampleBySymbol(sampleMap, address[0]);
    bool isDist = site && (site.getDistAttr() || site.getLogpdfAttr());
    bool changed = inList(regenerateAddresses, address);
    // Constrained sites outside branches read the old trace (readsConstraint).
    if (!changed && bare)
      changed = !isDist || address.size() > 1 ||
                site->getParentOfType<impulse::IfOp>();
    if (changed)
      windows.push_back({offset, offset + size});
    offset += size;
  }
  if (width > positionSize)
    windows.push_back({positionSize, width});
  return mergeColumnWindows(windows);
}

static LogicalResult
processImpulseBlock(Block &block, PatternRewriter &rewriter, BranchCarry &carry,
                    RankedTensorType traceType, ProcessSampleFn processSample,
                    SmallVectorImpl<Operation *> &toErase);

// Branches start from the pre-if snapshot; untaken slots stay zero.
static LogicalResult processImpulseIf(impulse::IfOp ifOp,
                                      PatternRewriter &rewriter,
                                      BranchCarry &carry,
                                      RankedTensorType traceType,
                                      ProcessSampleFn processSample,
                                      SmallVectorImpl<Operation *> &toErase) {
  auto loc = ifOp.getLoc();

  Value snapTrace = carry.currTrace;
  Value snapWeight = carry.weight;
  Value snapCorr = carry.corr;

  Block &trueBlock = ifOp.getTrueBranch().front();
  carry.currTrace = snapTrace;
  carry.weight = snapWeight;
  carry.corr = snapCorr;
  if (failed(processImpulseBlock(trueBlock, rewriter, carry, traceType,
                                 processSample, toErase)))
    return failure();
  Value trueTrace = carry.currTrace;
  Value trueWeight = carry.weight;
  Value trueCorr = carry.corr;

  // Keep `carry.offset`: the branches get disjoint trace columns.
  Block &falseBlock = ifOp.getFalseBranch().front();
  carry.currTrace = snapTrace;
  carry.weight = snapWeight;
  carry.corr = snapCorr;
  if (failed(processImpulseBlock(falseBlock, rewriter, carry, traceType,
                                 processSample, toErase)))
    return failure();
  Value falseTrace = carry.currTrace;
  Value falseWeight = carry.weight;
  Value falseCorr = carry.corr;

  auto extendYield = [&](Block &b, Value t, Value w, Value c) {
    auto yield = cast<impulse::YieldOp>(b.getTerminator());
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(yield);
    SmallVector<Value> ops(yield.getOperands().begin(),
                           yield.getOperands().end());
    ops.push_back(t);
    ops.push_back(w);
    if (c)
      ops.push_back(c);
    impulse::YieldOp::create(rewriter, yield.getLoc(), ops);
    rewriter.eraseOp(yield);
  };
  extendYield(trueBlock, trueTrace, trueWeight, trueCorr);
  extendYield(falseBlock, falseTrace, falseWeight, falseCorr);

  SmallVector<Type> newResultTypes(ifOp.getResultTypes().begin(),
                                   ifOp.getResultTypes().end());
  unsigned oldNumResults = ifOp.getNumResults();
  newResultTypes.push_back(traceType);
  // Not always scalar: an MH site-scores regenerate carries a vector.
  newResultTypes.push_back(snapWeight.getType());
  if (snapCorr)
    newResultTypes.push_back(snapCorr.getType());

  rewriter.setInsertionPoint(ifOp);
  auto newIf =
      impulse::IfOp::create(rewriter, loc, newResultTypes, ifOp.getPredicate());
  rewriter.inlineRegionBefore(ifOp.getTrueBranch(), newIf.getTrueBranch(),
                              newIf.getTrueBranch().end());
  rewriter.inlineRegionBefore(ifOp.getFalseBranch(), newIf.getFalseBranch(),
                              newIf.getFalseBranch().end());

  carry.currTrace = newIf.getResult(oldNumResults);
  carry.weight = newIf.getResult(oldNumResults + 1);
  if (snapCorr)
    carry.corr = newIf.getResult(oldNumResults + 2);

  rewriter.replaceOp(ifOp, newIf.getResults().take_front(oldNumResults));
  return success();
}

static LogicalResult
processImpulseBlock(Block &block, PatternRewriter &rewriter, BranchCarry &carry,
                    RankedTensorType traceType, ProcessSampleFn processSample,
                    SmallVectorImpl<Operation *> &toErase) {
  SmallVector<Operation *> ops;
  for (Operation &op : block)
    ops.push_back(&op);

  for (Operation *op : ops) {
    if (auto sampleOp = dyn_cast<impulse::SampleOp>(op)) {
      if (processSample(sampleOp, carry).wasInterrupted())
        return failure();
    } else if (auto ifOp = dyn_cast<impulse::IfOp>(op)) {
      bool hasSample = false;
      ifOp->walk([&](impulse::SampleOp) { hasSample = true; });
      if (hasSample && failed(processImpulseIf(ifOp, rewriter, carry, traceType,
                                               processSample, toErase)))
        return failure();
    } else {
      for (Region &r : op->getRegions())
        for (Block &b : r.getBlocks())
          if (failed(processImpulseBlock(b, rewriter, carry, traceType,
                                         processSample, toErase)))
            return failure();
    }
  }
  return success();
}

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
                mlir::complex::ComplexDialect, mlir::cf::ControlFlowDialect,
                mlir::enzyme::EnzymeDialect, mlir::impulse::ImpulseDialect>();
  }

  struct LowerUntracedCallPattern
      : public mlir::OpRewritePattern<impulse::UntracedCallOp> {
    using mlir::OpRewritePattern<impulse::UntracedCallOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(impulse::UntracedCallOp CI,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        CI.emitError("Impulse: trying to call an empty function");
        return failure();
      }

      auto putils = ImpulseUtils::CreateFromClone(fn, ImpulseMode::Call);
      FunctionOpInterface NewF = putils->newFunc;

      SmallVector<Operation *, 4> toErase;
      NewF.walk([&](impulse::SampleOp sampleOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sampleOp);

        if (auto dist = sampleOp.getDistAttr();
            dist && impulse::distSamplerEmittable(dist)) {
          Value rng = sampleOp.getInputs().front();
          SmallVector<Value> distArgs(std::next(sampleOp.getInputs().begin()),
                                      sampleOp.getInputs().end());
          auto sampleType =
              cast<RankedTensorType>(sampleOp.getResult(1).getType());
          SmallVector<Value> sampledValues = impulse::emitDistSampler(
              rewriter, sampleOp.getLoc(), dist, rng, distArgs, sampleType);
          sampleOp.replaceAllUsesWith(sampledValues);
        } else {
          auto distFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getFnAttr()));
          auto distCall = func::CallOp::create(
              rewriter, sampleOp.getLoc(), distFn.getName(),
              distFn.getResultTypes(), sampleOp.getInputs());
          sampleOp.replaceAllUsesWith(distCall);
        }

        toErase.push_back(sampleOp);
      });

      for (Operation *op : toErase)
        rewriter.eraseOp(op);

      rewriter.setInsertionPoint(CI);
      auto newCI =
          func::CallOp::create(rewriter, CI.getLoc(), NewF.getName(),
                               NewF.getResultTypes(), CI.getOperands());

      rewriter.replaceOp(CI, newCI.getResults());

      delete putils;

      return success();
    }
  };

  struct LowerSimulatePattern
      : public mlir::OpRewritePattern<impulse::SimulateOp> {
    using mlir::OpRewritePattern<impulse::SimulateOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(impulse::SimulateOp CI,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        CI.emitError(
            "Impulse: calling `simulate` on an empty function; if this "
            "is a distribution function, its sample op should have a "
            "logpdf attribute to avoid recursive `simulate` calls which is "
            "intended for generative functions");
        return failure();
      }

      ArrayAttr selection = CI.getSelectionAttr();
      int64_t positionSize =
          computePositionSizeForSelection(CI, fn, selection, symbolTable);
      if (positionSize <= 0) {
        CI.emitError("Impulse: failed to compute position size for simulate");
        return failure();
      }

      int64_t declaredWidth =
          cast<RankedTensorType>(CI.getTrace().getType()).getShape()[1];
      int64_t scoreTailSize = declaredWidth - positionSize;
      SmallVector<Value> siteScores;
      auto putils = ImpulseUtils::CreateFromClone(fn, ImpulseMode::Simulate,
                                                  declaredWidth);
      FunctionOpInterface NewF = putils->newFunc;

      OpBuilder entryBuilder(putils->initializationBlock,
                             putils->initializationBlock->begin());
      Location initLoc = putils->initializationBlock->begin()->getLoc();
      auto scalarType = RankedTensorType::get({}, entryBuilder.getF64Type());
      auto zeroWeight =
          arith::ConstantOp::create(entryBuilder, initLoc, scalarType,
                                    DenseElementsAttr::get(scalarType, 0.0));
      Value weightAccumulator = zeroWeight;

      auto traceType =
          RankedTensorType::get({1, positionSize}, entryBuilder.getF64Type());
      auto zeroTrace =
          arith::ConstantOp::create(entryBuilder, initLoc, traceType,
                                    DenseElementsAttr::get(traceType, 0.0));
      Value currTrace = zeroTrace;
      int64_t currentOffset = 0;

      SmallVector<Operation *> toErase;
      BranchCarry carry{currTrace, weightAccumulator, currentOffset};
      auto processSample = [&](impulse::SampleOp sampleOp,
                               BranchCarry &carry) -> WalkResult {
        Value &currTrace = carry.currTrace;
        Value &weightAccumulator = carry.weight;
        int64_t &currentOffset = carry.offset;
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sampleOp);

        SmallVector<Value> sampledValues; // Values to replace uses of sample op

        if (auto dist = sampleOp.getDistAttr()) {
          SmallVector<Value> distArgs(std::next(sampleOp.getInputs().begin()),
                                      sampleOp.getInputs().end());
          auto sampleType =
              cast<RankedTensorType>(sampleOp.getResult(1).getType());

          if (impulse::distSamplerEmittable(dist)) {
            Value rng = sampleOp.getInputs().front();
            sampledValues = impulse::emitDistSampler(
                rewriter, sampleOp.getLoc(), dist, rng, distArgs, sampleType);
          } else {
            auto distFn =
                cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                    sampleOp, sampleOp.getFnAttr()));
            auto distCall = func::CallOp::create(
                rewriter, sampleOp.getLoc(), distFn.getName(),
                distFn.getResultTypes(), sampleOp.getInputs());
            sampledValues.append(distCall.getResults().begin(),
                                 distCall.getResults().end());
          }

          Value logpdf = impulse::emitDistLogpdf(
              rewriter, sampleOp.getLoc(), dist, sampledValues[1], distArgs);
          weightAccumulator = arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                                    weightAccumulator, logpdf);
          if (scoreTailSize > 0)
            siteScores.push_back(logpdf);

          bool inSelection = false;
          for (auto addr : selection) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              inSelection = true;
              break;
            }
          }
          if (inSelection) {
            for (unsigned i = 1; i < sampledValues.size(); ++i) {
              auto sampleValue = sampledValues[i];
              auto sampleTy = cast<RankedTensorType>(sampleValue.getType());
              int64_t numElements = computeTensorElementCount(sampleTy);
              if (numElements < 0) {
                sampleOp.emitError(
                    "Impulse: dynamic tensor dimensions not supported");
                return WalkResult::interrupt();
              }
              auto flatSampleType = RankedTensorType::get(
                  {1, numElements}, sampleTy.getElementType());
              auto flatSample = impulse::ReshapeOp::create(
                  rewriter, sampleOp.getLoc(), flatSampleType, sampleValue);
              auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
              auto row0 = arith::ConstantOp::create(
                  rewriter, sampleOp.getLoc(), i64S,
                  DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(0)));
              auto colOff = arith::ConstantOp::create(
                  rewriter, sampleOp.getLoc(), i64S,
                  DenseElementsAttr::get(
                      i64S, rewriter.getI64IntegerAttr(currentOffset)));
              currTrace = impulse::DynamicUpdateSliceOp::create(
                              rewriter, sampleOp.getLoc(), traceType, currTrace,
                              flatSample, ValueRange{row0, colOff})
                              .getResult();
              currentOffset += numElements;
            }
          }

          sampleOp.replaceAllUsesWith(sampledValues);
          toErase.push_back(sampleOp);
          return WalkResult::advance();
        }

        bool isDistribution = static_cast<bool>(sampleOp.getLogpdfAttr());

        if (isDistribution) {
          // A1. Distribution function: call the distribution function.
          auto distFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getFnAttr()));

          auto distCall = func::CallOp::create(
              rewriter, sampleOp.getLoc(), distFn.getName(),
              distFn.getResultTypes(), sampleOp.getInputs());

          sampledValues.append(distCall.getResults().begin(),
                               distCall.getResults().end());

          auto logpdfFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getLogpdfAttr()));

          // logpdf operands: (<non-RNG outputs>..., <non-RNG inputs>...)
          SmallVector<Value> logpdfOperands;
          for (unsigned i = 1; i < sampledValues.size(); ++i) {
            logpdfOperands.push_back(sampledValues[i]);
          }
          for (unsigned i = 1; i < sampleOp.getNumOperands(); ++i) {
            logpdfOperands.push_back(sampleOp.getOperand(i));
          }

          if (logpdfOperands.size() != logpdfFn.getNumArguments()) {
            sampleOp.emitError("Impulse: failed to construct logpdf call; "
                               "logpdf function has wrong number of arguments");
            return WalkResult::interrupt();
          }

          // A2. Compute and accumulate weight.
          auto logpdf = func::CallOp::create(
              rewriter, sampleOp.getLoc(), logpdfFn.getName(),
              logpdfFn.getResultTypes(), logpdfOperands);
          weightAccumulator =
              arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                    weightAccumulator, logpdf.getResult(0));
          if (scoreTailSize > 0)
            siteScores.push_back(logpdf.getResult(0));

          // A3. Check if this sample is in the selection and insert into trace
          bool inSelection = false;
          for (auto addr : selection) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              inSelection = true;
              break;
            }
          }

          if (inSelection) {
            for (unsigned i = 1; i < sampledValues.size(); ++i) {
              auto sampleValue = sampledValues[i];
              auto sampleType = cast<RankedTensorType>(sampleValue.getType());
              int64_t numElements = computeTensorElementCount(sampleType);
              if (numElements < 0) {
                sampleOp.emitError(
                    "Impulse: dynamic tensor dimensions not supported");
                return WalkResult::interrupt();
              }

              auto flatSampleType = RankedTensorType::get(
                  {1, numElements}, sampleType.getElementType());
              auto flatSample = impulse::ReshapeOp::create(
                  rewriter, sampleOp.getLoc(), flatSampleType, sampleValue);
              auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
              auto row0 = arith::ConstantOp::create(
                  rewriter, sampleOp.getLoc(), i64S,
                  DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(0)));
              auto colOff = arith::ConstantOp::create(
                  rewriter, sampleOp.getLoc(), i64S,
                  DenseElementsAttr::get(
                      i64S, rewriter.getI64IntegerAttr(currentOffset)));
              currTrace = impulse::DynamicUpdateSliceOp::create(
                              rewriter, sampleOp.getLoc(), traceType, currTrace,
                              flatSample, ValueRange{row0, colOff})
                              .getResult();
              currentOffset += numElements;
            }
          }
        } else {
          // B. Generative function: recursively simulate the nested function
          if (scoreTailSize > 0) {
            sampleOp.emitError("Impulse: score-tail traces do not support "
                               "nested generative sites");
            return WalkResult::interrupt();
          }
          auto genFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getFnAttr()));

          if (genFn.getFunctionBody().empty()) {
            sampleOp.emitError(
                "Impulse: generative function body is empty; "
                "if this is a distribution, add a logpdf attribute");
            return WalkResult::interrupt();
          }

          ArrayAttr subSelection =
              buildSubSelection(rewriter, selection, sampleOp.getSymbolAttr());
          if (subSelection.empty()) {
            // No samples from this generative function are in the selection
            // Just call the function directly
            auto genCall = func::CallOp::create(
                rewriter, sampleOp.getLoc(), genFn.getName(),
                genFn.getResultTypes(), sampleOp.getInputs());
            sampledValues.append(genCall.getResults().begin(),
                                 genCall.getResults().end());
          } else {
            int64_t subPositionSize = computePositionSizeForSelection(
                sampleOp, genFn, subSelection, symbolTable);
            if (subPositionSize <= 0) {
              sampleOp.emitError(
                  "Impulse: failed to compute sub-position size");
              return WalkResult::interrupt();
            }

            // Build result types: (trace, weight, original_returns...)
            auto subTraceType = RankedTensorType::get({1, subPositionSize},
                                                      rewriter.getF64Type());
            auto scalarTy = RankedTensorType::get({}, rewriter.getF64Type());
            SmallVector<Type> simResultTypes;
            simResultTypes.push_back(subTraceType);
            simResultTypes.push_back(scalarTy);
            for (auto t : genFn.getResultTypes())
              simResultTypes.push_back(t);

            auto nestedSimulate = impulse::SimulateOp::create(
                rewriter, sampleOp.getLoc(), simResultTypes,
                sampleOp.getFnAttr(), sampleOp.getInputs(), subSelection);
            auto subTrace = nestedSimulate.getTrace();
            auto subWeight = nestedSimulate.getWeight();

            weightAccumulator = arith::AddFOp::create(
                rewriter, sampleOp.getLoc(), weightAccumulator, subWeight);

            int64_t mergeOffset = computeOffsetForNestedSample(
                sampleOp, fn, selection, sampleOp.getSymbolAttr(), symbolTable);
            if (mergeOffset < 0) {
              sampleOp.emitError("Impulse: failed to compute merge offset");
              return WalkResult::interrupt();
            }

            auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
            auto row0 = arith::ConstantOp::create(
                rewriter, sampleOp.getLoc(), i64S,
                DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(0)));
            auto colOff = arith::ConstantOp::create(
                rewriter, sampleOp.getLoc(), i64S,
                DenseElementsAttr::get(
                    i64S, rewriter.getI64IntegerAttr(mergeOffset)));
            currTrace = impulse::DynamicUpdateSliceOp::create(
                            rewriter, sampleOp.getLoc(), traceType, currTrace,
                            subTrace, ValueRange{row0, colOff})
                            .getResult();
            currentOffset =
                std::max(currentOffset, mergeOffset + subPositionSize);

            for (auto output : nestedSimulate.getOutputs())
              sampledValues.push_back(output);
          }
        }

        // D. Replace uses of the original sample op with the new values.
        sampleOp.replaceAllUsesWith(sampledValues);

        toErase.push_back(sampleOp);
        return WalkResult::advance();
      };

      auto result =
          processImpulseBlock(NewF.getFunctionBody().front(), rewriter, carry,
                              traceType, processSample, toErase);
      currTrace = carry.currTrace;
      weightAccumulator = carry.weight;
      currentOffset = carry.offset;

      for (Operation *op : toErase)
        rewriter.eraseOp(op);

      if (failed(result)) {
        CI.emitError("Impulse: failed to walk sample ops");
        return failure();
      }

      // E. Rewrite the return to return (trace, weight, <original returns>...)
      NewF.walk([&](func::ReturnOp retOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(retOp);
        SmallVector<Value> newRetVals;
        Value outTrace = currTrace;
        if (scoreTailSize > 0) {
          auto fullType =
              RankedTensorType::get({1, declaredWidth}, rewriter.getF64Type());
          auto zeroFull =
              arith::ConstantOp::create(rewriter, retOp.getLoc(), fullType,
                                        DenseElementsAttr::get(fullType, 0.0));
          auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
          auto c0 = arith::ConstantOp::create(
              rewriter, retOp.getLoc(), i64S,
              DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(0)));
          Value acc = impulse::DynamicUpdateSliceOp::create(
                          rewriter, retOp.getLoc(), fullType, zeroFull,
                          currTrace, ValueRange{c0, c0})
                          .getResult();
          for (auto [i, s] : llvm::enumerate(siteScores)) {
            auto s11 = impulse::ReshapeOp::create(
                rewriter, retOp.getLoc(),
                RankedTensorType::get({1, 1}, rewriter.getF64Type()), s);
            auto cOff = arith::ConstantOp::create(
                rewriter, retOp.getLoc(), i64S,
                DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(
                                                 positionSize + (int64_t)i)));
            acc = impulse::DynamicUpdateSliceOp::create(
                      rewriter, retOp.getLoc(), fullType, acc, s11,
                      ValueRange{c0, cOff})
                      .getResult();
          }
          outTrace = acc;
        }
        newRetVals.push_back(outTrace);
        newRetVals.push_back(weightAccumulator);
        newRetVals.append(retOp.getOperands().begin(),
                          retOp.getOperands().end());

        func::ReturnOp::create(rewriter, retOp.getLoc(), newRetVals);
        rewriter.eraseOp(retOp);
      });

      rewriter.setInsertionPoint(CI);
      auto newCI = func::CallOp::create(rewriter, CI.getLoc(), NewF.getName(),
                                        NewF.getResultTypes(), CI.getInputs());

      rewriter.replaceOp(CI, newCI.getResults());

      delete putils;

      return success();
    }
  };

  template <typename MCMCOpT>
  struct LowerMCMCPattern : public mlir::OpRewritePattern<MCMCOpT> {
    static constexpr bool isRegionForm =
        std::is_same_v<MCMCOpT, impulse::InferRegionOp>;

    bool debugDump;

    LowerMCMCPattern(MLIRContext *context, bool debugDump,
                     PatternBenefit benefit = 1)
        : mlir::OpRewritePattern<MCMCOpT>(context, benefit),
          debugDump(debugDump) {}

    LogicalResult matchAndRewrite(MCMCOpT mcmcOp,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      bool hasLogpdfFn = static_cast<bool>(mcmcOp.getLogpdfFnAttr());

      if constexpr (isRegionForm) {
        if (mcmcOp.getLogpdf().empty()) {
          mcmcOp.emitError("Impulse: infer_region requires a constructed "
                           "logpdf region (run sicm first)");
          return failure();
        }
        if (!hasLogpdfFn && mcmcOp.getSampler().empty()) {
          mcmcOp.emitError("Impulse: model-mode infer_region requires a "
                           "sampler region");
          return failure();
        }
      } else if (!hasLogpdfFn) {
        auto fnAttr = mcmcOp.getFnAttr();
        if (!fnAttr) {
          mcmcOp.emitError("Impulse: either fn or logpdf_fn must be provided");
          return failure();
        }
        auto fn = cast<FunctionOpInterface>(
            symbolTable.lookupNearestSymbolFrom(mcmcOp, fnAttr));
        if (fn.getFunctionBody().empty()) {
          mcmcOp.emitError("Impulse: calling `mcmc` on an empty function");
          return failure();
        }
      }

      // Other configs fall through to their own patterns; do not error here.
      bool isHMC = mcmcOp.getHmcConfig().has_value();
      bool isNUTS = mcmcOp.getNutsConfig().has_value();
      if (!isHMC && !isNUTS)
        return rewriter.notifyMatchFailure(
            mcmcOp, "no hmc_config/nuts_config; not an HMC/NUTS op");

      if (!mcmcOp.getInitStepSize()) {
        mcmcOp.emitError("Impulse: MCMC requires init_step_size parameter");
        return failure();
      }

      auto loc = mcmcOp.getLoc();
      Value invMass = mcmcOp.getInitInverseMassMatrix();
      Value stepSize = mcmcOp.getInitStepSize();

      auto inputs = mcmcOp.getInputs();
      if (inputs.empty()) {
        mcmcOp.emitError("Impulse: MCMC requires at least rng_state input");
        return failure();
      }

      auto rngInput = inputs[0];

      // Unconstrained and constrained widths; they differ only for SIMPLEX.
      int64_t positionSize;
      int64_t constrainedPositionSize;
      SmallVector<Value> fnInputs;
      SmallVector<Type> fnResultTypes;
      Value originalTrace;
      ArrayAttr selection, allAddresses;
      SmallVector<SupportInfo> supports;
      FlatSymbolRefAttr logpdfFnAttr;

      auto sizesFromSupports = [&](int64_t constrainedFallback) {
        constrainedPositionSize = 0;
        positionSize = 0;
        for (const auto &info : supports) {
          constrainedPositionSize += info.size;
          positionSize += info.unconstrainedSize;
        }
        // Nested addresses get no support info; use the constrained width.
        if (constrainedPositionSize != constrainedFallback) {
          constrainedPositionSize = constrainedFallback;
          positionSize = constrainedFallback;
        }
      };

      if (hasLogpdfFn) {
        logpdfFnAttr = mcmcOp.getLogpdfFnAttr();
        fnInputs.assign(inputs.begin() + 1, inputs.end());
        auto initialPos = mcmcOp.getInitPosition();
        auto initPosType = cast<RankedTensorType>(initialPos.getType());
        // `init_position` is `1xN` or chain-major `Cx1xN`; N is per chain.
        positionSize = initPosType.getNumElements() / mcmcOp.getNumChains();
        constrainedPositionSize = positionSize;
        selection = mcmcOp.getSelectionAttr();
        allAddresses = mcmcOp.getAllAddressesAttr();
      } else if constexpr (isRegionForm) {
        fnInputs.assign(inputs.begin() + 1, inputs.end());
        originalTrace = mcmcOp.getOriginalTrace();
        selection = mcmcOp.getSelectionAttr();
        allAddresses = mcmcOp.getAllAddressesAttr();

        supports = collectSupportInfoForRegion(mcmcOp);
        if (supports.empty()) {
          mcmcOp.emitError("Impulse: failed to collect per-site support info "
                           "from the infer_region sampler");
          return failure();
        }
        int64_t constrainedFallback = 0;
        for (const auto &info : supports)
          constrainedFallback += info.size;
        sizesFromSupports(constrainedFallback);
        if (positionSize <= 0)
          return failure();
      } else {
        fnInputs.assign(inputs.begin() + 1, inputs.end());
        originalTrace = mcmcOp.getOriginalTrace();
        selection = mcmcOp.getSelectionAttr();
        allAddresses = mcmcOp.getAllAddressesAttr();

        auto fn = cast<FunctionOpInterface>(
            symbolTable.lookupNearestSymbolFrom(mcmcOp, mcmcOp.getFnAttr()));
        int64_t constrainedFallback =
            computePositionSizeForSelection(mcmcOp, fn, selection, symbolTable);
        if (constrainedFallback <= 0)
          return failure();

        supports = collectSupportInfoForSelection(
            mcmcOp, fn, selection, allAddresses, symbolTable,
            /*positionDeclarationOrder=*/false);
        sizesFromSupports(constrainedFallback);

        auto fnType = cast<FunctionType>(fn.getFunctionType());
        fnResultTypes.assign(fnType.getResults().begin(),
                             fnType.getResults().end());
      }

      int64_t numSamples = mcmcOp.getNumSamples();
      int64_t thinning = mcmcOp.getThinning();
      int64_t numWarmup = mcmcOp.getNumWarmup();

      int64_t totalWarmup = numWarmup;
      if (auto tw = mcmcOp.getTotalWarmup())
        totalWarmup = *tw;

      ArrayAttr schema = mcmcOp.getStateSchemaAttr();
      bool isResume = !mcmcOp.getStateIn().empty();
      int64_t numChains = mcmcOp.getNumChains();
      bool exposeAdaptation =
          impulse::slotIndex(schema, state_slot::kWindowIdx).has_value();

      // Schema order matches DualAveragingState/WelfordState::toValues().
      SmallVector<Value> adaptationInValsBase;
      if (auto daIdx = impulse::slotIndex(schema, state_slot::kDaLogStepSize)) {
        auto stateIn = mcmcOp.getStateIn();
        for (unsigned i = 0; i < 9; ++i)
          adaptationInValsBase.push_back(
              *daIdx + i < stateIn.size() ? stateIn[*daIdx + i] : Value());
      }
      bool hasAdaptationInBase = isResume && !adaptationInValsBase.empty() &&
                                 adaptationInValsBase.front() != nullptr;

      // Multi-chain slots are chain-major; runOneChain reads them per chain.
      if (isResume && numChains <= 1) {
        if (Value s = stateInSlot(mcmcOp, state_slot::kStepSize))
          stepSize = s;
        if (Value m = stateInSlot(mcmcOp, state_slot::kInverseMassMatrix))
          invMass = m;
      }

      auto elemType =
          cast<RankedTensorType>(stepSize.getType()).getElementType();
      auto positionType = RankedTensorType::get({1, positionSize}, elemType);
      auto scalarType = RankedTensorType::get({}, elemType);
      auto i64TensorType = RankedTensorType::get({}, rewriter.getI64Type());
      auto i1TensorType = RankedTensorType::get({}, rewriter.getI1Type());

      // Algorithm-specific configuration
      Value trajectoryLength;
      Value maxDeltaEnergy;
      int64_t maxTreeDepth = 0;

      bool adaptStepSize = false;
      bool adaptMassMatrix = false;
      int64_t fixedNumSteps = 0;
      auto F64TensorType = RankedTensorType::get({}, rewriter.getF64Type());
      if (isHMC) {
        auto hmcConfig = mcmcOp.getHmcConfig().value();
        fixedNumSteps = hmcConfig.getNumSteps();
        double length = hmcConfig.getTrajectoryLength().getValueAsDouble();
        trajectoryLength = arith::ConstantOp::create(
            rewriter, loc, F64TensorType,
            DenseElementsAttr::get(F64TensorType,
                                   rewriter.getF64FloatAttr(length)));
        adaptStepSize = hmcConfig.getAdaptStepSize();
        adaptMassMatrix = hmcConfig.getAdaptMassMatrix();
      } else {
        auto nutsConfig = mcmcOp.getNutsConfig().value();
        maxTreeDepth = nutsConfig.getMaxTreeDepth();
        adaptStepSize = nutsConfig.getAdaptStepSize();
        adaptMassMatrix = nutsConfig.getAdaptMassMatrix();
        double maxDeltaEnergyVal =
            nutsConfig.getMaxDeltaEnergy()
                ? nutsConfig.getMaxDeltaEnergy().getValueAsDouble()
                : 1000.0;
        maxDeltaEnergy = arith::ConstantOp::create(
            rewriter, loc, F64TensorType,
            DenseElementsAttr::get(
                F64TensorType, rewriter.getF64FloatAttr(maxDeltaEnergyVal)));
      }

      bool diagonal = true;
      if (invMass) {
        // tensor<N> and tensor<1xN> are diagonal, as in buildStateSchema.
        auto invMassType = cast<RankedTensorType>(invMass.getType());
        diagonal =
            invMassType.getRank() == 1 ||
            (invMassType.getRank() == 2 && invMassType.getShape()[0] == 1);
      }

      auto autodiffAttrs = mcmcOp.getAutodiffAttrsAttr();

      auto makeHMCContext = [&](Value currentInvMass,
                                Value currentMassMatrixSqrt,
                                Value currentStepSize) -> HMCContext {
        if (hasLogpdfFn) {
          HMCContext ctx(logpdfFnAttr, fnInputs, currentInvMass,
                         currentMassMatrixSqrt, currentStepSize,
                         trajectoryLength, positionSize, autodiffAttrs);
          ctx.numSteps = fixedNumSteps;
          if constexpr (isRegionForm)
            ctx.logpdfRegion = &mcmcOp.getLogpdf();
          return ctx;
        } else {
          HMCContext ctx(mcmcOp.getFnAttr(), fnInputs, fnResultTypes,
                         originalTrace, selection, allAddresses, currentInvMass,
                         currentMassMatrixSqrt, currentStepSize,
                         trajectoryLength, positionSize, supports,
                         autodiffAttrs);
          ctx.numSteps = fixedNumSteps;
          if constexpr (isRegionForm)
            ctx.logpdfRegion = &mcmcOp.getLogpdf();
          return ctx;
        }
      };

      auto makeNUTSContext =
          [&](Value currentInvMass, Value currentMassMatrixSqrt,
              Value currentStepSize, Value U) -> NUTSContext {
        if (hasLogpdfFn) {
          NUTSContext ctx(logpdfFnAttr, fnInputs, currentInvMass,
                          currentMassMatrixSqrt, currentStepSize, positionSize,
                          U, maxDeltaEnergy, maxTreeDepth, autodiffAttrs);
          if constexpr (isRegionForm)
            ctx.logpdfRegion = &mcmcOp.getLogpdf();
          return ctx;
        } else {
          NUTSContext ctx(mcmcOp.getFnAttr(), fnInputs, fnResultTypes,
                          originalTrace, selection, allAddresses,
                          currentInvMass, currentMassMatrixSqrt,
                          currentStepSize, positionSize, supports, U,
                          maxDeltaEnergy, maxTreeDepth, autodiffAttrs);
          if constexpr (isRegionForm)
            ctx.logpdfRegion = &mcmcOp.getLogpdf();
          return ctx;
        }
      };

      auto sliceChainDim = [&](Value v, int64_t c) -> Value {
        auto ty = cast<RankedTensorType>(v.getType());
        ArrayRef<int64_t> shape = ty.getShape();
        SmallVector<int64_t> start(shape.size(), 0),
            limit(shape.begin(), shape.end()), stride(shape.size(), 1);
        start[0] = c;
        limit[0] = c + 1;
        SmallVector<int64_t> oneShape(shape.begin(), shape.end());
        oneShape[0] = 1;
        auto oneType = RankedTensorType::get(oneShape, ty.getElementType());
        Value sl = impulse::SliceOp::create(
            rewriter, loc, oneType, v, rewriter.getDenseI64ArrayAttr(start),
            rewriter.getDenseI64ArrayAttr(limit),
            rewriter.getDenseI64ArrayAttr(stride));
        auto dropType =
            RankedTensorType::get(shape.drop_front(), ty.getElementType());
        return impulse::ReshapeOp::create(rewriter, loc, dropType, sl);
      };

      // All rng-dependent work lives here; it is replayed per chain.
      auto runOneChain = [&](Value chainRng,
                             int64_t chainIdx) -> SmallVector<Value> {
        auto chainStateInSlot = [&](StringRef name) -> Value {
          Value v = stateInSlot(mcmcOp, name);
          if (v && numChains > 1)
            v = sliceChainDim(v, chainIdx);
          return v;
        };
        Value chainStepSize = stepSize;
        Value chainInvMass = invMass;
        SmallVector<Value> adaptationInVals = adaptationInValsBase;
        bool hasAdaptationIn = hasAdaptationInBase;
        if (isResume && numChains > 1) {
          if (Value s = stateInSlot(mcmcOp, state_slot::kStepSize))
            chainStepSize = sliceChainDim(s, chainIdx);
          if (Value m = stateInSlot(mcmcOp, state_slot::kInverseMassMatrix))
            chainInvMass = sliceChainDim(m, chainIdx);
          for (Value &av : adaptationInVals)
            if (av)
              av = sliceChainDim(av, chainIdx);
        }

        Value adaptedInvMass = chainInvMass;
        auto adaptedMassMatrixSqrt =
            computeMassMatrixSqrt(rewriter, loc, adaptedInvMass, positionType);

        Value currentQ, currentGrad, currentU, currentRng;

        Value resumePosition = chainStateInSlot(state_slot::kPosition);
        Value resumeGrad = chainStateInSlot(state_slot::kGradient);
        Value resumePE = chainStateInSlot(state_slot::kPotentialEnergy);

        if (isResume && resumePosition && resumeGrad && resumePE) {
          currentQ = resumePosition;
          currentGrad = resumeGrad;
          currentU = resumePE;
          currentRng = chainRng;
        } else {
          auto baseCtx = makeHMCContext(adaptedInvMass, adaptedMassMatrixSqrt,
                                        chainStepSize);
          Value chainInitPos;
          if (hasLogpdfFn) {
            chainInitPos = mcmcOp.getInitPosition();
            if (numChains > 1)
              chainInitPos = sliceChainDim(chainInitPos, chainIdx);
          }
          auto initState = InitHMC(rewriter, loc, chainRng, baseCtx,
                                   chainInitPos, debugDump);
          currentQ = initState.q0;
          currentGrad = initState.grad0;
          currentU = initState.U0;
          currentRng = initState.rng;
        }

        auto runSampleStepWithStepSize =
            [&](OpBuilder &builder, Location loc, Value q, Value grad, Value U,
                Value rng, Value currentStepSize) -> MCMCKernelResult {
          if (isHMC) {
            auto ctx = makeHMCContext(adaptedInvMass, adaptedMassMatrixSqrt,
                                      currentStepSize);
            return SampleHMC(builder, loc, q, grad, U, rng, ctx, debugDump);
          } else {
            auto nutsCtx = makeNUTSContext(
                adaptedInvMass, adaptedMassMatrixSqrt, currentStepSize, U);
            return SampleNUTS(builder, loc, q, grad, U, rng, nutsCtx,
                              debugDump);
          }
        };

        Value adaptedStepSize = chainStepSize;

        auto runSampleStepWithInvMass =
            [&](OpBuilder &builder, Location loc, Value q, Value grad, Value U,
                Value rng, Value currentStepSize, Value currentInvMass,
                Value currentMassMatrixSqrt) -> MCMCKernelResult {
          if (isHMC) {
            auto ctx = makeHMCContext(currentInvMass, currentMassMatrixSqrt,
                                      currentStepSize);
            return SampleHMC(builder, loc, q, grad, U, rng, ctx, debugDump);
          } else {
            auto nutsCtx = makeNUTSContext(
                currentInvMass, currentMassMatrixSqrt, currentStepSize, U);
            return SampleNUTS(builder, loc, q, grad, U, rng, nutsCtx,
                              debugDump);
          }
        };

        if (!adaptedInvMass) {
          adaptedInvMass = arith::ConstantOp::create(
              rewriter, loc, positionType,
              DenseElementsAttr::get(positionType,
                                     rewriter.getFloatAttr(elemType, 1.0)));
          adaptedMassMatrixSqrt = arith::ConstantOp::create(
              rewriter, loc, positionType,
              DenseElementsAttr::get(positionType,
                                     rewriter.getFloatAttr(elemType, 1.0)));
        }

        SmallVector<Value> finalAdaptationState;
        auto packAdaptation = [&](const DualAveragingState &da,
                                  const WelfordState &w,
                                  Value widx) -> SmallVector<Value> {
          SmallVector<Value> v;
          for (Value x : da.toValues())
            v.push_back(x);
          v.push_back(w.mean);
          v.push_back(w.m2);
          v.push_back(w.n);
          v.push_back(widx);
          return v;
        };

        if (numWarmup > 0) {
          auto c0 = arith::ConstantOp::create(
              rewriter, loc, i64TensorType,
              DenseElementsAttr::get(i64TensorType,
                                     rewriter.getI64IntegerAttr(0)));
          auto c1 = arith::ConstantOp::create(
              rewriter, loc, i64TensorType,
              DenseElementsAttr::get(i64TensorType,
                                     rewriter.getI64IntegerAttr(1)));
          auto numWarmupConst = arith::ConstantOp::create(
              rewriter, loc, i64TensorType,
              DenseElementsAttr::get(i64TensorType,
                                     rewriter.getI64IntegerAttr(numWarmup)));

          auto schedule = buildAdaptationSchedule(totalWarmup);
          int64_t numWindows = static_cast<int64_t>(schedule.size());

          SmallVector<Value> windowEndConstants;
          for (const auto &window : schedule) {
            windowEndConstants.push_back(arith::ConstantOp::create(
                rewriter, loc, i64TensorType,
                DenseElementsAttr::get(
                    i64TensorType, rewriter.getI64IntegerAttr(window.end))));
          }

          auto numWindowsMinusOne = arith::ConstantOp::create(
              rewriter, loc, i64TensorType,
              DenseElementsAttr::get(
                  i64TensorType, rewriter.getI64IntegerAttr(numWindows - 1)));
          auto lastIterConst = arith::ConstantOp::create(
              rewriter, loc, i64TensorType,
              DenseElementsAttr::get(
                  i64TensorType, rewriter.getI64IntegerAttr(totalWarmup - 1)));

          if (!adaptedInvMass) {
            adaptedInvMass = arith::ConstantOp::create(
                rewriter, loc, positionType,
                DenseElementsAttr::get(positionType,
                                       rewriter.getFloatAttr(elemType, 1.0)));
            adaptedMassMatrixSqrt = arith::ConstantOp::create(
                rewriter, loc, positionType,
                DenseElementsAttr::get(positionType,
                                       rewriter.getFloatAttr(elemType, 1.0)));
          }

          Value initialStepSize = chainStepSize;
          initialStepSize = conditionalDump(
              rewriter, loc, initialStepSize,
              "MCMC: initial step size before warmup", debugDump);
          DualAveragingState daState =
              hasAdaptationIn
                  ? DualAveragingState::fromValues(
                        ArrayRef<Value>(adaptationInVals).take_front(5))
                  : initDualAveraging(rewriter, loc, initialStepSize,
                                      /*logOfProduct=*/true);

          WelfordState welfordState;
          WelfordConfig welfordConfig;
          if (adaptMassMatrix) {
            welfordState =
                hasAdaptationIn
                    ? WelfordState::fromValues(
                          ArrayRef<Value>(adaptationInVals).slice(5, 3))
                    : initWelford(rewriter, loc, positionSize, diagonal);
            welfordConfig.diagonal = diagonal;
            welfordConfig.regularize = true;
          }

          Value windowIdx =
              hasAdaptationIn
                  ? adaptationInVals[8]
                  : arith::ConstantOp::create(
                        rewriter, loc, i64TensorType,
                        DenseElementsAttr::get(i64TensorType,
                                               rewriter.getI64IntegerAttr(0)));

          // Warmup loop carries by default:
          // [q, grad, U, rng, stepSize, invMass, massMatrixSqrt, daState(5),
          // welfordState(3)?, windowIdx]
          SmallVector<Type> warmupLoopTypes = {positionType,
                                               positionType,
                                               scalarType,
                                               currentRng.getType(),
                                               scalarType, // stepSize
                                               adaptedInvMass.getType(),
                                               adaptedMassMatrixSqrt.getType()};
          for (Type t : daState.getTypes())
            warmupLoopTypes.push_back(t);
          if (adaptMassMatrix) {
            for (Type t : welfordState.getTypes())
              warmupLoopTypes.push_back(t);
          }
          warmupLoopTypes.push_back(i64TensorType); // windowIdx

          SmallVector<Value> warmupInitArgs = {currentQ,
                                               currentGrad,
                                               currentU,
                                               currentRng,
                                               initialStepSize,
                                               adaptedInvMass,
                                               adaptedMassMatrixSqrt};
          for (Value v : daState.toValues())
            warmupInitArgs.push_back(v);
          if (adaptMassMatrix) {
            for (Value v : welfordState.toValues())
              warmupInitArgs.push_back(v);
          }
          warmupInitArgs.push_back(windowIdx);

          auto warmupLoop =
              impulse::ForOp::create(rewriter, loc, warmupLoopTypes, c0,
                                     numWarmupConst, c1, warmupInitArgs);

          Block *warmupBody = rewriter.createBlock(&warmupLoop.getRegion());
          warmupBody->addArgument(i64TensorType, loc); // iteration index t
          for (Type t : warmupLoopTypes)
            warmupBody->addArgument(t, loc);

          rewriter.setInsertionPointToStart(warmupBody);

          Value iterT = warmupBody->getArgument(0);
          Value iterTGlobal = iterT;
          if (Value warmupOffset = chainStateInSlot(state_slot::kWarmupOffset))
            iterTGlobal =
                arith::AddIOp::create(rewriter, loc, warmupOffset, iterT);
          Value qLoop = warmupBody->getArgument(1);
          Value gradLoop = warmupBody->getArgument(2);
          Value ULoop = warmupBody->getArgument(3);
          Value rngLoop = warmupBody->getArgument(4);
          Value stepSizeLoop = warmupBody->getArgument(5);
          Value invMassLoop = warmupBody->getArgument(6);
          Value massMatrixSqrtLoop = warmupBody->getArgument(7);

          SmallVector<Value> daStateLoopValues;
          for (int i = 0; i < 5; ++i)
            daStateLoopValues.push_back(warmupBody->getArgument(8 + i));
          auto daStateLoop = DualAveragingState::fromValues(daStateLoopValues);

          WelfordState welfordStateLoop;
          Value windowIdxLoop;
          if (adaptMassMatrix) {
            SmallVector<Value> welfordStateLoopValues;
            for (int i = 0; i < 3; ++i)
              welfordStateLoopValues.push_back(warmupBody->getArgument(13 + i));
            welfordStateLoop = WelfordState::fromValues(welfordStateLoopValues);
            windowIdxLoop = warmupBody->getArgument(16);
          } else {
            windowIdxLoop = warmupBody->getArgument(13);
          }

          auto sample = runSampleStepWithInvMass(
              rewriter, loc, qLoop, gradLoop, ULoop, rngLoop, stepSizeLoop,
              invMassLoop, massMatrixSqrtLoop);

          // Update dual averaging state
          DualAveragingConfig daConfig;
          DualAveragingState updatedDaState;
          Value currentStepSizeFromDA;
          Value finalStepSizeFromDA;

          if (adaptStepSize) {
            updatedDaState = updateDualAveraging(rewriter, loc, daStateLoop,
                                                 sample.accept_prob, daConfig);
            currentStepSizeFromDA =
                getStepSizeFromDualAveraging(rewriter, loc, updatedDaState);
            finalStepSizeFromDA = getStepSizeFromDualAveraging(
                rewriter, loc, updatedDaState, true);
          } else {
            updatedDaState = daStateLoop;
            currentStepSizeFromDA = stepSizeLoop;
            finalStepSizeFromDA = stepSizeLoop;
          }

          // Use log_step_size_avg at last iteration
          auto isLastIter =
              arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                    iterTGlobal, lastIterConst);
          Value adaptedStepSizeInLoop = impulse::SelectOp::create(
              rewriter, loc, scalarType, isLastIter, finalStepSizeFromDA,
              currentStepSizeFromDA);

          const auto &floatSemantics =
              cast<FloatType>(elemType).getFloatSemantics();
          auto tinyConst = arith::ConstantOp::create(
              rewriter, loc, scalarType,
              DenseElementsAttr::get(
                  scalarType,
                  FloatAttr::get(elemType,
                                 llvm::APFloat::getSmallest(floatSemantics))));
          auto maxConst = arith::ConstantOp::create(
              rewriter, loc, scalarType,
              DenseElementsAttr::get(
                  scalarType,
                  FloatAttr::get(elemType,
                                 llvm::APFloat::getLargest(floatSemantics))));
          adaptedStepSizeInLoop = arith::MaximumFOp::create(
              rewriter, loc, adaptedStepSizeInLoop, tinyConst);
          adaptedStepSizeInLoop = arith::MinimumFOp::create(
              rewriter, loc, adaptedStepSizeInLoop, maxConst);

          auto windowIdxGtZero = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::sgt, windowIdxLoop, c0);
          auto windowIdxLtLast =
              arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                    windowIdxLoop, numWindowsMinusOne);
          auto isMiddleWindow = arith::AndIOp::create(
              rewriter, loc, windowIdxGtZero, windowIdxLtLast);

          // Conditionally update Welford
          WelfordState conditionalWelford;
          if (adaptMassMatrix) {
            auto sampleType1D = RankedTensorType::get({positionSize}, elemType);
            Value sample1D = impulse::ReshapeOp::create(rewriter, loc,
                                                        sampleType1D, sample.q);
            WelfordState updatedWelfordAfterSample = updateWelford(
                rewriter, loc, welfordStateLoop, sample1D, welfordConfig);

            conditionalWelford.mean = impulse::SelectOp::create(
                rewriter, loc, welfordStateLoop.mean.getType(), isMiddleWindow,
                updatedWelfordAfterSample.mean, welfordStateLoop.mean);
            conditionalWelford.m2 = impulse::SelectOp::create(
                rewriter, loc, welfordStateLoop.m2.getType(), isMiddleWindow,
                updatedWelfordAfterSample.m2, welfordStateLoop.m2);
            conditionalWelford.n = impulse::SelectOp::create(
                rewriter, loc, welfordStateLoop.n.getType(), isMiddleWindow,
                updatedWelfordAfterSample.n, welfordStateLoop.n);
          }

          Value atWindowEnd = arith::ConstantOp::create(
              rewriter, loc, i1TensorType,
              DenseElementsAttr::get(i1TensorType,
                                     rewriter.getBoolAttr(false)));

          for (int64_t w = 0; w < numWindows; ++w) {
            auto windowIdxIsW = arith::CmpIOp::create(
                rewriter, loc, arith::CmpIPredicate::eq, windowIdxLoop,
                arith::ConstantOp::create(
                    rewriter, loc, i64TensorType,
                    DenseElementsAttr::get(i64TensorType,
                                           rewriter.getI64IntegerAttr(w))));
            auto tEqualsWindowEnd =
                arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      iterTGlobal, windowEndConstants[w]);
            auto matchesThisWindow = arith::AndIOp::create(
                rewriter, loc, windowIdxIsW, tEqualsWindowEnd);
            atWindowEnd = arith::OrIOp::create(rewriter, loc, atWindowEnd,
                                               matchesThisWindow);
          }

          Value newWindowIdx =
              arith::AddIOp::create(rewriter, loc, windowIdxLoop, c1);
          Value windowIdxAfterIncrement = impulse::SelectOp::create(
              rewriter, loc, i64TensorType, atWindowEnd, newWindowIdx,
              windowIdxLoop);

          auto atMiddleWindowEnd =
              arith::AndIOp::create(rewriter, loc, atWindowEnd, isMiddleWindow);

          Value finalInvMass;
          Value finalMassMatrixSqrt;
          WelfordState finalWelfordState;
          Value finalStepSizeValue;
          DualAveragingState finalDaState;

          SmallVector<Type> ifResultTypes;
          ifResultTypes.push_back(invMassLoop.getType());
          ifResultTypes.push_back(massMatrixSqrtLoop.getType());
          if (adaptMassMatrix) {
            ifResultTypes.push_back(conditionalWelford.mean.getType());
            ifResultTypes.push_back(conditionalWelford.m2.getType());
            ifResultTypes.push_back(conditionalWelford.n.getType());
          }
          for (Type t : updatedDaState.getTypes())
            ifResultTypes.push_back(t);

          auto ifOp = impulse::IfOp::create(rewriter, loc, ifResultTypes,
                                            atMiddleWindowEnd);

          {
            Block *trueBranch = rewriter.createBlock(&ifOp.getTrueBranch());
            rewriter.setInsertionPointToStart(trueBranch);

            SmallVector<Value> trueYieldValues;

            if (adaptMassMatrix) {
              Value newInvMass = finalizeWelford(
                  rewriter, loc, conditionalWelford, welfordConfig);
              Value newMassMatrixSqrt = computeMassMatrixSqrt(
                  rewriter, loc, newInvMass, positionType);
              // finalizeWelford returns rank-1 diagonals; match the carried
              // type.
              if (newInvMass.getType() != invMassLoop.getType())
                newInvMass = impulse::ReshapeOp::create(
                    rewriter, loc, invMassLoop.getType(), newInvMass);
              if (newMassMatrixSqrt.getType() != massMatrixSqrtLoop.getType())
                newMassMatrixSqrt = impulse::ReshapeOp::create(
                    rewriter, loc, massMatrixSqrtLoop.getType(),
                    newMassMatrixSqrt);
              auto reinitWelford =
                  initWelford(rewriter, loc, positionSize, diagonal);

              trueYieldValues.push_back(newInvMass);
              trueYieldValues.push_back(newMassMatrixSqrt);
              trueYieldValues.push_back(reinitWelford.mean);
              trueYieldValues.push_back(reinitWelford.m2);
              trueYieldValues.push_back(reinitWelford.n);
            } else {
              trueYieldValues.push_back(invMassLoop);
              trueYieldValues.push_back(massMatrixSqrtLoop);
            }

            if (adaptStepSize) {
              auto reinitDaState =
                  initDualAveraging(rewriter, loc, adaptedStepSizeInLoop);
              for (auto v : reinitDaState.toValues())
                trueYieldValues.push_back(v);
            } else {
              for (auto v : updatedDaState.toValues())
                trueYieldValues.push_back(v);
            }

            impulse::YieldOp::create(rewriter, loc, trueYieldValues);
          }

          {
            Block *falseBranch = rewriter.createBlock(&ifOp.getFalseBranch());
            rewriter.setInsertionPointToStart(falseBranch);

            SmallVector<Value> falseYieldValues;
            falseYieldValues.push_back(invMassLoop);
            falseYieldValues.push_back(massMatrixSqrtLoop);
            if (adaptMassMatrix) {
              falseYieldValues.push_back(conditionalWelford.mean);
              falseYieldValues.push_back(conditionalWelford.m2);
              falseYieldValues.push_back(conditionalWelford.n);
            }
            for (auto v : updatedDaState.toValues())
              falseYieldValues.push_back(v);

            impulse::YieldOp::create(rewriter, loc, falseYieldValues);
          }

          rewriter.setInsertionPointAfter(ifOp);

          size_t resultIdx = 0;
          finalInvMass = ifOp.getResult(resultIdx++);
          finalMassMatrixSqrt = ifOp.getResult(resultIdx++);
          if (adaptMassMatrix) {
            finalWelfordState.mean = ifOp.getResult(resultIdx++);
            finalWelfordState.m2 = ifOp.getResult(resultIdx++);
            finalWelfordState.n = ifOp.getResult(resultIdx++);
          }
          finalDaState.log_step_size = ifOp.getResult(resultIdx++);
          finalDaState.log_step_size_avg = ifOp.getResult(resultIdx++);
          finalDaState.gradient_avg = ifOp.getResult(resultIdx++);
          finalDaState.step_count = ifOp.getResult(resultIdx++);
          finalDaState.prox_center = ifOp.getResult(resultIdx++);

          finalStepSizeValue = adaptedStepSizeInLoop;

          SmallVector<Value> warmupYieldValues = {
              sample.q,           sample.grad,  sample.U,           sample.rng,
              finalStepSizeValue, finalInvMass, finalMassMatrixSqrt};
          for (Value v : finalDaState.toValues())
            warmupYieldValues.push_back(v);
          if (adaptMassMatrix) {
            for (Value v : finalWelfordState.toValues())
              warmupYieldValues.push_back(v);
          }
          warmupYieldValues.push_back(windowIdxAfterIncrement);

          impulse::YieldOp::create(rewriter, loc, warmupYieldValues);

          rewriter.setInsertionPointAfter(warmupLoop);

          currentQ = warmupLoop.getResult(0);
          currentGrad = warmupLoop.getResult(1);
          currentU = warmupLoop.getResult(2);
          currentRng = warmupLoop.getResult(3);
          adaptedStepSize = warmupLoop.getResult(4);
          adaptedInvMass = warmupLoop.getResult(5);
          adaptedMassMatrixSqrt = warmupLoop.getResult(6);

          adaptedStepSize = conditionalDump(
              rewriter, loc, adaptedStepSize,
              "MCMC: adapted step size after warmup", debugDump);
          if (adaptMassMatrix) {
            adaptedInvMass = conditionalDump(
                rewriter, loc, adaptedInvMass,
                "MCMC: adapted inverse mass matrix after warmup", debugDump);
          }

          if (exposeAdaptation) {
            DualAveragingState finalDa = DualAveragingState::fromValues(
                {warmupLoop.getResult(7), warmupLoop.getResult(8),
                 warmupLoop.getResult(9), warmupLoop.getResult(10),
                 warmupLoop.getResult(11)});
            WelfordState finalW;
            Value finalWidx;
            if (adaptMassMatrix) {
              finalW = WelfordState::fromValues({warmupLoop.getResult(12),
                                                 warmupLoop.getResult(13),
                                                 warmupLoop.getResult(14)});
              finalWidx = warmupLoop.getResult(15);
            } else {
              finalW = initWelford(rewriter, loc, positionSize, diagonal);
              finalWidx = warmupLoop.getResult(12);
            }
            finalAdaptationState = packAdaptation(finalDa, finalW, finalWidx);
          }
        }

        if (exposeAdaptation && finalAdaptationState.empty()) {
          DualAveragingState da =
              hasAdaptationIn
                  ? DualAveragingState::fromValues(
                        ArrayRef<Value>(adaptationInVals).take_front(5))
                  : initDualAveraging(rewriter, loc, chainStepSize,
                                      /*logOfProduct=*/true);
          WelfordState w =
              hasAdaptationIn
                  ? WelfordState::fromValues(
                        ArrayRef<Value>(adaptationInVals).slice(5, 3))
                  : initWelford(rewriter, loc, positionSize, diagonal);
          Value widx =
              hasAdaptationIn
                  ? adaptationInVals[8]
                  : arith::ConstantOp::create(
                        rewriter, loc, i64TensorType,
                        DenseElementsAttr::get(i64TensorType,
                                               rewriter.getI64IntegerAttr(0)));
          finalAdaptationState = packAdaptation(da, w, widx);
        }

        int64_t collectionSize = numSamples / thinning;
        int64_t startIdx = numSamples % thinning;

        auto samplesBufferType = RankedTensorType::get(
            {collectionSize, constrainedPositionSize}, elemType);
        auto diagnosticsBufferType =
            // Diagnostics buffer has two columns: [is_accepted, is_divergent]
            RankedTensorType::get({collectionSize, 2}, rewriter.getI1Type());
        auto logDensityBufferType =
            RankedTensorType::get({collectionSize}, elemType);

        auto samplesBuffer = arith::ConstantOp::create(
            rewriter, loc, samplesBufferType,
            DenseElementsAttr::get(samplesBufferType,
                                   rewriter.getFloatAttr(elemType, 0.0)));
        auto diagnosticsBuffer = arith::ConstantOp::create(
            rewriter, loc, diagnosticsBufferType,
            DenseElementsAttr::get(diagnosticsBufferType,
                                   rewriter.getBoolAttr(isNUTS)));
        auto logDensityBuffer = arith::ConstantOp::create(
            rewriter, loc, logDensityBufferType,
            DenseElementsAttr::get(logDensityBufferType,
                                   rewriter.getFloatAttr(elemType, 0.0)));

        auto c0 = arith::ConstantOp::create(
            rewriter, loc, i64TensorType,
            DenseElementsAttr::get(i64TensorType,
                                   rewriter.getI64IntegerAttr(0)));
        auto c1 = arith::ConstantOp::create(
            rewriter, loc, i64TensorType,
            DenseElementsAttr::get(i64TensorType,
                                   rewriter.getI64IntegerAttr(1)));
        auto numSamplesConst = arith::ConstantOp::create(
            rewriter, loc, i64TensorType,
            DenseElementsAttr::get(i64TensorType,
                                   rewriter.getI64IntegerAttr(numSamples)));
        auto startIdxConst = arith::ConstantOp::create(
            rewriter, loc, i64TensorType,
            DenseElementsAttr::get(i64TensorType,
                                   rewriter.getI64IntegerAttr(startIdx)));
        auto thinningConst = arith::ConstantOp::create(
            rewriter, loc, i64TensorType,
            DenseElementsAttr::get(i64TensorType,
                                   rewriter.getI64IntegerAttr(thinning)));

        Value finalQ, finalGrad, finalU, finalRng, finalSamplesBuffer,
            finalDiagnosticsBuffer, finalLogDensityBuffer;
        if (collectionSize == 0) {
          finalQ = currentQ;
          finalGrad = currentGrad;
          finalU = currentU;
          finalRng = currentRng;
          finalSamplesBuffer = samplesBuffer;
          finalDiagnosticsBuffer = diagnosticsBuffer;
          finalLogDensityBuffer = logDensityBuffer;
        } else {
          // Loop carries: [q, grad, U, rng, samplesBuffer, diagnosticsBuffer,
          // logDensityBuffer]
          SmallVector<Type> loopResultTypes = {
              positionType,         positionType,      scalarType,
              currentRng.getType(), samplesBufferType, diagnosticsBufferType,
              logDensityBufferType};
          auto forLoopOp = impulse::ForOp::create(
              rewriter, loc, loopResultTypes, c0, numSamplesConst, c1,
              ValueRange{currentQ, currentGrad, currentU, currentRng,
                         samplesBuffer, diagnosticsBuffer, logDensityBuffer});

          Block *loopBody = rewriter.createBlock(&forLoopOp.getRegion());
          loopBody->addArgument(i64TensorType, loc); // i (iteration index)
          loopBody->addArgument(positionType, loc);  // q
          loopBody->addArgument(positionType, loc);  // grad
          loopBody->addArgument(scalarType, loc);    // U
          loopBody->addArgument(currentRng.getType(), loc); // rng
          loopBody->addArgument(samplesBufferType, loc);    // samplesBuffer
          loopBody->addArgument(diagnosticsBufferType,
                                loc);                       // diagnosticsBuffer
          loopBody->addArgument(logDensityBufferType, loc); // logDensityBuffer

          rewriter.setInsertionPointToStart(loopBody);
          Value iterIdx = loopBody->getArgument(0);
          Value qLoop = loopBody->getArgument(1);
          Value gradLoop = loopBody->getArgument(2);
          Value ULoop = loopBody->getArgument(3);
          Value rngLoop = loopBody->getArgument(4);
          Value samplesBufferLoop = loopBody->getArgument(5);
          Value diagnosticsBufferLoop = loopBody->getArgument(6);
          Value logDensityBufferLoop = loopBody->getArgument(7);

          auto sample = runSampleStepWithStepSize(
              rewriter, loc, qLoop, gradLoop, ULoop, rngLoop, adaptedStepSize);
          auto q_constrained =
              impulse::constrainPosition(rewriter, loc, sample.q, supports);

          // Storage index: idx = (i - start_idx) / thinning
          auto iMinusStart =
              arith::SubIOp::create(rewriter, loc, iterIdx, startIdxConst);
          auto storageIdx =
              arith::DivSIOp::create(rewriter, loc, iMinusStart, thinningConst);

          // Store condition:
          // (i >= start_idx) && ((i - start_idx) % thinning == 0)
          auto geStartIdx = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::sge, iterIdx, startIdxConst);
          auto modThinning =
              arith::RemSIOp::create(rewriter, loc, iMinusStart, thinningConst);
          auto modIsZero = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::eq, modThinning, c0);
          auto shouldStore =
              arith::AndIOp::create(rewriter, loc, geStartIdx, modIsZero);

          auto zeroCol = arith::ConstantOp::create(
              rewriter, loc, i64TensorType,
              DenseElementsAttr::get(i64TensorType,
                                     rewriter.getI64IntegerAttr(0)));
          auto updatedSamplesBuffer = impulse::DynamicUpdateSliceOp::create(
              rewriter, loc, samplesBufferType, samplesBufferLoop,
              q_constrained, ValueRange{storageIdx, zeroCol});
          auto selectedSamplesBuffer = impulse::SelectOp::create(
              rewriter, loc, samplesBufferType, shouldStore,
              updatedSamplesBuffer, samplesBufferLoop);

          auto oneCol = arith::ConstantOp::create(
              rewriter, loc, i64TensorType,
              DenseElementsAttr::get(i64TensorType,
                                     rewriter.getI64IntegerAttr(1)));
          auto accepted1x1 = impulse::ReshapeOp::create(
              rewriter, loc,
              RankedTensorType::get({1, 1}, rewriter.getI1Type()),
              sample.accepted);
          auto divergent1x1 = impulse::ReshapeOp::create(
              rewriter, loc,
              RankedTensorType::get({1, 1}, rewriter.getI1Type()),
              sample.divergent);
          auto updatedDiagnosticsBuffer = impulse::DynamicUpdateSliceOp::create(
              rewriter, loc, diagnosticsBufferType, diagnosticsBufferLoop,
              accepted1x1, ValueRange{storageIdx, zeroCol});
          updatedDiagnosticsBuffer = impulse::DynamicUpdateSliceOp::create(
              rewriter, loc, diagnosticsBufferType, updatedDiagnosticsBuffer,
              divergent1x1, ValueRange{storageIdx, oneCol});
          auto selectedDiagnosticsBuffer = impulse::SelectOp::create(
              rewriter, loc, diagnosticsBufferType, shouldStore,
              updatedDiagnosticsBuffer, diagnosticsBufferLoop);

          auto negU = arith::NegFOp::create(rewriter, loc, sample.U);
          auto logDensity1D = impulse::ReshapeOp::create(
              rewriter, loc, RankedTensorType::get({1}, elemType), negU);
          auto updatedLogDensityBuffer = impulse::DynamicUpdateSliceOp::create(
              rewriter, loc, logDensityBufferType, logDensityBufferLoop,
              logDensity1D, ValueRange{storageIdx});
          auto selectedLogDensityBuffer = impulse::SelectOp::create(
              rewriter, loc, logDensityBufferType, shouldStore,
              updatedLogDensityBuffer, logDensityBufferLoop);

          impulse::YieldOp::create(rewriter, loc,
                                   ValueRange{sample.q, sample.grad, sample.U,
                                              sample.rng, selectedSamplesBuffer,
                                              selectedDiagnosticsBuffer,
                                              selectedLogDensityBuffer});

          rewriter.setInsertionPointAfter(forLoopOp);
          finalQ = forLoopOp.getResult(0);
          finalGrad = forLoopOp.getResult(1);
          finalU = forLoopOp.getResult(2);
          finalRng = forLoopOp.getResult(3);
          finalSamplesBuffer = forLoopOp.getResult(4);
          finalDiagnosticsBuffer = forLoopOp.getResult(5);
          finalLogDensityBuffer = forLoopOp.getResult(6);
        }

        finalSamplesBuffer =
            conditionalDump(rewriter, loc, finalSamplesBuffer,
                            "MCMC: collected samples", debugDump);

        SmallVector<Value> replacements = {finalSamplesBuffer,
                                           finalDiagnosticsBuffer,
                                           finalLogDensityBuffer, finalRng};

        if (schema && !schema.empty()) {
          Value warmupOffsetIn = chainStateInSlot(state_slot::kWarmupOffset);
          if (!warmupOffsetIn)
            warmupOffsetIn = arith::ConstantOp::create(
                rewriter, loc, i64TensorType,
                DenseElementsAttr::get(i64TensorType,
                                       rewriter.getI64IntegerAttr(0)));
          Value warmupOffsetOut = arith::AddIOp::create(
              rewriter, loc, warmupOffsetIn,
              arith::ConstantOp::create(
                  rewriter, loc, i64TensorType,
                  DenseElementsAttr::get(
                      i64TensorType, rewriter.getI64IntegerAttr(numWarmup))));

          auto slotValue = [&](StringRef name) -> Value {
            if (name == state_slot::kPosition)
              return finalQ;
            if (name == state_slot::kGradient)
              return finalGrad;
            if (name == state_slot::kPotentialEnergy)
              return finalU;
            if (name == state_slot::kStepSize)
              return adaptedStepSize;
            if (name == state_slot::kInverseMassMatrix)
              return adaptedInvMass;
            if (name == state_slot::kWarmupOffset)
              return warmupOffsetOut;
            // Remaining slots are the contiguous 9-value adaptation block.
            unsigned base =
                *impulse::slotIndex(schema, state_slot::kDaLogStepSize);
            unsigned idx = *impulse::slotIndex(schema, name);
            return finalAdaptationState[idx - base];
          };

          // The verifier guarantees the declared schema has this layout.
          auto invMassTy = cast<RankedTensorType>(adaptedInvMass.getType());
          for (const impulse::StateSlot &slot : impulse::buildStateSchema(
                   exposeAdaptation, positionSize, elemType, invMassTy))
            replacements.push_back(slotValue(slot.name));
        }

        return replacements;
      };

      if (numChains <= 1) {
        rewriter.replaceOp(mcmcOp, runOneChain(rngInput, 0));
        return success();
      }

      // As NumPyro's MCMC.run: split the key per chain, or reuse saved keys.
      Value rngSlotIn = stateInSlot(mcmcOp, state_slot::kRng);
      bool multiResume = isResume && rngSlotIn;
      impulse::RandomSplitOp splitOp;
      if (!multiResume) {
        SmallVector<Type> chainKeyTypes(numChains, rngInput.getType());
        splitOp = impulse::RandomSplitOp::create(rewriter, loc, chainKeyTypes,
                                                 rngInput);
      }

      SmallVector<SmallVector<Value>> perChain;
      perChain.reserve(numChains);
      for (int64_t c = 0; c < numChains; ++c) {
        Value chainRng =
            multiResume ? sliceChainDim(rngSlotIn, c) : splitOp.getResult(c);
        perChain.push_back(runOneChain(chainRng, c));
      }

      auto idxConst = [&](int64_t v) -> Value {
        return arith::ConstantOp::create(
            rewriter, loc, i64TensorType,
            DenseElementsAttr::get(i64TensorType,
                                   rewriter.getI64IntegerAttr(v)));
      };
      Value zeroIdx = idxConst(0);

      // Needs an arith.constant-able element type; not for the ui64 rng.
      auto stackChainMajor = [&](unsigned resultIdx) -> Value {
        auto slotTy = cast<RankedTensorType>(perChain[0][resultIdx].getType());
        SmallVector<int64_t> cmShape;
        cmShape.push_back(numChains);
        cmShape.append(slotTy.getShape().begin(), slotTy.getShape().end());
        auto cmType = RankedTensorType::get(cmShape, slotTy.getElementType());
        SmallVector<int64_t> oneShape;
        oneShape.push_back(1);
        oneShape.append(slotTy.getShape().begin(), slotTy.getShape().end());
        auto oneType = RankedTensorType::get(oneShape, slotTy.getElementType());
        Value buf = arith::ConstantOp::create(rewriter, loc, cmType,
                                              rewriter.getZeroAttr(cmType));
        for (int64_t c = 0; c < numChains; ++c) {
          Value reshaped = impulse::ReshapeOp::create(rewriter, loc, oneType,
                                                      perChain[c][resultIdx]);
          SmallVector<Value> idxs;
          idxs.push_back(idxConst(c));
          for (int64_t d = 0; d < slotTy.getRank(); ++d)
            idxs.push_back(zeroIdx);
          buf = impulse::DynamicUpdateSliceOp::create(rewriter, loc, cmType,
                                                      buf, reshaped, idxs);
        }
        return buf;
      };

      // ui64 has no arith.constant; seed the buffer with chain 0's key.
      auto stackRngChainMajor = [&]() -> Value {
        // perChain layout: trace, diagnostics, log_densities, rng, state...
        auto rngTy = cast<RankedTensorType>(perChain[0][3].getType());
        SmallVector<int64_t> cmShape;
        cmShape.push_back(numChains);
        cmShape.append(rngTy.getShape().begin(), rngTy.getShape().end());
        auto cmType = RankedTensorType::get(cmShape, rngTy.getElementType());
        SmallVector<int64_t> oneShape;
        oneShape.push_back(1);
        oneShape.append(rngTy.getShape().begin(), rngTy.getShape().end());
        auto oneType = RankedTensorType::get(oneShape, rngTy.getElementType());
        Value buf = enzyme::BroadcastOp::create(
            rewriter, loc, cmType, perChain[0][3],
            rewriter.getDenseI64ArrayAttr({numChains}));
        for (int64_t c = 0; c < numChains; ++c) {
          Value reshaped = impulse::ReshapeOp::create(rewriter, loc, oneType,
                                                      perChain[c][3]);
          SmallVector<Value> idxs;
          idxs.push_back(idxConst(c));
          for (int64_t d = 0; d < rngTy.getRank(); ++d)
            idxs.push_back(zeroIdx);
          buf = impulse::DynamicUpdateSliceOp::create(rewriter, loc, cmType,
                                                      buf, reshaped, idxs);
        }
        return buf;
      };

      SmallVector<Value> stacked;
      stacked.push_back(stackChainMajor(0));
      stacked.push_back(stackChainMajor(1));
      stacked.push_back(stackChainMajor(2));
      // output_rng_state is the last chain's key; all keys go in `rng`.
      stacked.push_back(perChain[numChains - 1][3]);
      for (unsigned j = 4; j < perChain[0].size(); ++j)
        stacked.push_back(stackChainMajor(j));
      if (schema && !schema.empty())
        stacked.push_back(stackRngChainMajor());

      rewriter.replaceOp(mcmcOp, stacked);
      return success();
    }
  };

  // The VI learning rate is the `init_step_size` operand.
  template <typename OpT>
  struct LowerVIPattern : public mlir::OpRewritePattern<OpT> {
    static constexpr bool isRegionForm =
        std::is_same_v<OpT, impulse::InferRegionOp>;

    bool debugDump;

    LowerVIPattern(MLIRContext *context, bool debugDump,
                   PatternBenefit benefit = 1)
        : mlir::OpRewritePattern<OpT>(context, benefit), debugDump(debugDump) {}

    LogicalResult matchAndRewrite(OpT op,
                                  PatternRewriter &rewriter) const override {
      if (!op.getViConfig().has_value())
        return rewriter.notifyMatchFailure(op, "no vi_config; not a VI op");

      SymbolTableCollection symbolTable;
      bool hasLogpdfFn = static_cast<bool>(op.getLogpdfFnAttr());

      if (!op.getInitStepSize()) {
        op.emitError("Impulse VI: requires init_step_size (learning rate)");
        return failure();
      }
      Value stepSize = op.getInitStepSize();

      auto inputs = op.getInputs();
      if (inputs.empty()) {
        op.emitError("Impulse VI: requires at least rng_state input");
        return failure();
      }

      int64_t positionSize;
      SmallVector<Value> fnInputs;
      SmallVector<Type> fnResultTypes;
      Value originalTrace;
      ArrayAttr selection, allAddresses;
      SmallVector<SupportInfo> supports;
      FlatSymbolRefAttr logpdfFnAttr;
      RegionVILayout regionLayout;

      if constexpr (isRegionForm) {
        if (hasLogpdfFn) {
          op.emitError("Impulse VI: custom-logpdf mode is not supported in "
                       "region form");
          return failure();
        }
        if (op.getLogpdf().empty()) {
          op.emitError("Impulse VI: infer_region requires a constructed logpdf "
                       "region (run sicm first)");
          return failure();
        }
        fnInputs.assign(inputs.begin() + 1, inputs.end());
        originalTrace = op.getOriginalTrace();
        selection = op.getSelectionAttr();
        allAddresses = op.getAllAddressesAttr();
        if (!collectRegionVILayout(op, regionLayout)) {
          op.emitError("Impulse VI: failed to derive position layout from the "
                       "infer_region sampler");
          return failure();
        }
        supports = regionLayout.supports;
        positionSize = regionLayout.positionSize;
        if (positionSize <= 0)
          return failure();
      } else if (hasLogpdfFn) {
        logpdfFnAttr = op.getLogpdfFnAttr();
        fnInputs.assign(inputs.begin() + 1, inputs.end());
        auto initialPos = op.getInitPosition();
        positionSize =
            cast<RankedTensorType>(initialPos.getType()).getNumElements();
        selection = op.getSelectionAttr();
        allAddresses = op.getAllAddressesAttr();
      } else {
        fnInputs.assign(inputs.begin() + 1, inputs.end());
        originalTrace = op.getOriginalTrace();
        selection = op.getSelectionAttr();
        allAddresses = op.getAllAddressesAttr();

        auto fn = cast<FunctionOpInterface>(
            symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr()));
        positionSize =
            computePositionSizeForSelection(op, fn, selection, symbolTable);
        if (positionSize <= 0)
          return failure();

        supports = collectSupportInfoForSelection(
            op, fn, selection, allAddresses, symbolTable,
            /*positionDeclarationOrder=*/true);

        auto fnType = cast<FunctionType>(fn.getFunctionType());
        fnResultTypes.assign(fnType.getResults().begin(),
                             fnType.getResults().end());
      }

      auto viConfig = op.getViConfig().value();
      auto floatOr = [](FloatAttr a, double d) {
        return a ? a.getValueAsDouble() : d;
      };
      double initScaleVal = floatOr(viConfig.getInitScale(), 0.1);
      double adamB1Val = floatOr(viConfig.getAdamB1(), 0.9);
      double adamB2Val = floatOr(viConfig.getAdamB2(), 0.999);
      double adamEpsVal = floatOr(viConfig.getAdamEps(), 1.0e-8);
      double clipNormVal = floatOr(viConfig.getClipNorm(), 10.0);
      double initRadiusVal = floatOr(viConfig.getInitRadius(), 2.0);
      double momentumMassVal = floatOr(viConfig.getMomentumMass(), 0.9);
      double rmspropGammaVal = floatOr(viConfig.getRmspropGamma(), 0.9);
      double rmspropEpsVal = floatOr(viConfig.getRmspropEps(), 1.0e-8);

      VI::VIContext viCtx(
          hasLogpdfFn ? FlatSymbolRefAttr() : op.getFnAttr(), fnInputs,
          fnResultTypes, originalTrace, selection, allAddresses, stepSize,
          positionSize, op.getNumSamples(), supports,
          viConfig.getGuideKind().getValue(),
          viConfig.getOptimizerKind().getValue(), viConfig.getNumParticles(),
          initScaleVal, adamB1Val, adamB2Val, adamEpsVal);
      if (hasLogpdfFn)
        viCtx.logpdfFn = logpdfFnAttr;
      viCtx.autodiffAttrs = op.getAutodiffAttrsAttr();
      viCtx.debugDump = debugDump;
      viCtx.clipNorm = clipNormVal;
      viCtx.initRadius = initRadiusVal;
      viCtx.momentumMass = momentumMassVal;
      viCtx.rmspropGamma = rmspropGammaVal;
      viCtx.rmspropEps = rmspropEpsVal;
      viCtx.stableUpdate = viConfig.getStableUpdate();

      if constexpr (isRegionForm) {
        viCtx.logpdfRegion = &op.getLogpdf();
        viCtx.logpdfArgSlices = regionLayout.argSlices;
      }

      return VI::lowerVI(rewriter, op, viCtx, debugDump);
    }
  };

  // Which per-site yields (terminator operands 1..) depend on `seeds`.
  static SmallVector<bool> taintedSiteYields(Block &lpBlock, ValueRange seeds) {
    DenseSet<Value> tainted(seeds.begin(), seeds.end());
    auto isTainted = [&](Value v) { return tainted.contains(v); };
    for (Operation &o : lpBlock.without_terminator()) {
      bool uses = llvm::any_of(o.getOperands(), isTainted);
      if (!uses && o.getNumRegions() != 0)
        uses = o.walk([&](Operation *inner) {
                  return llvm::any_of(inner->getOperands(), isTainted)
                             ? WalkResult::interrupt()
                             : WalkResult::advance();
                }).wasInterrupted();
      if (uses)
        tainted.insert(o.result_begin(), o.result_end());
    }
    Operation *term = lpBlock.getTerminator();
    SmallVector<bool> dirty;
    for (Value v : llvm::drop_begin(term->getOperands()))
      dirty.push_back(isTainted(v));
    return dirty;
  }

  // Bit k: depends on selected site k. An `impulse.if` result depends on its
  // predicate only when its two yielded values differ.
  struct GibbsSiteDependence {
    unsigned numSites;
    DenseMap<Value, llvm::BitVector> sites;

    explicit GibbsSiteDependence(unsigned numSites) : numSites(numSites) {}

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
        llvm::BitVector pred = get(ifOp.getPredicate());
        Block &tb = ifOp.getTrueBranch().front();
        Block &fb = ifOp.getFalseBranch().front();
        run(tb);
        run(fb);
        Operation *ty = tb.getTerminator(), *fy = fb.getTerminator();
        for (auto [i, r] : llvm::enumerate(ifOp.getResults())) {
          Value a = ty->getOperand(i), b = fy->getOperand(i);
          llvm::BitVector s = get(a);
          s |= get(b);
          if (a != b)
            s |= pred;
          if (s.any())
            sites[r] = s;
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

  template <typename OpT>
  struct LowerMHInferPattern : public mlir::OpRewritePattern<OpT> {
    static constexpr bool isRegionForm =
        std::is_same_v<OpT, impulse::InferRegionOp>;
    using mlir::OpRewritePattern<OpT>::OpRewritePattern;

    static SmallVector<bool> dirtySiteYields(Block &lpBlock) {
      return taintedSiteYields(lpBlock, lpBlock.getArguments());
    }

    // With `delta`, weight is the dirty-term delta: XLA:CPU fuses the accept
    // decision into every write-back element, so it must stay a short chain.
    std::tuple<Value, Value, Value, Value>
    proposeRegion(OpBuilder &b, Location loc, OpT op, Value trace, Value rng,
                  ArrayAttr selection, ArrayAttr allAddresses, bool delta,
                  ColumnWindows &changedColumns,
                  const DenseMap<Operation *, Value> *drawOverride = nullptr,
                  Value *logRatioOut = nullptr) const {
      changedColumns.clear();
      Region &sampler = op.getSampler();
      Region &logpdf = op.getLogpdf();
      Block &sBlock = sampler.front();
      auto traceType = cast<RankedTensorType>(trace.getType());
      auto i64S = RankedTensorType::get({}, b.getI64Type());

      DenseMap<Attribute, impulse::SampleRegionOp> siteByAddress;
      for (Operation &o : sBlock)
        if (auto site = dyn_cast<impulse::SampleRegionOp>(&o))
          if (auto addr = site.getAddressAttr())
            siteByAddress[addr] = site;

      DenseMap<Attribute, int64_t> traceOffsetOf;
      int64_t offset = 0;
      for (Attribute addr : allAddresses) {
        traceOffsetOf[addr] = offset;
        impulse::SampleRegionOp site = siteByAddress.lookup(addr);
        for (unsigned i = 1, e = site.getNumResults(); i < e; ++i)
          offset += computeTensorElementCount(
              cast<RankedTensorType>(site.getResult(i).getType()));
      }
      const int64_t positionSize = offset;

      DenseSet<Attribute> selectedSet(selection.begin(), selection.end());

      IRMapping map;
      map.map(sBlock.getArgument(0), rng);
      for (unsigned i = 1, e = sBlock.getNumArguments(); i < e; ++i)
        map.map(sBlock.getArgument(i), op.getInputs()[i]);

      Value curRng = rng;
      Value newTrace = trace;
      DenseMap<Attribute, SmallVector<Value>> proposedSelected;
      // Program-order ordinal = per-site yield index = score-tail column.
      SmallVector<int64_t> selectedOrdinals;
      int64_t siteOrdinal = -1;

      for (Operation &o : sBlock.without_terminator()) {
        auto site = dyn_cast<impulse::SampleRegionOp>(&o);
        if (!site) {
          b.clone(o, map);
          continue;
        }
        ++siteOrdinal;
        Attribute addr = site.getAddressAttr();
        int64_t siteOff = traceOffsetOf.lookup(addr);
        if (selectedSet.contains(addr)) {
          selectedOrdinals.push_back(siteOrdinal);
          int64_t siteSize = 0;
          for (unsigned k = 1, e = site.getNumResults(); k < e; ++k)
            siteSize += computeTensorElementCount(
                cast<RankedTensorType>(site.getResult(k).getType()));
          changedColumns.push_back({siteOff, siteOff + siteSize});
          Block &sub = site.getSampler().front();
          IRMapping subMap;
          subMap.map(sub.getArgument(0), curRng);
          for (unsigned j = 1, e = site.getInputs().size(); j < e; ++j)
            subMap.map(sub.getArgument(j),
                       map.lookupOrDefault(site.getInputs()[j]));
          for (Operation &subOp : sub.without_terminator()) {
            if (drawOverride) {
              auto it = drawOverride->find(&subOp);
              if (it != drawOverride->end()) {
                subMap.map(subOp.getResult(0),
                           subMap.lookupOrDefault(subOp.getOperand(0)));
                subMap.map(subOp.getResult(1), it->second);
                continue;
              }
            }
            b.clone(subOp, subMap);
          }
          Operation *y = sub.getTerminator();
          curRng = subMap.lookupOrDefault(y->getOperand(0));
          map.map(site.getResult(0), curRng);
          SmallVector<Value> vals;
          int64_t wOff = siteOff;
          for (unsigned k = 1, e = site.getNumResults(); k < e; ++k) {
            Value v = subMap.lookupOrDefault(y->getOperand(k));
            map.map(site.getResult(k), v);
            vals.push_back(v);
            auto vt = cast<RankedTensorType>(v.getType());
            int64_t n = computeTensorElementCount(vt);
            auto flatType = RankedTensorType::get({1, n}, vt.getElementType());
            Value flat = impulse::ReshapeOp::create(b, loc, flatType, v);
            Value row0 = arith::ConstantOp::create(
                b, loc, i64S,
                DenseElementsAttr::get(i64S, b.getI64IntegerAttr(0)));
            Value col = arith::ConstantOp::create(
                b, loc, i64S,
                DenseElementsAttr::get(i64S, b.getI64IntegerAttr(wOff)));
            newTrace =
                impulse::DynamicUpdateSliceOp::create(
                    b, loc, traceType, newTrace, flat, ValueRange{row0, col})
                    .getResult();
            wOff += n;
          }
          proposedSelected[addr] = vals;
        } else {
          map.map(site.getResult(0), curRng);
          int64_t rOff = siteOff;
          for (unsigned k = 1, e = site.getNumResults(); k < e; ++k) {
            auto rt = cast<RankedTensorType>(site.getResult(k).getType());
            int64_t n = computeTensorElementCount(rt);
            auto sliceType = RankedTensorType::get({1, n}, rt.getElementType());
            Value slice = impulse::SliceOp::create(
                b, loc, sliceType, trace, b.getDenseI64ArrayAttr({0, rOff}),
                b.getDenseI64ArrayAttr({1, rOff + n}),
                b.getDenseI64ArrayAttr({1, 1}));
            Value val = impulse::ReshapeOp::create(b, loc, rt, slice);
            map.map(site.getResult(k), val);
            rOff += n;
          }
        }
      }

      Block &lpBlock = logpdf.front();
      IRMapping lpMap;
      unsigned argIdx = 0;
      for (Attribute addr : selection)
        for (Value v : proposedSelected[addr])
          lpMap.map(lpBlock.getArgument(argIdx++), v);
      for (Operation &lo : lpBlock.without_terminator())
        b.clone(lo, lpMap);
      Operation *lpTerm = lpBlock.getTerminator();
      Value newWeight = lpMap.lookupOrDefault(lpTerm->getOperand(0));
      SmallVector<bool> dirty =
          delta ? dirtySiteYields(lpBlock)
                : SmallVector<bool>(lpTerm->getNumOperands() - 1, true);

      // A trace wider than positionSize carries a per-site score tail.
      if (traceType.getShape()[1] > positionSize) {
        Value row0 = arith::ConstantOp::create(
            b, loc, i64S, DenseElementsAttr::get(i64S, b.getI64IntegerAttr(0)));
        auto s11 = RankedTensorType::get({1, 1}, traceType.getElementType());
        for (unsigned j = 1, e = lpTerm->getNumOperands(); j < e; ++j) {
          if (!dirty[j - 1])
            continue;
          Value score = lpMap.lookupOrDefault(lpTerm->getOperand(j));
          Value flat = impulse::ReshapeOp::create(b, loc, s11, score);
          Value col = arith::ConstantOp::create(
              b, loc, i64S,
              DenseElementsAttr::get(
                  i64S, b.getI64IntegerAttr(positionSize + (int64_t)j - 1)));
          newTrace =
              impulse::DynamicUpdateSliceOp::create(b, loc, traceType, newTrace,
                                                    flat, ValueRange{row0, col})
                  .getResult();
        }
      }

      IRMapping oldLpMap;
      if (traceType.getShape()[1] == positionSize) {
        unsigned oldArgIdx = 0;
        for (Attribute addr : selection) {
          int64_t rOff = traceOffsetOf.lookup(addr);
          for (Value v : proposedSelected[addr]) {
            auto vt = cast<RankedTensorType>(v.getType());
            int64_t n = computeTensorElementCount(vt);
            Value slice = impulse::SliceOp::create(
                b, loc, RankedTensorType::get({1, n}, vt.getElementType()),
                trace, b.getDenseI64ArrayAttr({0, rOff}),
                b.getDenseI64ArrayAttr({1, rOff + n}),
                b.getDenseI64ArrayAttr({1, 1}));
            oldLpMap.map(lpBlock.getArgument(oldArgIdx++),
                         impulse::ReshapeOp::create(b, loc, vt, slice));
            rOff += n;
          }
        }
        for (Operation &lo : lpBlock.without_terminator())
          b.clone(lo, oldLpMap);
      }
      Value corr;
      for (int64_t j : selectedOrdinals) {
        Value newScore = lpMap.lookupOrDefault(lpTerm->getOperand(1 + j));
        Value oldScore =
            traceType.getShape()[1] == positionSize
                ? oldLpMap.lookupOrDefault(lpTerm->getOperand(1 + j))
                : scalarAt(b, loc, trace, {0, positionSize + j});
        Value d = arith::SubFOp::create(b, loc, newScore, oldScore);
        corr = corr ? arith::AddFOp::create(b, loc, corr, d).getResult() : d;
      }

      if (delta) {
        // Strided vector families reduce in one op; this reorders the sum.
        auto familyElem =
            [&](Value v) -> std::optional<std::pair<Value, int64_t>> {
          auto r = v.getDefiningOp<impulse::ReshapeOp>();
          auto sl = r ? r->getOperand(0).getDefiningOp<impulse::SliceOp>()
                      : v.getDefiningOp<impulse::SliceOp>();
          if (!sl)
            return std::nullopt;
          Value src = sl->getOperand(0);
          auto st = cast<RankedTensorType>(src.getType());
          if (st.getRank() != 1 || sl.getStrides()[0] != 1 ||
              sl.getLimitIndices()[0] != sl.getStartIndices()[0] + 1)
            return std::nullopt;
          return std::make_pair(src, sl.getStartIndices()[0]);
        };
        SmallVector<int64_t> dirtyIdx;
        for (auto [j, isDirty] : llvm::enumerate(dirty))
          if (isDirty)
            dirtyIdx.push_back(j);
        llvm::SmallDenseSet<int64_t> inFamily;
        SmallVector<std::tuple<Value, int64_t, int64_t, int64_t, int64_t>>
            families; // (V, k0, K, j0, stride)
        for (size_t a = 0; a < dirtyIdx.size();) {
          auto fa = familyElem(
              lpMap.lookupOrDefault(lpTerm->getOperand(1 + dirtyIdx[a])));
          if (!fa) {
            ++a;
            continue;
          }
          size_t e = a + 1;
          int64_t stride = 0;
          while (e < dirtyIdx.size()) {
            auto fe = familyElem(
                lpMap.lookupOrDefault(lpTerm->getOperand(1 + dirtyIdx[e])));
            int64_t s = dirtyIdx[e] - dirtyIdx[e - 1];
            if (!fe || fe->first != fa->first ||
                fe->second != fa->second + (int64_t)(e - a) ||
                (stride && s != stride))
              break;
            stride = s;
            ++e;
          }
          if (e - a >= 4) {
            families.push_back(
                {fa->first, fa->second, (int64_t)(e - a), dirtyIdx[a], stride});
            for (size_t i = a; i < e; ++i)
              inFamily.insert(dirtyIdx[i]);
          }
          a = e;
        }
        // log alpha is the other dirty terms, so XLA does not recompute draws.
        bool splitRatio = logRatioOut && selectedOrdinals.size() >= 2 &&
                          traceType.getShape()[1] > positionSize &&
                          llvm::none_of(selectedOrdinals, [&](int64_t j) {
                            return inFamily.contains(j);
                          });
        llvm::SmallDenseSet<int64_t> own(selectedOrdinals.begin(),
                                         selectedOrdinals.end());
        Value ratio;
        Value sum;
        for (int64_t j : dirtyIdx) {
          if (inFamily.contains(j))
            continue;
          Value newScore = lpMap.lookupOrDefault(lpTerm->getOperand(1 + j));
          Value oldScore =
              scalarAt(b, loc, trace, {0, positionSize + (int64_t)j});
          Value d = arith::SubFOp::create(b, loc, newScore, oldScore);
          sum = sum ? arith::AddFOp::create(b, loc, sum, d).getResult() : d;
          if (splitRatio && !own.contains(j))
            ratio =
                ratio ? arith::AddFOp::create(b, loc, ratio, d).getResult() : d;
        }
        for (auto [V, k0, K, j0, stride] : families) {
          auto vt = cast<RankedTensorType>(V.getType());
          auto kt = RankedTensorType::get({K}, vt.getElementType());
          Value newVec = V;
          if (k0 != 0 || vt.getDimSize(0) != K)
            newVec = impulse::SliceOp::create(
                b, loc, kt, V, b.getDenseI64ArrayAttr({k0}),
                b.getDenseI64ArrayAttr({k0 + K}), b.getDenseI64ArrayAttr({1}));
          int64_t c0 = positionSize + j0;
          Value old = impulse::SliceOp::create(
              b, loc, RankedTensorType::get({1, K}, vt.getElementType()), trace,
              b.getDenseI64ArrayAttr({0, c0}),
              b.getDenseI64ArrayAttr({1, c0 + stride * (K - 1) + 1}),
              b.getDenseI64ArrayAttr({1, stride}));
          old = impulse::ReshapeOp::create(b, loc, kt, old);
          Value d = arith::SubFOp::create(b, loc, newVec, old);
          Value ones = arith::ConstantOp::create(
              b, loc, kt, DenseElementsAttr::get(kt, 1.0));
          Value fs = impulse::DotOp::create(
              b, loc, RankedTensorType::get({}, vt.getElementType()), d, ones,
              b.getDenseI64ArrayAttr({}), b.getDenseI64ArrayAttr({}),
              b.getDenseI64ArrayAttr({0}), b.getDenseI64ArrayAttr({0}));
          sum = sum ? arith::AddFOp::create(b, loc, sum, fs).getResult() : fs;
          // No family member is a selected site's own term (checked above).
          if (splitRatio)
            ratio = ratio ? arith::AddFOp::create(b, loc, ratio, fs).getResult()
                          : fs;
        }
        if (splitRatio) {
          if (!ratio) {
            auto scalarType = RankedTensorType::get({}, b.getF64Type());
            ratio = arith::ConstantOp::create(
                b, loc, scalarType, DenseElementsAttr::get(scalarType, 0.0));
          }
          *logRatioOut = ratio;
        }
        if (!sum) {
          auto scalarType = RankedTensorType::get({}, b.getF64Type());
          sum = arith::ConstantOp::create(
              b, loc, scalarType, DenseElementsAttr::get(scalarType, 0.0));
        }
        newWeight = sum;
      }

      // Only dirty scores: XLA:CPU copies the accept chain into every element.
      if (delta) {
        for (auto [j, isDirty] : llvm::enumerate(dirty))
          if (isDirty)
            changedColumns.push_back(
                {positionSize + (int64_t)j, positionSize + (int64_t)j + 1});
      } else if (traceType.getShape()[1] > positionSize) {
        changedColumns.push_back({positionSize, traceType.getShape()[1]});
      }
      changedColumns = mergeColumnWindows(changedColumns);

      return {newTrace, newWeight, corr, curRng};
    }

    // Weight is the dirty-term delta for a score-tail trace, else the joint.
    std::tuple<Value, Value, Value, Value>
    proposeFlat(OpBuilder &b, Location loc, OpT op, FunctionOpInterface fn,
                ArrayRef<Type> fnResultTypes, Value trace, Value rng,
                ArrayAttr selection, ArrayAttr allAddresses) const {
      auto weightType = RankedTensorType::get({2}, b.getF64Type());
      SmallVector<Type> regenResultTypes{trace.getType(), weightType};
      for (Type t : fnResultTypes)
        regenResultTypes.push_back(t);
      SmallVector<Value> inputs(op.getInputs().begin(), op.getInputs().end());
      inputs[0] = rng;
      auto regen = impulse::RegenerateOp::create(
          b, loc, regenResultTypes, op.getFnAttr(), inputs, trace,
          op.getConstraint(), /*selection=*/allAddresses,
          /*regenerate_addresses=*/selection, op.getConstrainedAddressesAttr(),
          b.getStringAttr(""));
      regen->setAttr(kMHHastingsAttr, b.getUnitAttr());
      auto [weight, corr] = unpackHastingsWeight(b, loc, regen.getWeight());
      return {regen.getNewTrace(), weight, corr, regen.getOutputs()[0]};
    }

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
        if (op.getSampler().empty() || op.getLogpdf().empty()) {
          op.emitError("Impulse MH: infer_region requires sampler and "
                       "constructed logpdf regions (run sicm first)");
          return failure();
        }
        unsigned numSites = 0;
        for (Operation &o : op.getSampler().front())
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
      auto weightType = cast<RankedTensorType>(initWeight.getType());
      Type rngType = initRng.getType();
      auto i1Scalar = RankedTensorType::get({}, rewriter.getI1Type());

      ArrayAttr selection = op.getSelectionAttr();
      ArrayAttr allAddresses = op.getAllAddressesAttr();

      SymbolTableCollection symbolTable;
      FunctionOpInterface fn;
      SmallVector<Type> fnResultTypes;
      if constexpr (!isRegionForm) {
        fn = dyn_cast_or_null<FunctionOpInterface>(
            symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr()));
        if (!fn || fn.getFunctionBody().empty()) {
          op.emitError("Impulse MH: `fn` must reference a non-empty funcOp");
          return failure();
        }
        for (Type t : fn.getResultTypes())
          fnResultTypes.push_back(t);
      }

      // A trace wider than the positions carries a score tail (delta weight).
      int64_t fullPositionSize = 0;
      if constexpr (isRegionForm) {
        for (Operation &o : op.getSampler().front())
          if (auto site = dyn_cast<impulse::SampleRegionOp>(&o))
            for (unsigned k = 1, e = site.getNumResults(); k < e; ++k)
              fullPositionSize += computeTensorElementCount(
                  cast<RankedTensorType>(site.getResult(k).getType()));
      } else {
        fullPositionSize =
            computePositionSizeForSelection(op, fn, allAddresses, symbolTable);
      }
      bool deltaWeight = traceType.getShape()[1] > fullPositionSize;

      // The region form fills this in during the proposal.
      ColumnWindows changedColumns;
      if constexpr (!isRegionForm)
        changedColumns = mhChangedColumnsFlat(
            op, fn, allAddresses, selection, op.getConstrainedAddressesAttr(),
            static_cast<bool>(op.getConstraint()), traceType.getShape()[1],
            symbolTable);

      auto oneStep = [&](OpBuilder &b, Value trace, Value weight,
                         Value rng) -> std::array<Value, 4> {
        Value newTrace, newWeight, corr, stepRng, logRatio;
        if constexpr (isRegionForm)
          std::tie(newTrace, newWeight, corr, stepRng) =
              proposeRegion(b, loc, op, trace, rng, selection, allAddresses,
                            deltaWeight, changedColumns, nullptr, &logRatio);
        else
          std::tie(newTrace, newWeight, corr, stepRng) =
              proposeFlat(b, loc, op, fn, fnResultTypes, trace, rng, selection,
                          allAddresses);

        Value jointRatio =
            deltaWeight
                ? newWeight
                : arith::SubFOp::create(b, loc, newWeight, weight).getResult();
        Value logAlpha =
            logRatio
                ? logRatio
                : arith::SubFOp::create(b, loc, jointRatio, corr).getResult();
        Value acceptWeight =
            deltaWeight
                ? arith::AddFOp::create(b, loc, weight, newWeight).getResult()
                : newWeight;
        Value zero = arith::ConstantOp::create(
            b, loc, weightType, DenseElementsAttr::get(weightType, 0.0));
        Value one = arith::ConstantOp::create(
            b, loc, weightType, DenseElementsAttr::get(weightType, 1.0));
        auto rnd = impulse::RandomOp::create(
            b, loc, TypeRange{rngType, weightType}, stepRng, zero, one,
            impulse::RngDistributionAttr::get(
                b.getContext(), impulse::RngDistribution::UNIFORM));
        Value logRand = math::LogOp::create(b, loc, rnd.getResult());
        Value finalRng = rnd.getOutputRngState();
        Value accepted = arith::CmpFOp::create(
            b, loc, arith::CmpFPredicate::OLT, logRand, logAlpha);
        Value selTrace = writeBackMHWindows(b, loc, trace, newTrace, accepted,
                                            changedColumns);
        Value selWeight =
            arith::SelectOp::create(b, loc, accepted, acceptWeight, weight);
        return {selTrace, selWeight, finalRng, accepted};
      };

      int64_t numSamples = op.getNumSamples();
      Value outTrace, outWeight, outRng, outAccepted;

      if (numSamples <= 1) {
        auto r = oneStep(rewriter, initTrace, initWeight, initRng);
        outTrace = r[0];
        outWeight = r[1];
        outRng = r[2];
        outAccepted = r[3];
      } else {
        auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
        auto iConst = [&](int64_t v) {
          return arith::ConstantOp::create(
              rewriter, loc, i64S,
              DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(v)));
        };
        Value c0 = iConst(0), cN = iConst(numSamples), c1 = iConst(1);
        Value falseC = arith::ConstantOp::create(
            rewriter, loc, i1Scalar,
            DenseElementsAttr::get(i1Scalar, rewriter.getBoolAttr(false)));
        SmallVector<Type> loopTypes{traceType, weightType, rngType, i1Scalar};
        SmallVector<Value> initArgs{initTrace, initWeight, initRng, falseC};
        auto forOp = impulse::ForOp::create(rewriter, loc, loopTypes, c0, cN,
                                            c1, initArgs);
        Block *body = rewriter.createBlock(&forOp.getRegion());
        body->addArgument(i64S, loc);
        for (Type t : loopTypes)
          body->addArgument(t, loc);
        rewriter.setInsertionPointToStart(body);
        auto r = oneStep(rewriter, body->getArgument(1), body->getArgument(2),
                         body->getArgument(3));
        impulse::YieldOp::create(rewriter, loc,
                                 ValueRange{r[0], r[1], r[2], r[3]});
        rewriter.setInsertionPointAfter(forOp);
        outTrace = forOp.getResult(0);
        outWeight = forOp.getResult(1);
        outRng = forOp.getResult(2);
        outAccepted = forOp.getResult(3);
      }

      // MH reports no per-sample log density; `log_densities` is zero.
      Value outLogDensities = arith::ConstantOp::create(
          rewriter, loc, op.getLogDensities().getType(),
          DenseElementsAttr::get(
              cast<ShapedType>(op.getLogDensities().getType()),
              rewriter.getFloatAttr(
                  cast<ShapedType>(op.getLogDensities().getType())
                      .getElementType(),
                  0.0)));
      rewriter.replaceOp(
          op, {outTrace, outAccepted, outLogDensities, outRng, outWeight});
      return success();
    }
  };

  // Batches a run of single-site MH moves. Sound only when each per-site yield
  // depends on at most one run site, no site's parameters depend on another,
  // written columns are disjoint and all proposals are one matching draw.
  struct MHRunBatcher {
    using MHRegion = LowerMHInferPattern<impulse::InferRegionOp>;

    struct Move {
      // Op handles are mutable: their accessors are non-const.
      mutable impulse::InferRegionOp op;
      Attribute address;
      mutable impulse::SampleRegionOp site;
      mutable impulse::RandomOp draw;
      ColumnWindows windows;    // value columns + dirty score columns
      int64_t positionSize = 0; // score column of yield j: positionSize + j
      int64_t ordinal = -1;     // the site's yield index (program order)
    };

    static std::optional<Move> analyzeMove(impulse::InferRegionOp op) {
      if (!op.getMhConfig().has_value() || op.getSampler().empty() ||
          op.getLogpdf().empty() || op.getNumSamples() > 1 ||
          op.getStateIn().size() != 1 || op.getInputs().empty() ||
          op.getConstraint() || !op.getOriginalTrace())
        return std::nullopt;
      ArrayAttr selection = op.getSelectionAttr();
      ArrayAttr allAddresses = op.getAllAddressesAttr();
      if (!selection || selection.size() != 1 || !allAddresses)
        return std::nullopt;
      Block &sBlock = op.getSampler().front();
      Block &lpBlock = op.getLogpdf().front();
      DenseMap<Attribute, impulse::SampleRegionOp> siteByAddress;
      SmallVector<impulse::SampleRegionOp> sites;
      for (Operation &o : sBlock)
        if (auto site = dyn_cast<impulse::SampleRegionOp>(&o)) {
          sites.push_back(site);
          if (auto addr = site.getAddressAttr())
            siteByAddress[addr] = site;
        }
      Operation *lpTerm = lpBlock.getTerminator();
      if (lpTerm->getNumOperands() != 1 + sites.size() ||
          lpBlock.getNumArguments() != 1)
        return std::nullopt;
      Attribute address = selection[0];
      impulse::SampleRegionOp site = siteByAddress.lookup(address);
      if (!site || site.getNumResults() != 2)
        return std::nullopt;
      DenseMap<Attribute, int64_t> traceOffsetOf;
      int64_t positionSize = 0;
      for (Attribute addr : allAddresses) {
        impulse::SampleRegionOp s = siteByAddress.lookup(addr);
        if (!s)
          return std::nullopt;
        traceOffsetOf[addr] = positionSize;
        for (unsigned i = 1, e = s.getNumResults(); i < e; ++i)
          positionSize += computeTensorElementCount(
              cast<RankedTensorType>(s.getResult(i).getType()));
      }
      auto traceType = cast<RankedTensorType>(op.getOriginalTrace().getType());
      if (traceType.getRank() != 2 || traceType.getShape()[0] != 1 ||
          traceType.getShape()[1] <= positionSize)
        return std::nullopt;
      Block &sub = site.getSampler().front();
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
      auto dist = draw.getRngDistribution();
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
      Move m;
      m.op = op;
      m.address = address;
      m.site = site;
      m.draw = draw;
      m.positionSize = positionSize;
      m.ordinal = llvm::find(sites, site) - sites.begin();
      int64_t off = traceOffsetOf.lookup(address);
      int64_t size = computeTensorElementCount(
          cast<RankedTensorType>(site.getResult(1).getType()));
      m.windows.push_back({off, off + size});
      SmallVector<bool> dirty =
          taintedSiteYields(lpBlock, lpBlock.getArguments());
      for (auto [j, d] : llvm::enumerate(dirty))
        if (d)
          m.windows.push_back(
              {positionSize + (int64_t)j, positionSize + (int64_t)j + 1});
      m.windows = mergeColumnWindows(m.windows);
      return m;
    }

    static bool chained(const Move &prev, const Move &next) {
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
      return p.getOutputRngState().hasOneUse() &&
             p.getStateOut()[0].hasOneUse();
    }

    static bool sameValueOrConstant(Value a, Value b) {
      if (a == b)
        return true;
      Attribute ca, cb;
      return matchPattern(a, m_Constant(&ca)) &&
             matchPattern(b, m_Constant(&cb)) && ca == cb &&
             a.getType() == b.getType();
    }

    // Intermediate traces are those of every move but the last.
    struct Between {
      bool ok = true;
      DenseMap<Value, unsigned> intermediate; // trace of move i -> i
      SmallVector<Operation *> derived;       // program order
      DenseSet<Operation *> derivedSet;
      DenseMap<Value, llvm::BitVector> outer; // run-site dependence
      // Used after the run; recomputed on the final trace when finalSafe.
      SetVector<Value> escaping;
    };

    static Between analyzeBetween(ArrayRef<Move> ms) {
      Between bt;
      unsigned K = ms.size();
      Block *block = ms.front().op->getBlock();
      Operation *first = ms.front().op, *last = ms.back().op;
      DenseSet<Operation *> moveOps;
      for (const Move &m : ms)
        moveOps.insert(m.op);
      for (unsigned i = 0; i + 1 < K; ++i)
        bt.intermediate[ms[i].op.getTrace()] = i;
      auto writtenBy = [&](int64_t c, unsigned from, unsigned to) {
        llvm::BitVector s(K);
        for (unsigned i = from; i < to; ++i)
          for (auto [ws, we] : ms[i].windows)
            if (c >= ws && c < we)
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
        llvm::BitVector s(K);
        auto addValue = [&](Value v) {
          if (bt.intermediate.count(v)) {
            fromIntermediate = true;
            safe = false;
            s.set();
          } else if (Operation *d = v.getDefiningOp();
                     d && bt.derivedSet.contains(d)) {
            fromIntermediate = true;
            safe &= finalSafe.lookup(d);
            s |= bt.outer.lookup(v);
          }
        };
        auto slice = dyn_cast<impulse::SliceOp>(o);
        if (slice && bt.intermediate.count(slice.getOperand())) {
          ArrayRef<int64_t> st = slice.getStartIndices(),
                            li = slice.getLimitIndices(),
                            sd = slice.getStrides();
          if (st.size() != 2) {
            bt.ok = false;
            return bt;
          }
          unsigned i = bt.intermediate.lookup(slice.getOperand());
          fromIntermediate = true;
          for (int64_t c = st[1]; c < li[1]; c += sd[1]) {
            s |= writtenBy(c, 0, K);
            safe &= writtenBy(c, i + 1, K).none();
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
        if (!isMemoryEffectFree(o)) {
          bt.ok = false;
          return bt;
        }
        bt.derived.push_back(o);
        bt.derivedSet.insert(o);
        finalSafe[o] = safe;
        for (Value r : o->getResults())
          bt.outer[r] = s;
      }
      for (unsigned i = 0; i + 1 < K; ++i)
        for (OpOperand &u : ms[i].op.getTrace().getUses()) {
          if (u.getOwner() == ms[i + 1].op.getOperation() &&
              u.getOperandNumber() ==
                  ms[i + 1].op.getOriginalTraceMutable()[0].getOperandNumber())
            continue;
          Operation *anc = block->findAncestorOpInBlock(*u.getOwner());
          if (!anc || !bt.derivedSet.contains(anc)) {
            bt.ok = false;
            return bt;
          }
        }
      for (Operation *o : bt.derived)
        for (Value r : o->getResults())
          for (OpOperand &u : r.getUses()) {
            Operation *anc = block->findAncestorOpInBlock(*u.getOwner());
            if (!anc || (moveOps.contains(anc) && anc == u.getOwner())) {
              bt.ok = false; // an operand of a move, not a region capture
              return bt;
            }
            if (last->isBeforeInBlock(anc)) {
              if (!finalSafe.lookup(o)) {
                bt.ok = false;
                return bt;
              }
              bt.escaping.insert(r);
            }
          }
      return bt;
    }

    static bool batchable(ArrayRef<Move> ms) {
      unsigned K = ms.size();
      Block *block = ms.front().op->getBlock();
      Operation *last = ms.back().op;
      impulse::RandomOp d0 = ms.front().draw;
      for (const Move &m : ms) {
        if (m.draw.getRngDistribution() != d0.getRngDistribution() ||
            m.draw.getResult().getType() != d0.getResult().getType() ||
            !sameValueOrConstant(m.draw.getA(), d0.getA()) ||
            !sameValueOrConstant(m.draw.getB(), d0.getB()))
          return false;
      }
      ColumnWindows all;
      for (const Move &m : ms)
        for (auto w : m.windows)
          all.push_back(w);
      llvm::sort(all);
      for (unsigned i = 1; i < all.size(); ++i)
        if (all[i].first < all[i - 1].second)
          return false;
      for (const Move &m : ms)
        for (Value v : {m.op.getDiagnostics(), m.op.getLogDensities()})
          for (OpOperand &u : v.getUses()) {
            Operation *anc = block->findAncestorOpInBlock(*u.getOwner());
            if (!anc || !last->isBeforeInBlock(anc))
              return false;
          }
      Between bt = analyzeBetween(ms);
      if (!bt.ok)
        return false;
      DenseMap<Value, unsigned> &intermediate = bt.intermediate;
      DenseMap<Value, llvm::BitVector> &outer = bt.outer;
      for (unsigned k = 0; k < K; ++k) {
        impulse::InferRegionOp op = ms[k].op;
        auto seedCaptured = [&](GibbsSiteDependence &dep, Region &r) {
          SetVector<Value> captured;
          getUsedValuesDefinedAbove(r, captured);
          for (Value v : captured) {
            if (intermediate.count(v))
              return false;
            auto it = outer.find(v);
            if (it != outer.end())
              dep.seed(v, it->second);
          }
          return true;
        };
        GibbsSiteDependence lpDep(K);
        if (!seedCaptured(lpDep, op.getLogpdf()))
          return false;
        Block &lpBlock = op.getLogpdf().front();
        lpDep.seed(lpBlock.getArgument(0), k);
        lpDep.run(lpBlock);
        for (Value y : llvm::drop_begin(lpBlock.getTerminator()->getOperands()))
          if (lpDep.get(y).count() > 1)
            return false;
        GibbsSiteDependence sDep(K);
        if (!seedCaptured(sDep, op.getSampler()))
          return false;
        Block &sBlock = op.getSampler().front();
        for (Operation &o : sBlock)
          if (auto s = dyn_cast<impulse::SampleRegionOp>(&o))
            for (unsigned j = 0; j < K; ++j)
              if (s.getAddressAttr() == ms[j].address)
                for (Value r : llvm::drop_begin(s.getResults()))
                  sDep.seed(r, j);
        sDep.run(sBlock);
        for (Value param : llvm::drop_begin(ms[k].site->getOperands())) {
          llvm::BitVector p = sDep.get(param);
          p.reset(k);
          if (p.any())
            return false;
        }
      }
      return true;
    }

    static void emit(ArrayRef<Move> ms, MLIRContext *ctx) {
      unsigned K = ms.size();
      impulse::InferRegionOp first = ms.front().op, last = ms.back().op;
      Location loc = last.getLoc();
      Value trace0 = first.getOriginalTrace();
      Value weight0 = first.getStateIn()[0];
      Value rng0 = first.getInputs()[0];
      auto traceType = cast<RankedTensorType>(trace0.getType());
      auto weightType = cast<RankedTensorType>(weight0.getType());
      Type rngType = rng0.getType();
      Between bt = analyzeBetween(ms);
      DenseMap<Operation *, SmallVector<unsigned>> readsIntermediate;
      for (Operation *o : bt.derived)
        for (OpOperand &u : o->getOpOperands())
          if (bt.intermediate.count(u.get()))
            readsIntermediate[o].push_back(u.getOperandNumber());
      for (unsigned i = 0; i + 1 < K; ++i)
        ms[i].op.getTrace().replaceUsesWithIf(trace0, [&](OpOperand &u) {
          return !(
              u.getOwner() == ms[i + 1].op.getOperation() &&
              u.getOperandNumber() ==
                  ms[i + 1].op.getOriginalTraceMutable()[0].getOperandNumber());
        });

      OpBuilder b(last);
      auto i64S = RankedTensorType::get({}, b.getI64Type());
      auto idx = [&](int64_t v) -> Value {
        return arith::ConstantOp::create(
            b, loc, i64S, DenseElementsAttr::get(i64S, b.getI64IntegerAttr(v)));
      };
      auto element = [&](Value vec, int64_t k, RankedTensorType elemType) {
        auto vt = cast<RankedTensorType>(vec.getType());
        SmallVector<int64_t> start(vt.getRank(), 0), limit(vt.getShape()),
            strides(vt.getRank(), 1);
        start[0] = k;
        limit[0] = k + 1;
        SmallVector<int64_t> shape(vt.getShape());
        shape[0] = 1;
        Value sl = impulse::SliceOp::create(
            b, loc, RankedTensorType::get(shape, vt.getElementType()), vec,
            b.getDenseI64ArrayAttr(start), b.getDenseI64ArrayAttr(limit),
            b.getDenseI64ArrayAttr(strides));
        return impulse::ReshapeOp::create(b, loc, elemType, sl).getResult();
      };

      impulse::RandomOp d0 = ms.front().draw;
      auto drawType = cast<RankedTensorType>(d0.getResult().getType());
      SmallVector<int64_t> batchedShape{(int64_t)K};
      batchedShape.append(drawType.getShape().begin(),
                          drawType.getShape().end());
      auto proposals = impulse::RandomOp::create(
          b, loc,
          TypeRange{rngType, RankedTensorType::get(batchedShape,
                                                   drawType.getElementType())},
          rng0, d0.getA(), d0.getB(), d0.getRngDistributionAttr());
      auto vecType =
          RankedTensorType::get({(int64_t)K}, weightType.getElementType());
      Value zero = arith::ConstantOp::create(
          b, loc, weightType, DenseElementsAttr::get(weightType, 0.0));
      Value one = arith::ConstantOp::create(
          b, loc, weightType, DenseElementsAttr::get(weightType, 1.0));
      auto accepts = impulse::RandomOp::create(
          b, loc, TypeRange{rngType, vecType}, proposals.getOutputRngState(),
          zero, one,
          impulse::RngDistributionAttr::get(ctx,
                                            impulse::RngDistribution::UNIFORM));
      Value logRand = math::LogOp::create(b, loc, accepts.getResult());

      MHRegion mh(ctx);
      SmallVector<Value> newTraces, deltaCells, corrCells;
      SmallVector<ColumnWindows> changed(K);
      auto elemType = traceType.getElementType();
      auto oneType = RankedTensorType::get({1}, elemType);
      auto rowType = RankedTensorType::get({1, (int64_t)K}, elemType);
      auto colType = RankedTensorType::get({1, 1}, elemType);
      auto assembleVec = [&](ArrayRef<Value> cells) -> Value {
        Value v = arith::ConstantOp::create(
            b, loc, vecType, DenseElementsAttr::get(vecType, 0.0));
        for (auto [k, c] : llvm::enumerate(cells)) {
          Value cellVec = impulse::ReshapeOp::create(b, loc, oneType, c);
          v = impulse::DynamicUpdateSliceOp::create(b, loc, vecType, v, cellVec,
                                                    ValueRange{idx(k)});
        }
        return v;
      };
      auto unitSlice = [&](Value t, int64_t c) -> Value {
        return impulse::SliceOp::create(
            b, loc, colType, t, b.getDenseI64ArrayAttr({0, c}),
            b.getDenseI64ArrayAttr({1, c + 1}), b.getDenseI64ArrayAttr({1, 1}));
      };
      for (unsigned k = 0; k < K; ++k) {
        DenseMap<Operation *, Value> override;
        override[ms[k].draw] = element(proposals.getResult(), k, drawType);
        auto [newTrace, delta, corr, stepRng] = mh.proposeRegion(
            b, loc, ms[k].op, trace0, rng0, ms[k].op.getSelectionAttr(),
            ms[k].op.getAllAddressesAttr(), /*delta=*/true, changed[k],
            &override);
        newTraces.push_back(newTrace);
        deltaCells.push_back(delta);
        corrCells.push_back(corr);
      }

      // Slotted: all moves write the same column kinds, own score at one slot.
      SmallVector<SmallVector<int64_t>> cols(K);
      for (unsigned k = 0; k < K; ++k)
        for (auto [ws, we] : changed[k])
          for (int64_t c = ws; c < we; ++c)
            cols[k].push_back(c);
      size_t S = cols.front().size();
      int64_t ownSlot = -1;
      bool slotted = true;
      for (unsigned k = 0; k < K && slotted; ++k) {
        slotted = cols[k].size() == S;
        for (size_t s = 0; s < S && slotted; ++s)
          slotted = (cols[k][s] >= ms[k].positionSize) ==
                    (cols[0][s] >= ms[0].positionSize);
        int64_t own = -1;
        for (size_t s = 0; s < S && slotted; ++s)
          if (cols[k][s] == ms[k].positionSize + ms[k].ordinal)
            own = s;
        slotted = slotted && own >= 0 && (k == 0 || own == ownSlot);
        ownSlot = own;
      }
      SmallVector<Value> newOf(S), oldOf(S);
      Value deltaVec, logAlphaVec;
      if (slotted) {
        for (size_t s = 0; s < S; ++s) {
          SmallVector<Value> proposed;
          for (unsigned k = 0; k < K; ++k)
            proposed.push_back(unitSlice(newTraces[k], cols[k][s]));
          newOf[s] = assembleVec(proposed);
          int64_t stride = cols[1][s] - cols[0][s];
          bool even = stride >= 1;
          for (unsigned k = 1; k < K && even; ++k)
            even = cols[k][s] - cols[k - 1][s] == stride;
          if (even) {
            oldOf[s] = impulse::ReshapeOp::create(
                b, loc, vecType,
                impulse::SliceOp::create(
                    b, loc, rowType, trace0,
                    b.getDenseI64ArrayAttr({0, cols[0][s]}),
                    b.getDenseI64ArrayAttr({1, cols[K - 1][s] + 1}),
                    b.getDenseI64ArrayAttr({1, stride})));
          } else {
            SmallVector<Value> incoming;
            for (unsigned k = 0; k < K; ++k)
              incoming.push_back(unitSlice(trace0, cols[k][s]));
            oldOf[s] = assembleVec(incoming);
          }
        }
        for (size_t s = 0; s < S; ++s) {
          if (cols[0][s] < ms[0].positionSize)
            continue;
          Value d = arith::SubFOp::create(b, loc, newOf[s], oldOf[s]);
          deltaVec =
              deltaVec ? arith::AddFOp::create(b, loc, deltaVec, d).getResult()
                       : d;
        }
        Value corrVec =
            arith::SubFOp::create(b, loc, newOf[ownSlot], oldOf[ownSlot]);
        logAlphaVec = arith::SubFOp::create(b, loc, deltaVec, corrVec);
      } else {
        deltaVec = assembleVec(deltaCells);
        logAlphaVec =
            arith::SubFOp::create(b, loc, deltaVec, assembleVec(corrCells));
      }
      Value accepted = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                             logRand, logAlphaVec);
      auto i1S = RankedTensorType::get({}, b.getI1Type());
      Value trace = trace0, weight = weight0;
      SmallVector<Value> acceptedOf;
      // In sweep order, matching the sequential weight updates.
      for (unsigned k = 0; k < K; ++k) {
        Value acc = element(accepted, k, i1S);
        acceptedOf.push_back(acc);
        Value acceptWeight = arith::AddFOp::create(
            b, loc, weight, element(deltaVec, k, weightType));
        weight = arith::SelectOp::create(b, loc, acc, acceptWeight, weight);
      }

      if (!slotted) {
        for (unsigned k = 0; k < K; ++k)
          trace = writeBackMHWindows(b, loc, trace, newTraces[k], acceptedOf[k],
                                     changed[k]);
      } else {
        SmallVector<Value> selOf(S);
        for (size_t s = 0; s < S; ++s)
          selOf[s] =
              arith::SelectOp::create(b, loc, accepted, newOf[s], oldOf[s]);
        // Slots within D columns share a window, viewed as K rows of D columns.
        int64_t D = cols[1][0] - cols[0][0];
        bool strided = D >= 1;
        for (size_t s = 0; s < S && strided; ++s)
          for (unsigned k = 0; k < K && strided; ++k)
            strided = cols[k][s] == cols[0][s] + (int64_t)k * D;
        SmallVector<std::pair<int64_t, SmallVector<size_t>>> groups;
        if (strided) {
          SmallVector<size_t> order(S);
          for (size_t i = 0; i < S; ++i)
            order[i] = i;
          llvm::sort(order, [&](size_t x, size_t y) {
            return cols[0][x] < cols[0][y];
          });
          for (size_t s : order) {
            if (!groups.empty() && cols[0][s] < groups.back().first + D)
              groups.back().second.push_back(s);
            else
              groups.push_back({cols[0][s], {s}});
          }
          for (size_t i = 0; i < groups.size() && strided; ++i) {
            int64_t end = groups[i].first + (int64_t)K * D;
            strided = end <= traceType.getShape()[1] &&
                      (i + 1 == groups.size() || end <= groups[i + 1].first);
          }
        }
        if (strided) {
          auto blockType = RankedTensorType::get({(int64_t)K, D}, elemType);
          auto blockColType = RankedTensorType::get({(int64_t)K, 1}, elemType);
          auto winType = RankedTensorType::get({1, (int64_t)K * D}, elemType);
          for (auto &[g, slots] : groups) {
            Value block = impulse::ReshapeOp::create(
                b, loc, blockType,
                impulse::SliceOp::create(
                    b, loc, winType, trace0, b.getDenseI64ArrayAttr({0, g}),
                    b.getDenseI64ArrayAttr({1, g + (int64_t)K * D}),
                    b.getDenseI64ArrayAttr({1, 1})));
            for (size_t s : slots) {
              Value slotCol =
                  impulse::ReshapeOp::create(b, loc, blockColType, selOf[s]);
              block = impulse::DynamicUpdateSliceOp::create(
                  b, loc, blockType, block, slotCol,
                  ValueRange{idx(0), idx(cols[0][s] - g)});
            }
            Value blockWin = impulse::ReshapeOp::create(b, loc, winType, block);
            trace = impulse::DynamicUpdateSliceOp::create(
                b, loc, traceType, trace, blockWin, ValueRange{idx(0), idx(g)});
          }
        } else {
          for (size_t s = 0; s < S; ++s) {
            Value row = impulse::ReshapeOp::create(b, loc, rowType, selOf[s]);
            for (unsigned k = 0; k < K; ++k) {
              Value unitCell = unitSlice(row, k);
              trace = impulse::DynamicUpdateSliceOp::create(
                  b, loc, traceType, trace, unitCell,
                  ValueRange{idx(0), idx(cols[k][s])});
            }
          }
        }
      }

      if (!bt.escaping.empty()) {
        IRMapping fm;
        for (Operation *o : bt.derived) {
          Operation *c = b.clone(*o, fm);
          for (unsigned i : readsIntermediate.lookup(o))
            c->setOperand(i, trace);
        }
        for (Value v : bt.escaping) {
          Value nv = fm.lookup(v);
          v.replaceUsesWithIf(nv, [&](OpOperand &u) {
            Operation *anc =
                last->getBlock()->findAncestorOpInBlock(*u.getOwner());
            return anc && last->isBeforeInBlock(anc);
          });
        }
      }

      for (unsigned k = 0; k < K; ++k) {
        impulse::InferRegionOp op = ms[k].op;
        op.getDiagnostics().replaceAllUsesWith(acceptedOf[k]);
        auto ldType = cast<ShapedType>(op.getLogDensities().getType());
        if (!op.getLogDensities().use_empty())
          op.getLogDensities().replaceAllUsesWith(arith::ConstantOp::create(
              b, loc, ldType,
              DenseElementsAttr::get(
                  ldType, b.getFloatAttr(ldType.getElementType(), 0.0))));
      }
      last.getTrace().replaceAllUsesWith(trace);
      last.getOutputRngState().replaceAllUsesWith(accepts.getOutputRngState());
      last.getStateOut()[0].replaceAllUsesWith(weight);
      for (unsigned k = K; k-- > 0;)
        ms[k].op->erase();
    }

    static void run(Operation *root, MLIRContext *ctx) {
      SmallVector<SmallVector<Move>> chains;
      root->walk([&](Block *block) {
        SmallVector<Move> chain;
        for (Operation &o : *block) {
          auto op = dyn_cast<impulse::InferRegionOp>(&o);
          if (!op)
            continue;
          std::optional<Move> m = analyzeMove(op);
          if (!m) {
            continue;
          }
          if (!chain.empty() && !chained(chain.back(), *m)) {
            if (chain.size() >= 2)
              chains.push_back(chain);
            chain.clear();
          }
          chain.push_back(*m);
        }
        if (chain.size() >= 2)
          chains.push_back(chain);
      });
      // Longest segment first: a shorter one can fail where a longer one holds.
      constexpr size_t kMaxSegment = 256;
      SmallVector<SmallVector<Move>> segments;
      for (auto &chain : chains) {
        size_t i = 0;
        while (i < chain.size()) {
          size_t j = std::min(chain.size(), i + kMaxSegment) - 1;
          for (; j > i; --j)
            if (batchable(ArrayRef<Move>(chain).slice(i, j + 1 - i)))
              break;
          if (j > i)
            segments.push_back(
                SmallVector<Move>(chain.begin() + i, chain.begin() + j + 1));
          i = j + 1;
        }
      }
      for (auto &seg : segments)
        emit(seg, ctx);
    }
  };

  // Enumerative Gibbs over scalar BERNOULLI sites: new = u < logistic(s1 - s0).
  // The batched sweep is sound only when each per-site yield depends on at
  // most one selected site and no site's parameters depend on another.
  struct LowerGibbsRegionPattern
      : public mlir::OpRewritePattern<impulse::InferRegionOp> {
    using mlir::OpRewritePattern<impulse::InferRegionOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(impulse::InferRegionOp op,
                                  PatternRewriter &rewriter) const override {
      if (!op.getGibbsConfig().has_value())
        return rewriter.notifyMatchFailure(op, "no gibbs_config");
      Location loc = op.getLoc();
      if (op.getSampler().empty() || op.getLogpdf().empty()) {
        op.emitError("Impulse Gibbs: infer_region requires sampler and "
                     "constructed logpdf regions (run sicm first)");
        return failure();
      }
      if (op.getStateIn().size() != 1 || op.getInputs().empty() ||
          !op.getOriginalTrace()) {
        op.emitError("Impulse Gibbs: expects the rng input, a trace and a "
                     "single `weight` state_in slot");
        return failure();
      }
      Block &samplerEntry = op.getSampler().front();
      Block &lpBlock = op.getLogpdf().front();
      Operation *lpTerm = lpBlock.getTerminator();
      unsigned numYields = lpTerm->getNumOperands() - 1;
      ArrayAttr selection = op.getSelectionAttr();
      ArrayAttr allAddresses = op.getAllAddressesAttr();

      DenseMap<Attribute, impulse::SampleRegionOp> siteByAddress;
      op.getSampler().walk([&](impulse::SampleRegionOp site) {
        if (auto address = site.getAddressAttr())
          siteByAddress[address] = site;
      });
      auto siteSize = [](impulse::SampleRegionOp site) {
        int64_t n = 0;
        for (unsigned i = 1, e = site.getNumResults(); i < e; ++i)
          n += cast<RankedTensorType>(site.getResult(i).getType())
                   .getNumElements();
        return n;
      };
      DenseMap<Attribute, int64_t> traceOffsetOf;
      int64_t positionSize = 0;
      for (Attribute addr : allAddresses) {
        impulse::SampleRegionOp site = siteByAddress.lookup(addr);
        if (!site) {
          op.emitError("Impulse Gibbs: all_addresses entry without a site");
          return failure();
        }
        traceOffsetOf[addr] = positionSize;
        positionSize += siteSize(site);
      }
      Value initTrace = op.getOriginalTrace();
      Value initWeight = op.getStateIn()[0];
      Value initRng = op.getInputs()[0];
      auto traceType = cast<RankedTensorType>(initTrace.getType());
      auto weightType = cast<RankedTensorType>(initWeight.getType());
      Type rngType = initRng.getType();
      int64_t width = traceType.getShape()[1];
      bool scoreTail = width > positionSize;
      if (scoreTail && width != positionSize + (int64_t)numYields) {
        op.emitError("Impulse Gibbs: score-tail width does not match the "
                     "logpdf region's per-site yields");
        return failure();
      }
      if (lpBlock.getNumArguments() != selection.size()) {
        op.emitError("Impulse Gibbs: one position argument per selected "
                     "site expected");
        return failure();
      }
      SmallVector<int64_t> selectedCols;
      for (auto [k, addr] : llvm::enumerate(selection)) {
        impulse::SampleRegionOp site = siteByAddress.lookup(addr);
        impulse::DistAttr dist = site ? site.getDistAttr() : nullptr;
        if (!site || site->getBlock() != &samplerEntry || !dist ||
            dist.getBase() != impulse::DistBaseKind::BERNOULLI ||
            (dist.getTransforms() && !dist.getTransforms().empty()) ||
            site.getNumResults() != 2 ||
            cast<RankedTensorType>(lpBlock.getArgument(k).getType())
                    .getNumElements() != 1) {
          op.emitError("Impulse Gibbs: every selected site must be a "
                       "top-level scalar BERNOULLI site");
          return failure();
        }
        selectedCols.push_back(traceOffsetOf.lookup(addr));
      }
      // (trace col, constraint offset, size) of constrained branch-local sites.
      SmallVector<std::tuple<int64_t, int64_t, int64_t>> observedBranchCols;
      if (Value constraint = op.getConstraint()) {
        int64_t off = 0;
        for (Attribute addr : op.getConstrainedAddressesAttr()) {
          impulse::SampleRegionOp site = siteByAddress.lookup(addr);
          if (!site) {
            op.emitError("Impulse Gibbs: constrained address without a site");
            return failure();
          }
          if (site->getBlock() != &samplerEntry)
            observedBranchCols.push_back(
                {traceOffsetOf.lookup(addr), off, siteSize(site)});
          off += siteSize(site);
        }
      }
      SmallVector<SmallVector<bool>> dirtyOf;
      for (unsigned k = 0; k < selection.size(); ++k)
        dirtyOf.push_back(
            taintedSiteYields(lpBlock, ValueRange{lpBlock.getArgument(k)}));

      // siteOfYield[j]: the selected site yield j depends on, or -1 for none.
      unsigned numSites = selection.size();
      SmallVector<int> siteOfYield(numYields, -1);
      bool batched = numSites >= 2;
      if (batched) {
        GibbsSiteDependence lpDep(numSites);
        for (unsigned k = 0; k < numSites; ++k)
          lpDep.seed(lpBlock.getArgument(k), k);
        lpDep.run(lpBlock);
        for (unsigned j = 0; j < numYields && batched; ++j) {
          llvm::BitVector s = lpDep.get(lpTerm->getOperand(1 + j));
          if (s.count() > 1)
            batched = false;
          else if (s.any())
            siteOfYield[j] = s.find_first();
        }
      }
      if (batched) {
        GibbsSiteDependence samplerDep(numSites);
        SmallVector<impulse::SampleRegionOp> selectedSites;
        for (auto [k, addr] : llvm::enumerate(selection)) {
          impulse::SampleRegionOp site = siteByAddress.lookup(addr);
          selectedSites.push_back(site);
          samplerDep.seed(site.getResult(1), k);
        }
        samplerDep.run(samplerEntry);
        for (auto [k, site] : llvm::enumerate(selectedSites))
          for (Value param : llvm::drop_begin(site->getOperands()))
            if (samplerDep.get(param).any())
              batched = false;
      }

      auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
      auto idx = [&](OpBuilder &b, int64_t v) -> Value {
        return arith::ConstantOp::create(
            b, loc, i64S, DenseElementsAttr::get(i64S, b.getI64IntegerAttr(v)));
      };
      auto fconst = [&](OpBuilder &b, RankedTensorType t, double v) -> Value {
        return arith::ConstantOp::create(b, loc, t,
                                         DenseElementsAttr::get(t, v));
      };
      auto writeCols = [&](OpBuilder &b, Value trace, int64_t col,
                           Value value) -> Value {
        auto vt = cast<RankedTensorType>(value.getType());
        auto flatType = RankedTensorType::get({1, vt.getNumElements()},
                                              traceType.getElementType());
        Value flat = impulse::ReshapeOp::create(b, loc, flatType, value);
        return impulse::DynamicUpdateSliceOp::create(
            b, loc, traceType, trace, flat, ValueRange{idx(b, 0), idx(b, col)});
      };

      // For later regions (e.g. HMC) that pin these sites from the trace.
      auto writeObserved = [&](OpBuilder &b, Value trace) -> Value {
        Value constraint = op.getConstraint();
        for (auto [col, off, n] : observedBranchCols) {
          Value obs = impulse::SliceOp::create(
              b, loc, RankedTensorType::get({1, n}, traceType.getElementType()),
              constraint, b.getDenseI64ArrayAttr({0, off}),
              b.getDenseI64ArrayAttr({1, off + n}),
              b.getDenseI64ArrayAttr({1, 1}));
          trace = writeCols(b, trace, col, obs);
        }
        return trace;
      };

      auto batchedSweep = [&](OpBuilder &b, Value trace, Value weight,
                              Value rng) -> std::array<Value, 3> {
        int64_t n = numSites;
        Type elemType = traceType.getElementType();
        auto scalarType = RankedTensorType::get({}, elemType);
        auto elemVecType = RankedTensorType::get({1}, elemType);
        auto vecType = RankedTensorType::get({n}, elemType);
        auto rowType = RankedTensorType::get({1, n}, elemType);
        auto colType = RankedTensorType::get({1, 1}, elemType);
        auto element = [&](Value vec, int64_t k) -> Value {
          auto vt = cast<RankedTensorType>(vec.getType());
          Value sl = impulse::SliceOp::create(
              b, loc, RankedTensorType::get({1}, vt.getElementType()), vec,
              b.getDenseI64ArrayAttr({k}), b.getDenseI64ArrayAttr({k + 1}),
              b.getDenseI64ArrayAttr({1}));
          return impulse::ReshapeOp::create(
              b, loc, RankedTensorType::get({}, vt.getElementType()), sl);
        };
        auto assemble = [&](ArrayRef<Value> scalars) -> Value {
          Value vec = fconst(b, vecType, 0.0);
          for (auto [k, s] : llvm::enumerate(scalars)) {
            Value elemCell = impulse::ReshapeOp::create(b, loc, elemVecType, s);
            vec = impulse::DynamicUpdateSliceOp::create(
                b, loc, vecType, vec, elemCell, ValueRange{idx(b, k)});
          }
          return vec;
        };
        int64_t stride = n > 1 ? selectedCols[1] - selectedCols[0] : 1;
        bool evenlySpaced = stride >= 1;
        for (int64_t k = 1; k < n && evenlySpaced; ++k)
          evenlySpaced = selectedCols[k] - selectedCols[k - 1] == stride;

        Value curVec;
        if (evenlySpaced) {
          Value row = impulse::SliceOp::create(
              b, loc, rowType, trace,
              b.getDenseI64ArrayAttr({0, selectedCols[0]}),
              b.getDenseI64ArrayAttr({1, selectedCols[n - 1] + 1}),
              b.getDenseI64ArrayAttr({1, stride}));
          curVec = impulse::ReshapeOp::create(b, loc, vecType, row);
        } else {
          SmallVector<Value> cur;
          for (int64_t col : selectedCols)
            cur.push_back(impulse::ReshapeOp::create(
                b, loc, scalarType,
                impulse::SliceOp::create(b, loc, colType, trace,
                                         b.getDenseI64ArrayAttr({0, col}),
                                         b.getDenseI64ArrayAttr({1, col + 1}),
                                         b.getDenseI64ArrayAttr({1, 1}))));
          curVec = assemble(cur);
        }

        // All sites at v at once: each yield sees at most one selected site.
        SmallVector<Value> yieldsAt[2];
        for (int v = 0; v < 2; ++v) {
          IRMapping m;
          for (unsigned j = 0; j < lpBlock.getNumArguments(); ++j)
            m.map(
                lpBlock.getArgument(j),
                fconst(b,
                       cast<RankedTensorType>(lpBlock.getArgument(j).getType()),
                       (double)v));
          for (Operation &lo : lpBlock.without_terminator())
            b.clone(lo, m);
          for (unsigned j = 0; j < numYields; ++j)
            yieldsAt[v].push_back(m.lookupOrDefault(lpTerm->getOperand(1 + j)));
        }
        SmallVector<Value> logOdds(n);
        for (unsigned j = 0; j < numYields; ++j) {
          int k = siteOfYield[j];
          if (k < 0)
            continue;
          Value d =
              arith::SubFOp::create(b, loc, yieldsAt[1][j], yieldsAt[0][j]);
          logOdds[k] =
              logOdds[k]
                  ? arith::AddFOp::create(b, loc, logOdds[k], d).getResult()
                  : d;
        }
        for (Value &l : logOdds)
          if (!l)
            l = fconst(b, scalarType, 0.0);
        Value lVec = assemble(logOdds);

        Value p1 = impulse::LogisticOp::create(b, loc, vecType, lVec);
        Value uLo = fconst(b, scalarType, 0.0);
        Value uHi = fconst(b, scalarType, 1.0);
        auto rnd = impulse::RandomOp::create(
            b, loc, TypeRange{rngType, vecType}, rng, uLo, uHi,
            impulse::RngDistributionAttr::get(
                b.getContext(), impulse::RngDistribution::UNIFORM));
        Value take1 = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                            rnd.getResult(), p1);
        Value curIs1 = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT,
                                             curVec, fconst(b, vecType, 0.5));
        Value zeros = fconst(b, vecType, 0.0);
        Value newVals = arith::SelectOp::create(b, loc, take1,
                                                fconst(b, vecType, 1.0), zeros);

        // +L_k on a 0 -> 1 flip, -L_k on 1 -> 0, else 0.
        Value negL = arith::NegFOp::create(b, loc, lVec);
        Value toOne = arith::SelectOp::create(b, loc, curIs1, zeros, lVec);
        Value toZero = arith::SelectOp::create(b, loc, curIs1, negL, zeros);
        Value delta = arith::SelectOp::create(b, loc, take1, toOne, toZero);
        for (int64_t k = 0; k < n; ++k)
          weight = arith::AddFOp::create(b, loc, weight, element(delta, k));

        Value newRow = impulse::ReshapeOp::create(b, loc, rowType, newVals);
        if (evenlySpaced && stride == 1) {
          trace = impulse::DynamicUpdateSliceOp::create(
              b, loc, traceType, trace, newRow,
              ValueRange{idx(b, 0), idx(b, selectedCols[0])});
        } else {
          for (auto [k, col] : llvm::enumerate(selectedCols)) {
            Value colSlice = impulse::SliceOp::create(
                b, loc, colType, newRow,
                b.getDenseI64ArrayAttr({0, (int64_t)k}),
                b.getDenseI64ArrayAttr({1, (int64_t)k + 1}),
                b.getDenseI64ArrayAttr({1, 1}));
            trace = impulse::DynamicUpdateSliceOp::create(
                b, loc, traceType, trace, colSlice,
                ValueRange{idx(b, 0), idx(b, col)});
          }
        }
        if (scoreTail)
          for (unsigned j = 0; j < numYields; ++j)
            if (siteOfYield[j] >= 0)
              trace = writeCols(b, trace, positionSize + j,
                                arith::SelectOp::create(
                                    b, loc, element(take1, siteOfYield[j]),
                                    yieldsAt[1][j], yieldsAt[0][j]));
        return {writeObserved(b, trace), weight, rnd.getOutputRngState()};
      };

      auto oneSweep = [&](OpBuilder &b, Value trace, Value weight,
                          Value rng) -> std::array<Value, 3> {
        if (batched)
          return batchedSweep(b, trace, weight, rng);
        SmallVector<Value> cur;
        for (auto [k, col] : llvm::enumerate(selectedCols)) {
          auto argType =
              cast<RankedTensorType>(lpBlock.getArgument(k).getType());
          Value sl = impulse::SliceOp::create(
              b, loc, RankedTensorType::get({1, 1}, traceType.getElementType()),
              trace, b.getDenseI64ArrayAttr({0, col}),
              b.getDenseI64ArrayAttr({1, col + 1}),
              b.getDenseI64ArrayAttr({1, 1}));
          cur.push_back(impulse::ReshapeOp::create(b, loc, argType, sl));
        }
        auto scalarType = RankedTensorType::get({}, traceType.getElementType());
        for (unsigned k = 0; k < selectedCols.size(); ++k) {
          auto argType =
              cast<RankedTensorType>(lpBlock.getArgument(k).getType());
          SmallVector<Value> yieldsAt[2];
          Value scoreAt[2];
          for (int v = 0; v < 2; ++v) {
            IRMapping m;
            for (unsigned j = 0; j < lpBlock.getNumArguments(); ++j)
              m.map(lpBlock.getArgument(j),
                    j == k ? fconst(b, argType, (double)v) : cur[j]);
            for (Operation &lo : lpBlock.without_terminator())
              b.clone(lo, m);
            for (unsigned j = 0; j < numYields; ++j) {
              Value y = m.lookupOrDefault(lpTerm->getOperand(1 + j));
              yieldsAt[v].push_back(y);
              if (!dirtyOf[k][j])
                continue;
              scoreAt[v] =
                  scoreAt[v]
                      ? arith::AddFOp::create(b, loc, scoreAt[v], y).getResult()
                      : y;
            }
            if (!scoreAt[v])
              scoreAt[v] = fconst(b, scalarType, 0.0);
          }
          Value diff = arith::SubFOp::create(b, loc, scoreAt[1], scoreAt[0]);
          Value p1 = impulse::LogisticOp::create(b, loc, diff.getType(), diff);
          Value zero = fconst(b, weightType, 0.0);
          Value one = fconst(b, weightType, 1.0);
          auto rnd = impulse::RandomOp::create(
              b, loc, TypeRange{rngType, weightType}, rng, zero, one,
              impulse::RngDistributionAttr::get(
                  b.getContext(), impulse::RngDistribution::UNIFORM));
          rng = rnd.getOutputRngState();
          Value take1 = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                              rnd.getResult(), p1);
          Value curScalar =
              impulse::ReshapeOp::create(b, loc, scalarType, cur[k]);
          Value curIs1 =
              arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT,
                                    curScalar, fconst(b, scalarType, 0.5));
          Value sNew =
              arith::SelectOp::create(b, loc, take1, scoreAt[1], scoreAt[0]);
          Value sCur =
              arith::SelectOp::create(b, loc, curIs1, scoreAt[1], scoreAt[0]);
          weight = arith::AddFOp::create(
              b, loc, weight, arith::SubFOp::create(b, loc, sNew, sCur));
          Value oneArg = fconst(b, argType, 1.0);
          Value zeroArg = fconst(b, argType, 0.0);
          Value newVal =
              arith::SelectOp::create(b, loc, take1, oneArg, zeroArg);
          cur[k] = newVal;
          trace = writeCols(b, trace, selectedCols[k], newVal);
          if (scoreTail)
            for (unsigned j = 0; j < numYields; ++j)
              if (dirtyOf[k][j])
                trace = writeCols(b, trace, positionSize + j,
                                  arith::SelectOp::create(b, loc, take1,
                                                          yieldsAt[1][j],
                                                          yieldsAt[0][j]));
        }
        return {writeObserved(b, trace), weight, rng};
      };

      int64_t numSamples = op.getNumSamples();
      Value outTrace, outWeight, outRng;
      if (numSamples <= 1) {
        auto r = oneSweep(rewriter, initTrace, initWeight, initRng);
        outTrace = r[0];
        outWeight = r[1];
        outRng = r[2];
      } else {
        Value c0 = idx(rewriter, 0), cN = idx(rewriter, numSamples),
              c1 = idx(rewriter, 1);
        SmallVector<Type> loopTypes{traceType, weightType, rngType};
        SmallVector<Value> initArgs{initTrace, initWeight, initRng};
        auto forOp = impulse::ForOp::create(rewriter, loc, loopTypes, c0, cN,
                                            c1, initArgs);
        Block *body = rewriter.createBlock(&forOp.getRegion());
        body->addArgument(i64S, loc);
        for (Type t : loopTypes)
          body->addArgument(t, loc);
        rewriter.setInsertionPointToStart(body);
        auto r = oneSweep(rewriter, body->getArgument(1), body->getArgument(2),
                          body->getArgument(3));
        impulse::YieldOp::create(rewriter, loc, ValueRange{r[0], r[1], r[2]});
        rewriter.setInsertionPointAfter(forOp);
        outTrace = forOp.getResult(0);
        outWeight = forOp.getResult(1);
        outRng = forOp.getResult(2);
      }

      // Gibbs always accepts; `log_densities` is a zero placeholder.
      auto diagType = cast<ShapedType>(op.getDiagnostics().getType());
      Value diagnostics = arith::ConstantOp::create(
          rewriter, loc, diagType,
          DenseElementsAttr::get(diagType, rewriter.getBoolAttr(true)));
      auto ldType = cast<ShapedType>(op.getLogDensities().getType());
      Value logDensities = arith::ConstantOp::create(
          rewriter, loc, ldType,
          DenseElementsAttr::get(
              ldType, rewriter.getFloatAttr(ldType.getElementType(), 0.0)));
      rewriter.replaceOp(
          op, {outTrace, diagnostics, logDensities, outRng, outWeight});
      return success();
    }
  };

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

  struct LowerMHPattern : public mlir::OpRewritePattern<impulse::MHOp> {
    using mlir::OpRewritePattern<impulse::MHOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(impulse::MHOp mhOp,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(mhOp, mhOp.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        mhOp.emitError(
            "Impulse: calling `mh` on an empty function; if this is a "
            "distribution function, its sample op should have a logpdf "
            "attribute to avoid recursive `mh` calls which is intended for "
            "generative functions");
        return failure();
      }

      auto loc = mhOp.getLoc();

      Value oldTrace = mhOp.getOriginalTrace();
      Value oldWeight = mhOp.getOriginalWeight();
      SmallVector<Value> inputs(mhOp.getInputs().begin(),
                                mhOp.getInputs().end());
      auto selection = mhOp.getSelectionAttr();

      auto traceType = oldTrace.getType();
      auto weightType = cast<RankedTensorType>(oldWeight.getType());
      auto rngStateType = inputs[0].getType();

      // 1. Create regenerate op with the same function and selection
      auto nameAttr = mhOp.getNameAttr();
      if (!nameAttr)
        nameAttr = rewriter.getStringAttr("");

      auto regenerateAddresses = mhOp.getRegenerateAddressesAttr();

      // On a score-tail trace, regenerate returns the dirty-term weight delta.
      int64_t fullPositionSize =
          computePositionSizeForSelection(mhOp, fn, selection, symbolTable);
      bool delta =
          cast<RankedTensorType>(traceType).getShape()[1] > fullPositionSize;

      SmallVector<Type> regenResultTypes;
      regenResultTypes.push_back(traceType);
      regenResultTypes.push_back(
          RankedTensorType::get({2}, rewriter.getF64Type()));
      for (auto t : fn.getResultTypes())
        regenResultTypes.push_back(t);

      auto regenerateOp = rewriter.create<impulse::RegenerateOp>(
          loc,
          /*resultTypes*/ regenResultTypes,
          /*fn*/ mhOp.getFnAttr(),
          /*inputs*/ inputs,
          /*original_trace*/ oldTrace,
          /*constraint*/ mhOp.getConstraint(),
          /*selection*/ selection,
          /*regenerate_addresses*/ regenerateAddresses,
          /*constrained_addresses*/ mhOp.getConstrainedAddressesAttr(),
          /*name*/ nameAttr);

      regenerateOp->setAttr(kMHHastingsAttr, rewriter.getUnitAttr());
      Value newTrace = regenerateOp.getNewTrace();
      auto [newWeight, corr] =
          unpackHastingsWeight(rewriter, loc, regenerateOp.getWeight());
      Value newRng = regenerateOp.getOutputs()[0];

      // 2. Compute log_alpha = new_weight - old_weight - corr
      Value jointRatio =
          delta ? newWeight
                : arith::SubFOp::create(rewriter, loc, newWeight, oldWeight)
                      .getResult();
      Value logAlpha = arith::SubFOp::create(rewriter, loc, jointRatio, corr);
      Value acceptWeight =
          delta ? arith::AddFOp::create(rewriter, loc, oldWeight, newWeight)
                      .getResult()
                : newWeight;

      // 3. Sample uniform random in (0, 1) and compute log
      auto zeroConst = arith::ConstantOp::create(
          rewriter, loc, weightType, DenseElementsAttr::get(weightType, 0.0));
      auto oneConst = arith::ConstantOp::create(
          rewriter, loc, weightType, DenseElementsAttr::get(weightType, 1.0));

      auto randomOp = impulse::RandomOp::create(
          rewriter, loc, TypeRange{rngStateType, weightType}, newRng, zeroConst,
          oneConst,
          impulse::RngDistributionAttr::get(rewriter.getContext(),
                                            impulse::RngDistribution::UNIFORM));
      auto logRand = math::LogOp::create(rewriter, loc, randomOp.getResult());
      Value finalRng = randomOp.getOutputRngState();

      // 4. Check if proposal is accepted: log(rand()) < log_alpha
      auto accepted = arith::CmpFOp::create(
          rewriter, loc, arith::CmpFPredicate::OLT, logRand, logAlpha);

      // 5. Select trace and weight based on acceptance
      Value selectedTrace = writeBackMHWindows(
          rewriter, loc, oldTrace, newTrace, accepted,
          mhChangedColumnsFlat(mhOp, fn, selection, regenerateAddresses,
                               mhOp.getConstrainedAddressesAttr(),
                               static_cast<bool>(mhOp.getConstraint()),
                               cast<RankedTensorType>(traceType).getShape()[1],
                               symbolTable));
      auto selectedWeight = arith::SelectOp::create(rewriter, loc, accepted,
                                                    acceptWeight, oldWeight);

      rewriter.replaceOp(mhOp,
                         {selectedTrace, selectedWeight, accepted, finalRng});
      return success();
    }
  };

  struct LowerGeneratePattern
      : public mlir::OpRewritePattern<impulse::GenerateOp> {
    using mlir::OpRewritePattern<impulse::GenerateOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(impulse::GenerateOp CI,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        CI.emitError(
            "Impulse: calling `generate` on an empty function; if this "
            "is a distribution function, its sample op should have a "
            "logpdf attribute to avoid recursive `generate` calls which is "
            "intended for generative functions");
        return failure();
      }

      ArrayAttr selection = CI.getSelectionAttr();
      int64_t positionSize =
          computePositionSizeForSelection(CI, fn, selection, symbolTable);
      if (positionSize <= 0) {
        CI.emitError("Impulse: failed to compute position size for generate");
        return failure();
      }

      // Operand width, not the packed size: extra columns are simply ignored.
      int64_t constraintSize =
          cast<RankedTensorType>(CI.getConstraint().getType()).getShape()[1];

      int64_t declaredWidth =
          cast<RankedTensorType>(CI.getTrace().getType()).getShape()[1];
      int64_t scoreTailSize = declaredWidth - positionSize;
      SmallVector<Value> siteScores;
      auto putils = ImpulseUtils::CreateFromClone(
          fn, ImpulseMode::Generate, declaredWidth, constraintSize);
      FunctionOpInterface NewF = putils->newFunc;

      OpBuilder entryBuilder(putils->initializationBlock,
                             putils->initializationBlock->begin());
      Location initLoc = putils->initializationBlock->begin()->getLoc();

      auto scalarType = RankedTensorType::get({}, entryBuilder.getF64Type());
      auto zeroWeight =
          arith::ConstantOp::create(entryBuilder, initLoc, scalarType,
                                    DenseElementsAttr::get(scalarType, 0.0));
      Value weightAccumulator = zeroWeight;

      auto traceType =
          RankedTensorType::get({1, positionSize}, entryBuilder.getF64Type());
      auto zeroTrace =
          arith::ConstantOp::create(entryBuilder, initLoc, traceType,
                                    DenseElementsAttr::get(traceType, 0.0));
      Value currTrace = zeroTrace;
      Value constraint = NewF.getArgument(0);
      int64_t currentTraceOffset = 0;

      SmallVector<Operation *> toErase;
      BranchCarry carry{currTrace, weightAccumulator, currentTraceOffset};
      auto processSample = [&](impulse::SampleOp sampleOp,
                               BranchCarry &carry) -> WalkResult {
        Value &currTrace = carry.currTrace;
        Value &weightAccumulator = carry.weight;
        int64_t &currentTraceOffset = carry.offset;
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sampleOp);

        SmallVector<Value> sampledValues;

        if (auto dist = sampleOp.getDistAttr()) {
          Location loc = sampleOp.getLoc();
          Value rng = sampleOp.getInputs().front();
          SmallVector<Value> distArgs(std::next(sampleOp.getInputs().begin()),
                                      sampleOp.getInputs().end());
          auto distSampleType =
              cast<RankedTensorType>(sampleOp.getResult(1).getType());

          bool isConstrained = false;
          int64_t constrainedOffset = -1;
          for (auto addr : CI.getConstrainedAddressesAttr()) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              if (address.size() != 1) {
                sampleOp.emitError("Impulse: distribution function cannot have "
                                   "composite constrained address");
                return WalkResult::interrupt();
              }
              isConstrained = true;
              constrainedOffset = computeOffsetForSampleInSelection(
                  CI, fn, CI.getConstrainedAddressesAttr(),
                  sampleOp.getSymbolAttr(), symbolTable);
              break;
            }
          }

          sampledValues.resize(sampleOp.getNumResults());
          if (isConstrained) {
            sampledValues[0] = rng;
            for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
              auto resultType =
                  cast<RankedTensorType>(sampleOp.getResult(i).getType());
              int64_t numElements = computeTensorElementCount(resultType);
              if (numElements < 0) {
                sampleOp.emitError(
                    "Impulse: dynamic tensor dimensions not supported");
                return WalkResult::interrupt();
              }
              auto sliceType = RankedTensorType::get(
                  {1, numElements}, resultType.getElementType());
              auto sliced = impulse::SliceOp::create(
                  rewriter, loc, sliceType, constraint,
                  rewriter.getDenseI64ArrayAttr({0, constrainedOffset}),
                  rewriter.getDenseI64ArrayAttr(
                      {1, constrainedOffset + numElements}),
                  rewriter.getDenseI64ArrayAttr({1, 1}));
              sampledValues[i] =
                  impulse::ReshapeOp::create(rewriter, loc, resultType, sliced)
                      .getResult();
              constrainedOffset += numElements;
            }
          } else {
            if (!impulse::distSamplerEmittable(dist)) {
              sampleOp.emitError("Impulse: distribution sampler is not "
                                 "emittable; the site must be constrained");
              return WalkResult::interrupt();
            }
            auto sv = impulse::emitDistSampler(rewriter, loc, dist, rng,
                                               distArgs, distSampleType);
            for (unsigned i = 0; i < sv.size(); ++i)
              sampledValues[i] = sv[i];
          }

          Value logpdf = impulse::emitDistLogpdf(rewriter, loc, dist,
                                                 sampledValues[1], distArgs);
          weightAccumulator =
              arith::AddFOp::create(rewriter, loc, weightAccumulator, logpdf);
          if (scoreTailSize > 0)
            siteScores.push_back(logpdf);

          bool inSelection = false;
          for (auto addr : selection) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              inSelection = true;
              break;
            }
          }
          if (inSelection) {
            for (unsigned i = 1; i < sampledValues.size(); ++i) {
              auto sampleValue = sampledValues[i];
              auto sampleTy = cast<RankedTensorType>(sampleValue.getType());
              int64_t numElements = computeTensorElementCount(sampleTy);
              if (numElements < 0) {
                sampleOp.emitError(
                    "Impulse: dynamic tensor dimensions not supported");
                return WalkResult::interrupt();
              }
              auto flatSampleType = RankedTensorType::get(
                  {1, numElements}, sampleTy.getElementType());
              auto flatSample = impulse::ReshapeOp::create(
                  rewriter, loc, flatSampleType, sampleValue);
              auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
              auto row0 = arith::ConstantOp::create(
                  rewriter, loc, i64S,
                  DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(0)));
              auto colOff = arith::ConstantOp::create(
                  rewriter, loc, i64S,
                  DenseElementsAttr::get(
                      i64S, rewriter.getI64IntegerAttr(currentTraceOffset)));
              currTrace = impulse::DynamicUpdateSliceOp::create(
                              rewriter, loc, traceType, currTrace, flatSample,
                              ValueRange{row0, colOff})
                              .getResult();
              currentTraceOffset += numElements;
            }
          }

          sampleOp.replaceAllUsesWith(sampledValues);
          toErase.push_back(sampleOp);
          return WalkResult::advance();
        }

        bool isDistribution = static_cast<bool>(sampleOp.getLogpdfAttr());

        if (isDistribution) {
          // A1. Distribution function: call the distribution function.
          bool isConstrained = false;
          int64_t constrainedOffset = -1;
          for (auto addr : CI.getConstrainedAddressesAttr()) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              if (address.size() != 1) {
                sampleOp.emitError(
                    "Impulse: distribution function cannot have composite "
                    "constrained address");
                return WalkResult::interrupt();
              }
              isConstrained = true;
              constrainedOffset = computeOffsetForSampleInSelection(
                  CI, fn, CI.getConstrainedAddressesAttr(),
                  sampleOp.getSymbolAttr(), symbolTable);
              break;
            }
          }

          if (isConstrained) {
            // Extract sampled values from constraint tensor
            sampledValues.resize(sampleOp.getNumResults());
            sampledValues[0] = sampleOp.getOperand(0); // RNG state

            for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
              auto resultType =
                  cast<RankedTensorType>(sampleOp.getResult(i).getType());
              int64_t numElements = computeTensorElementCount(resultType);
              if (numElements < 0) {
                sampleOp.emitError(
                    "Impulse: dynamic tensor dimensions not supported");
                return WalkResult::interrupt();
              }

              auto sliceType = RankedTensorType::get(
                  {1, numElements}, resultType.getElementType());
              auto sliced = impulse::SliceOp::create(
                  rewriter, sampleOp.getLoc(), sliceType, constraint,
                  rewriter.getDenseI64ArrayAttr({0, constrainedOffset}),
                  rewriter.getDenseI64ArrayAttr(
                      {1, constrainedOffset + numElements}),
                  rewriter.getDenseI64ArrayAttr({1, 1}));
              auto extracted = impulse::ReshapeOp::create(
                  rewriter, sampleOp.getLoc(), resultType, sliced);
              sampledValues[i] = extracted.getResult();
              constrainedOffset += numElements;
            }

            // Compute weight via logpdf using constrained values.
            auto logpdfFn =
                cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                    sampleOp, sampleOp.getLogpdfAttr()));

            // logpdf operands: (<non-RNG outputs>..., <non-RNG inputs>...)
            SmallVector<Value> logpdfOperands;
            for (unsigned i = 1; i < sampledValues.size(); ++i) {
              logpdfOperands.push_back(sampledValues[i]);
            }
            for (unsigned i = 1; i < sampleOp.getNumOperands(); ++i) {
              logpdfOperands.push_back(sampleOp.getOperand(i));
            }

            if (logpdfOperands.size() != logpdfFn.getNumArguments()) {
              sampleOp.emitError(
                  "Impulse: failed to construct logpdf call for constrained "
                  "sample; logpdf function has wrong number of arguments");
              return WalkResult::interrupt();
            }

            auto logpdf = func::CallOp::create(
                rewriter, sampleOp.getLoc(), logpdfFn.getName(),
                logpdfFn.getResultTypes(), logpdfOperands);
            weightAccumulator =
                arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                      weightAccumulator, logpdf.getResult(0));
            if (scoreTailSize > 0)
              siteScores.push_back(logpdf.getResult(0));
          } else {
            // Unconstrained: call the distribution function
            auto distFn =
                cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                    sampleOp, sampleOp.getFnAttr()));

            auto distCall = func::CallOp::create(
                rewriter, sampleOp.getLoc(), distFn.getName(),
                distFn.getResultTypes(), sampleOp.getInputs());

            sampledValues.append(distCall.getResults().begin(),
                                 distCall.getResults().end());

            auto logpdfFn =
                cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                    sampleOp, sampleOp.getLogpdfAttr()));

            SmallVector<Value> logpdfOperands;
            for (unsigned i = 1; i < sampledValues.size(); ++i) {
              logpdfOperands.push_back(sampledValues[i]);
            }
            for (unsigned i = 1; i < sampleOp.getNumOperands(); ++i) {
              logpdfOperands.push_back(sampleOp.getOperand(i));
            }

            if (logpdfOperands.size() != logpdfFn.getNumArguments()) {
              sampleOp.emitError(
                  "Impulse: failed to construct logpdf call; "
                  "logpdf function has wrong number of arguments");
              return WalkResult::interrupt();
            }

            auto logpdf = func::CallOp::create(
                rewriter, sampleOp.getLoc(), logpdfFn.getName(),
                logpdfFn.getResultTypes(), logpdfOperands);
            weightAccumulator =
                arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                      weightAccumulator, logpdf.getResult(0));
            if (scoreTailSize > 0)
              siteScores.push_back(logpdf.getResult(0));
          }

          bool inSelection = false;
          for (auto addr : selection) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              inSelection = true;
              break;
            }
          }

          if (inSelection) {
            for (unsigned i = 1; i < sampledValues.size(); ++i) {
              auto sampleValue = sampledValues[i];
              auto sampleType = cast<RankedTensorType>(sampleValue.getType());
              int64_t numElements = computeTensorElementCount(sampleType);
              if (numElements < 0) {
                sampleOp.emitError(
                    "Impulse: dynamic tensor dimensions not supported");
                return WalkResult::interrupt();
              }

              auto flatSampleType = RankedTensorType::get(
                  {1, numElements}, sampleType.getElementType());
              auto flatSample = impulse::ReshapeOp::create(
                  rewriter, sampleOp.getLoc(), flatSampleType, sampleValue);
              auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
              auto row0 = arith::ConstantOp::create(
                  rewriter, sampleOp.getLoc(), i64S,
                  DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(0)));
              auto colOff = arith::ConstantOp::create(
                  rewriter, sampleOp.getLoc(), i64S,
                  DenseElementsAttr::get(
                      i64S, rewriter.getI64IntegerAttr(currentTraceOffset)));
              currTrace = impulse::DynamicUpdateSliceOp::create(
                              rewriter, sampleOp.getLoc(), traceType, currTrace,
                              flatSample, ValueRange{row0, colOff})
                              .getResult();
              currentTraceOffset += numElements;
            }
          }
        } else {
          // B. Generative function: recursively generate the nested function
          if (scoreTailSize > 0) {
            sampleOp.emitError("Impulse: score-tail traces do not support "
                               "nested generative sites");
            return WalkResult::interrupt();
          }
          auto genFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getFnAttr()));

          if (genFn.getFunctionBody().empty()) {
            sampleOp.emitError(
                "Impulse: generative function body is empty; "
                "if this is a distribution, add a logpdf attribute");
            return WalkResult::interrupt();
          }

          ArrayAttr subSelection =
              buildSubSelection(rewriter, selection, sampleOp.getSymbolAttr());
          ArrayAttr subConstrainedAddrs =
              buildSubSelection(rewriter, CI.getConstrainedAddressesAttr(),
                                sampleOp.getSymbolAttr());

          if (subSelection.empty()) {
            // No samples from this generative function are in the selection
            // Just call the function directly
            auto genCall = func::CallOp::create(
                rewriter, sampleOp.getLoc(), genFn.getName(),
                genFn.getResultTypes(), sampleOp.getInputs());
            sampledValues.append(genCall.getResults().begin(),
                                 genCall.getResults().end());
          } else {
            int64_t subPositionSize = computePositionSizeForSelection(
                sampleOp, genFn, subSelection, symbolTable);
            int64_t subConstraintSize = computePositionSizeForSelection(
                sampleOp, genFn, subConstrainedAddrs, symbolTable);
            if (subPositionSize <= 0 || subConstraintSize < 0) {
              sampleOp.emitError("Impulse: failed to compute sub-position or "
                                 "sub-constraint size");
              return WalkResult::interrupt();
            }

            Value subConstraint;
            auto subConstraintType = RankedTensorType::get(
                {1, subConstraintSize}, rewriter.getF64Type());

            if (subConstraintSize > 0) {
              int64_t subConstraintOffset = computeOffsetForNestedSample(
                  sampleOp, fn, CI.getConstrainedAddressesAttr(),
                  sampleOp.getSymbolAttr(), symbolTable);

              subConstraint = impulse::SliceOp::create(
                  rewriter, sampleOp.getLoc(), subConstraintType, constraint,
                  rewriter.getDenseI64ArrayAttr({0, subConstraintOffset}),
                  rewriter.getDenseI64ArrayAttr(
                      {1, subConstraintOffset + subConstraintSize}),
                  rewriter.getDenseI64ArrayAttr({1, 1}));
            } else {
              subConstraint = arith::ConstantOp::create(
                  rewriter, sampleOp.getLoc(), subConstraintType,
                  DenseElementsAttr::get(subConstraintType, {0.0}));
            }

            // Build result types: (trace, weight, original_returns...)
            auto subTraceType = RankedTensorType::get({1, subPositionSize},
                                                      rewriter.getF64Type());
            auto scalarTy = RankedTensorType::get({}, rewriter.getF64Type());
            SmallVector<Type> genResultTypes;
            genResultTypes.push_back(subTraceType);
            genResultTypes.push_back(scalarTy);
            for (auto t : genFn.getResultTypes())
              genResultTypes.push_back(t);

            auto nestedGenerate = impulse::GenerateOp::create(
                rewriter, sampleOp.getLoc(), genResultTypes,
                sampleOp.getFnAttr(), sampleOp.getInputs(), subConstraint,
                subSelection, subConstrainedAddrs);

            Value subTrace = nestedGenerate.getTrace();
            Value subWeight = nestedGenerate.getWeight();

            weightAccumulator = arith::AddFOp::create(
                rewriter, sampleOp.getLoc(), weightAccumulator, subWeight);

            int64_t mergeOffset = computeOffsetForNestedSample(
                sampleOp, fn, selection, sampleOp.getSymbolAttr(), symbolTable);

            auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
            auto row0 = arith::ConstantOp::create(
                rewriter, sampleOp.getLoc(), i64S,
                DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(0)));
            auto colOff = arith::ConstantOp::create(
                rewriter, sampleOp.getLoc(), i64S,
                DenseElementsAttr::get(
                    i64S, rewriter.getI64IntegerAttr(mergeOffset)));
            currTrace = impulse::DynamicUpdateSliceOp::create(
                            rewriter, sampleOp.getLoc(), traceType, currTrace,
                            subTrace, ValueRange{row0, colOff})
                            .getResult();
            currentTraceOffset =
                std::max(currentTraceOffset, mergeOffset + subPositionSize);

            for (auto output : nestedGenerate.getOutputs())
              sampledValues.push_back(output);
          }
        }

        sampleOp.replaceAllUsesWith(sampledValues);
        toErase.push_back(sampleOp);
        return WalkResult::advance();
      };

      auto result =
          processImpulseBlock(NewF.getFunctionBody().front(), rewriter, carry,
                              traceType, processSample, toErase);
      currTrace = carry.currTrace;
      weightAccumulator = carry.weight;
      currentTraceOffset = carry.offset;

      for (Operation *op : toErase)
        rewriter.eraseOp(op);

      if (failed(result)) {
        CI.emitError("Impulse: failed to walk sample ops");
        return failure();
      }

      // Rewrite the return to return (trace, weight, <original returns>...)
      NewF.walk([&](func::ReturnOp retOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(retOp);

        SmallVector<Value> newRetVals;
        Value outTrace = currTrace;
        if (scoreTailSize > 0) {
          auto fullType =
              RankedTensorType::get({1, declaredWidth}, rewriter.getF64Type());
          auto zeroFull =
              arith::ConstantOp::create(rewriter, retOp.getLoc(), fullType,
                                        DenseElementsAttr::get(fullType, 0.0));
          auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
          auto c0 = arith::ConstantOp::create(
              rewriter, retOp.getLoc(), i64S,
              DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(0)));
          Value acc = impulse::DynamicUpdateSliceOp::create(
                          rewriter, retOp.getLoc(), fullType, zeroFull,
                          currTrace, ValueRange{c0, c0})
                          .getResult();
          for (auto [i, s] : llvm::enumerate(siteScores)) {
            auto s11 = impulse::ReshapeOp::create(
                rewriter, retOp.getLoc(),
                RankedTensorType::get({1, 1}, rewriter.getF64Type()), s);
            auto cOff = arith::ConstantOp::create(
                rewriter, retOp.getLoc(), i64S,
                DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(
                                                 positionSize + (int64_t)i)));
            acc = impulse::DynamicUpdateSliceOp::create(
                      rewriter, retOp.getLoc(), fullType, acc, s11,
                      ValueRange{c0, cOff})
                      .getResult();
          }
          outTrace = acc;
        }
        newRetVals.push_back(outTrace);
        newRetVals.push_back(weightAccumulator);
        newRetVals.append(retOp.getOperands().begin(),
                          retOp.getOperands().end());

        func::ReturnOp::create(rewriter, retOp.getLoc(), newRetVals);
        rewriter.eraseOp(retOp);
      });

      rewriter.setInsertionPoint(CI);
      SmallVector<Value> operands;
      operands.push_back(CI.getConstraint());
      operands.append(CI.getInputs().begin(), CI.getInputs().end());
      auto newCI = func::CallOp::create(rewriter, CI.getLoc(), NewF.getName(),
                                        NewF.getResultTypes(), operands);

      rewriter.replaceOp(CI, newCI.getResults());

      delete putils;

      return success();
    }
  };

  // Delta weight: sum over dirty terms (regenerated or dependent on one) of
  // logp_new - logp_old; clean terms cancel in the MH ratio.

  static bool addressInList(impulse::SampleOp sampleOp, ArrayAttr addresses) {
    for (auto addr : addresses) {
      auto address = cast<ArrayAttr>(addr);
      if (!address.empty() && address[0] == sampleOp.getSymbolAttr())
        return true;
    }
    return false;
  }

  // Sample ops block taint: an unregenerated site's value comes from the trace.
  // An op with regions is tainted if anything nested uses a tainted value.
  static DenseSet<Value> computeRegenerateTaint(FunctionOpInterface fn,
                                                ArrayAttr regenerateAddresses) {
    DenseSet<Value> tainted;
    auto isTainted = [&](Value v) { return tainted.contains(v); };
    auto usesTainted = [&](Operation *op) {
      if (llvm::any_of(op->getOperands(), isTainted))
        return true;
      if (op->getNumRegions() == 0)
        return false;
      return op
          ->walk([&](Operation *inner) {
            return llvm::any_of(inner->getOperands(), isTainted)
                       ? WalkResult::interrupt()
                       : WalkResult::advance();
          })
          .wasInterrupted();
    };
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
        if (!usesTainted(op))
          return;
        taint(op->getResults());
        for (Region &region : op->getRegions())
          for (Block &block : region)
            taint(block.getArguments());
      });
    }
    return tainted;
  }

  // Each site must execute exactly once: one score-tail column per site.
  static bool deltaWeightFeasible(FunctionOpInterface fn) {
    bool feasible = true;
    fn.getFunctionBody().walk([&](Operation *op) {
      if (isa<impulse::IfOp>(op))
        feasible = false;
      if (auto s = dyn_cast<impulse::SampleOp>(op))
        if ((!s.getDistAttr() && !s.getLogpdfAttr()) ||
            s->getParentOp() != fn.getOperation())
          feasible = false;
    });
    return feasible;
  }

  struct LowerRegeneratePattern
      : public mlir::OpRewritePattern<impulse::RegenerateOp> {
    using mlir::OpRewritePattern<impulse::RegenerateOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(impulse::RegenerateOp CI,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        CI.emitError(
            "Impulse: calling `regenerate` on an empty function; if this "
            "is a distribution function, its sample op should have a "
            "logpdf attribute to avoid recursive `regenerate` calls which is "
            "intended for generative functions");
        return failure();
      }

      ArrayAttr selection = CI.getSelectionAttr();
      int64_t positionSize =
          computePositionSizeForSelection(CI, fn, selection, symbolTable);
      if (positionSize <= 0) {
        CI.emitError("Impulse: failed to compute position size for regenerate");
        return failure();
      }

      ArrayAttr constrainedAddresses = CI.getConstrainedAddressesAttr();
      bool hasConstraint = static_cast<bool>(CI.getConstraint());
      // Operand width, not the packed size, as in generate.
      int64_t constraintSize =
          hasConstraint ? cast<RankedTensorType>(CI.getConstraint().getType())
                              .getShape()[1]
                        : -1;

      int64_t declaredWidth =
          cast<RankedTensorType>(CI.getNewTrace().getType()).getShape()[1];
      int64_t scoreTailSize = declaredWidth - positionSize;
      SmallVector<Value> siteScores;
      int64_t siteOrdinal = 0;
      auto putils = ImpulseUtils::CreateFromClone(
          fn, ImpulseMode::Regenerate, declaredWidth, constraintSize);
      FunctionOpInterface NewF = putils->newFunc;

      OpBuilder entryBuilder(putils->initializationBlock,
                             putils->initializationBlock->begin());
      Location initLoc = putils->initializationBlock->begin()->getLoc();

      auto scalarType = RankedTensorType::get({}, entryBuilder.getF64Type());
      auto zeroWeight =
          arith::ConstantOp::create(entryBuilder, initLoc, scalarType,
                                    DenseElementsAttr::get(scalarType, 0.0));
      Value weightAccumulator = zeroWeight;

      auto traceType =
          RankedTensorType::get({1, positionSize}, entryBuilder.getF64Type());
      auto zeroTrace =
          arith::ConstantOp::create(entryBuilder, initLoc, traceType,
                                    DenseElementsAttr::get(traceType, 0.0));
      Value currTrace = zeroTrace;

      Value prevTrace = NewF.getArgument(0);
      Value constraint = hasConstraint ? NewF.getArgument(1) : Value();
      int64_t currentTraceOffset = 0;

      // Taint is computed up front; fresh draws are added as they are created.
      bool deltaMode = scoreTailSize > 0;
      DenseSet<Value> tainted;
      if (deltaMode) {
        if (!deltaWeightFeasible(NewF)) {
          CI.emitError("Impulse: regenerate over a score-tail trace requires "
                       "a straight-line body of top-level distribution sites");
          return failure();
        }
        tainted = computeRegenerateTaint(NewF, CI.getRegenerateAddressesAttr());
        // Non-regenerated values are unchanged: start from the previous trace.
        {
          auto valueSliceType = RankedTensorType::get(
              {1, positionSize}, entryBuilder.getF64Type());
          currTrace = impulse::SliceOp::create(
                          entryBuilder, initLoc, valueSliceType, prevTrace,
                          entryBuilder.getDenseI64ArrayAttr({0, 0}),
                          entryBuilder.getDenseI64ArrayAttr({1, positionSize}),
                          entryBuilder.getDenseI64ArrayAttr({1, 1}))
                          .getResult();
        }
      }

      ArrayAttr regenAddrs = CI.getRegenerateAddressesAttr();
      bool hastings = CI->hasAttr(kMHHastingsAttr);
      ArrayAttr siteScoresAttr =
          CI->getAttrOfType<ArrayAttr>(kMHSiteScoresAttr);
      bool scoring = static_cast<bool>(siteScoresAttr);
      // Site symbol -> column of the scores vector or of `oldInfo`.
      DenseMap<Attribute, int64_t> siteIndex;
      Value oldInfo;
      DenseSet<Value> hTaint;
      auto enclosingIfs = [&](Operation *op) {
        SmallVector<impulse::IfOp> ifs;
        for (Operation *p = op->getParentOp(); p && p != NewF.getOperation();
             p = p->getParentOp())
          if (auto ifOp = dyn_cast<impulse::IfOp>(p))
            ifs.push_back(ifOp);
        return ifs;
      };
      auto inFlippableBranch = [&](Operation *op) {
        return llvm::any_of(enclosingIfs(op), [&](impulse::IfOp ifOp) {
          return hTaint.contains(ifOp.getPredicate());
        });
      };
      if (scoring) {
        for (auto [k, addr] : llvm::enumerate(siteScoresAttr))
          siteIndex[cast<ArrayAttr>(addr)[0]] = k;
        auto kType = RankedTensorType::get({2, (int64_t)siteScoresAttr.size()},
                                           entryBuilder.getF64Type());
        weightAccumulator = arith::ConstantOp::create(
            entryBuilder, initLoc, kType, DenseElementsAttr::get(kType, 0.0));
      } else if (hastings && !deltaMode) {
        // Sites whose old density cannot be evaluated inline at the new state.
        hTaint = computeRegenerateTaint(NewF, regenAddrs);
        SmallVector<Attribute> kAddrs;
        NewF.getFunctionBody().walk([&](impulse::SampleOp s) {
          if (!s.getDistAttr() && !s.getLogpdfAttr())
            return;
          bool needsOld;
          if (addressInList(s, regenAddrs))
            needsOld =
                !enclosingIfs(s).empty() ||
                llvm::any_of(llvm::drop_begin(s.getInputs()),
                             [&](Value v) { return hTaint.contains(v); });
          else
            needsOld = !(constraint && constrainedAddresses &&
                         addressInList(s, constrainedAddresses)) &&
                       inFlippableBranch(s);
          if (!needsOld)
            return;
          siteIndex[s.getSymbolAttr()] = kAddrs.size();
          kAddrs.push_back(rewriter.getArrayAttr({s.getSymbolAttr()}));
        });
        if (!kAddrs.empty()) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&NewF.getFunctionBody().front());
          auto kType = RankedTensorType::get({2, (int64_t)kAddrs.size()},
                                             rewriter.getF64Type());
          SmallVector<Type> scoreTypes{prevTrace.getType(), kType};
          scoreTypes.append(fn.getResultTypes().begin(),
                            fn.getResultTypes().end());
          SmallVector<Value> scoreInputs(
              NewF.getArguments().drop_front(hasConstraint ? 2 : 1));
          auto scoreOp = impulse::RegenerateOp::create(
              rewriter, CI.getLoc(), scoreTypes, CI.getFnAttr(), scoreInputs,
              prevTrace, constraint, selection, rewriter.getArrayAttr({}),
              hasConstraint ? constrainedAddresses : ArrayAttr(),
              rewriter.getStringAttr(""));
          scoreOp->setAttr(kMHSiteScoresAttr, rewriter.getArrayAttr(kAddrs));
          oldInfo = scoreOp.getWeight();
        }
      }
      // In MH, constraint entries outside branches may be latent init values.
      auto readsConstraint = [&](impulse::SampleOp s) {
        if (!constraint || !constrainedAddresses)
          return false;
        return !(hastings || scoring) || !enclosingIfs(s).empty();
      };
      // `oldInfo` column of a flippable latent (kept if live, else redrawn).
      auto flipLatentIndex = [&](impulse::SampleOp s, bool isSelected) {
        if (!hastings || isSelected)
          return (int64_t)-1;
        auto it = siteIndex.find(s.getSymbolAttr());
        return it == siteIndex.end() ? (int64_t)-1 : it->second;
      };
      auto oldLiveAt = [&](Location loc, int64_t k) -> Value {
        Value flag = scalarAt(rewriter, loc, oldInfo, {0, k});
        Value half = arith::ConstantOp::create(
            rewriter, loc, scalarType, DenseElementsAttr::get(scalarType, 0.5));
        return arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGT,
                                     flag, half);
      };
      auto recordSiteScore = [&](Location loc, Value acc, int64_t k,
                                 Value score) -> Value {
        auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
        auto t11 = RankedTensorType::get({1, 1}, rewriter.getF64Type());
        auto idx = [&](int64_t v) -> Value {
          return arith::ConstantOp::create(
              rewriter, loc, i64S,
              DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(v)));
        };
        Value one = arith::ConstantOp::create(rewriter, loc, t11,
                                              DenseElementsAttr::get(t11, 1.0));
        Value scoreRow = impulse::ReshapeOp::create(rewriter, loc, t11, score);
        acc = impulse::DynamicUpdateSliceOp::create(
            rewriter, loc, acc.getType(), acc, one, ValueRange{idx(0), idx(k)});
        return impulse::DynamicUpdateSliceOp::create(
            rewriter, loc, acc.getType(), acc, scoreRow,
            ValueRange{idx(1), idx(k)});
      };
      auto sliceOldValues = [&](impulse::SampleOp sampleOp, int64_t offset) {
        SmallVector<Value> vals(sampleOp.getNumResults());
        for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
          auto resultType =
              cast<RankedTensorType>(sampleOp.getResult(i).getType());
          int64_t n = computeTensorElementCount(resultType);
          auto sliced = impulse::SliceOp::create(
              rewriter, sampleOp.getLoc(),
              RankedTensorType::get({1, n}, resultType.getElementType()),
              prevTrace, rewriter.getDenseI64ArrayAttr({0, offset}),
              rewriter.getDenseI64ArrayAttr({1, offset + n}),
              rewriter.getDenseI64ArrayAttr({1, 1}));
          vals[i] = impulse::ReshapeOp::create(rewriter, sampleOp.getLoc(),
                                               resultType, sliced);
          offset += n;
        }
        return vals;
      };

      SmallVector<Operation *> toErase;
      BranchCarry carry{currTrace, weightAccumulator, currentTraceOffset,
                        hastings ? Value(zeroWeight) : Value()};
      auto processSample = [&](impulse::SampleOp sampleOp,
                               BranchCarry &carry) -> WalkResult {
        Value &currTrace = carry.currTrace;
        Value &weightAccumulator = carry.weight;
        int64_t &currentTraceOffset = carry.offset;
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sampleOp);

        SmallVector<Value> sampledValues;

        if (auto dist = sampleOp.getDistAttr()) {
          Location loc = sampleOp.getLoc();
          Value rng = sampleOp.getInputs().front();
          SmallVector<Value> distArgs(std::next(sampleOp.getInputs().begin()),
                                      sampleOp.getInputs().end());
          auto distSampleType =
              cast<RankedTensorType>(sampleOp.getResult(1).getType());

          bool isSelected = false;
          for (auto addr : CI.getRegenerateAddressesAttr()) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              if (address.size() != 1) {
                sampleOp.emitError("Impulse: distribution function cannot have "
                                   "composite selected address");
                return WalkResult::interrupt();
              }
              isSelected = true;
              break;
            }
          }
          int64_t sampleOffset = computeOffsetForSampleInSelection(
              CI, fn, selection, sampleOp.getSymbolAttr(), symbolTable);

          sampledValues.resize(sampleOp.getNumResults());
          auto drawFresh = [&]() {
            SmallVector<Value> drawn(sampleOp.getNumResults());
            if (!impulse::distSamplerEmittable(dist)) {
              auto distFn =
                  cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                      sampleOp, sampleOp.getFnAttr()));
              auto distCall = func::CallOp::create(
                  rewriter, loc, distFn.getName(), distFn.getResultTypes(),
                  sampleOp.getInputs());
              for (unsigned i = 0; i < distCall.getNumResults(); ++i)
                drawn[i] = distCall.getResult(i);
            } else {
              auto sv = impulse::emitDistSampler(rewriter, loc, dist, rng,
                                                 distArgs, distSampleType);
              for (unsigned i = 0; i < sv.size(); ++i)
                drawn[i] = sv[i];
            }
            return drawn;
          };
          int64_t flipIdx = flipLatentIndex(sampleOp, isSelected);
          Value oldLive;
          if (isSelected) {
            sampledValues = drawFresh();
          } else {
            bool isConstrained = false;
            int64_t constrainedOffset = -1;
            if (readsConstraint(sampleOp)) {
              for (auto addr : constrainedAddresses) {
                auto address = cast<ArrayAttr>(addr);
                if (!address.empty() &&
                    address[0] == sampleOp.getSymbolAttr()) {
                  if (address.size() != 1) {
                    sampleOp.emitError("Impulse: distribution function cannot "
                                       "have composite constrained address");
                    return WalkResult::interrupt();
                  }
                  isConstrained = true;
                  constrainedOffset = computeOffsetForSampleInSelection(
                      CI, fn, constrainedAddresses, sampleOp.getSymbolAttr(),
                      symbolTable);
                  break;
                }
              }
            }
            Value source = isConstrained ? constraint : prevTrace;
            int64_t extractOffset =
                isConstrained ? constrainedOffset : sampleOffset;
            sampledValues[0] = rng;
            for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
              auto resultType =
                  cast<RankedTensorType>(sampleOp.getResult(i).getType());
              int64_t numElements = computeTensorElementCount(resultType);
              if (numElements < 0) {
                sampleOp.emitError(
                    "Impulse: dynamic tensor dimensions not supported");
                return WalkResult::interrupt();
              }
              auto sliceType = RankedTensorType::get(
                  {1, numElements}, resultType.getElementType());
              auto sliced = impulse::SliceOp::create(
                  rewriter, loc, sliceType, source,
                  rewriter.getDenseI64ArrayAttr({0, extractOffset}),
                  rewriter.getDenseI64ArrayAttr(
                      {1, extractOffset + numElements}),
                  rewriter.getDenseI64ArrayAttr({1, 1}));
              sampledValues[i] =
                  impulse::ReshapeOp::create(rewriter, loc, resultType, sliced)
                      .getResult();
              extractOffset += numElements;
            }
            if (flipIdx >= 0) {
              oldLive = oldLiveAt(loc, flipIdx);
              SmallVector<Value> drawn = drawFresh();
              for (unsigned i = 1; i < sampledValues.size(); ++i)
                sampledValues[i] = impulse::SelectOp::create(
                    rewriter, loc, sampledValues[i].getType(), oldLive,
                    sampledValues[i], drawn[i]);
              sampledValues[0] = drawn[0];
            }
          }
          if (hastings && !deltaMode && (isSelected || flipIdx >= 0))
            for (unsigned i = 1; i < sampledValues.size(); ++i)
              hTaint.insert(sampledValues[i]);

          bool termDirty = isSelected;
          if (deltaMode && !termDirty)
            for (Value arg : distArgs)
              if (tainted.contains(arg)) {
                termDirty = true;
                break;
              }

          if (deltaMode && isSelected)
            for (unsigned i = 1; i < sampledValues.size(); ++i)
              tainted.insert(sampledValues[i]);

          Value oldScore;
          if (scoreTailSize > 0) {
            auto s11 = RankedTensorType::get({1, 1}, rewriter.getF64Type());
            auto sliced = impulse::SliceOp::create(
                rewriter, loc, s11, prevTrace,
                rewriter.getDenseI64ArrayAttr({0, positionSize + siteOrdinal}),
                rewriter.getDenseI64ArrayAttr(
                    {1, positionSize + siteOrdinal + 1}),
                rewriter.getDenseI64ArrayAttr({1, 1}));
            oldScore =
                impulse::ReshapeOp::create(
                    rewriter, loc,
                    RankedTensorType::get({}, rewriter.getF64Type()), sliced)
                    .getResult();
          }

          if (scoring) {
            auto it = siteIndex.find(sampleOp.getSymbolAttr());
            if (it != siteIndex.end()) {
              Value logpdf = impulse::emitDistLogpdf(
                  rewriter, loc, dist, sampledValues[1], distArgs);
              weightAccumulator =
                  recordSiteScore(loc, weightAccumulator, it->second, logpdf);
            }
          } else if (!deltaMode || termDirty) {
            Value logpdf = impulse::emitDistLogpdf(rewriter, loc, dist,
                                                   sampledValues[1], distArgs);
            Value contribution =
                deltaMode
                    ? arith::SubFOp::create(rewriter, loc, logpdf, oldScore)
                          .getResult()
                    : logpdf;
            weightAccumulator = arith::AddFOp::create(
                rewriter, loc, weightAccumulator, contribution);
            if (scoreTailSize > 0)
              siteScores.push_back(logpdf);
            if (hastings && isSelected) {
              // A prior draw's proposal density cancels its own term.
              carry.corr = arith::AddFOp::create(rewriter, loc, carry.corr,
                                                 contribution);
              if (!deltaMode && !siteIndex.count(sampleOp.getSymbolAttr())) {
                // Parameters unchanged by the move: score the old value inline.
                SmallVector<Value> oldVals =
                    sliceOldValues(sampleOp, sampleOffset);
                Value oldLp = impulse::emitDistLogpdf(rewriter, loc, dist,
                                                      oldVals[1], distArgs);
                carry.corr =
                    arith::SubFOp::create(rewriter, loc, carry.corr, oldLp);
              }
            } else if (flipIdx >= 0) {
              // Kept: old density, cancelled via oldInfo; fresh: new density.
              Value term = arith::SelectOp::create(
                  rewriter, loc, oldLive,
                  scalarAt(rewriter, loc, oldInfo, {1, flipIdx}), logpdf);
              carry.corr =
                  arith::AddFOp::create(rewriter, loc, carry.corr, term);
            }
          } else {
            siteScores.push_back(oldScore);
          }
          ++siteOrdinal;

          bool inSelection = false;
          for (auto addr : selection) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              inSelection = true;
              break;
            }
          }
          if (inSelection) {
            for (unsigned i = 1; i < sampledValues.size(); ++i) {
              auto sampleValue = sampledValues[i];
              auto sampleTy = cast<RankedTensorType>(sampleValue.getType());
              int64_t numElements = computeTensorElementCount(sampleTy);
              if (numElements < 0) {
                sampleOp.emitError(
                    "Impulse: dynamic tensor dimensions not supported");
                return WalkResult::interrupt();
              }
              // The offset advances even when the value is not rewritten.
              if (!deltaMode || isSelected) {
                auto flatSampleType = RankedTensorType::get(
                    {1, numElements}, sampleTy.getElementType());
                auto flatSample = impulse::ReshapeOp::create(
                    rewriter, loc, flatSampleType, sampleValue);
                auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
                auto row0 = arith::ConstantOp::create(
                    rewriter, loc, i64S,
                    DenseElementsAttr::get(i64S,
                                           rewriter.getI64IntegerAttr(0)));
                auto colOff = arith::ConstantOp::create(
                    rewriter, loc, i64S,
                    DenseElementsAttr::get(
                        i64S, rewriter.getI64IntegerAttr(currentTraceOffset)));
                currTrace = impulse::DynamicUpdateSliceOp::create(
                                rewriter, loc, traceType, currTrace, flatSample,
                                ValueRange{row0, colOff})
                                .getResult();
              }
              currentTraceOffset += numElements;
            }
          }

          sampleOp.replaceAllUsesWith(sampledValues);
          toErase.push_back(sampleOp);
          return WalkResult::advance();
        }

        bool isDistribution = static_cast<bool>(sampleOp.getLogpdfAttr());

        if (isDistribution) {
          // A1. Distribution function: call the distribution function.
          bool isSelected = false;
          for (auto addr : CI.getRegenerateAddressesAttr()) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              if (address.size() != 1) {
                sampleOp.emitError(
                    "Impulse: distribution function cannot have composite "
                    "selected address");
                return WalkResult::interrupt();
              }
              isSelected = true;
              break;
            }
          }

          int64_t sampleOffset = computeOffsetForSampleInSelection(
              CI, fn, selection, sampleOp.getSymbolAttr(), symbolTable);

          if (isSelected) {
            // A2. Regenerate: call the distribution function.
            auto distFn =
                cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                    sampleOp, sampleOp.getFnAttr()));

            auto distCall = func::CallOp::create(
                rewriter, sampleOp.getLoc(), distFn.getName(),
                distFn.getResultTypes(), sampleOp.getInputs());

            sampledValues.append(distCall.getResults().begin(),
                                 distCall.getResults().end());
          } else {
            bool isConstrained = false;
            int64_t constrainedOffset = -1;
            if (readsConstraint(sampleOp)) {
              for (auto addr : constrainedAddresses) {
                auto address = cast<ArrayAttr>(addr);
                if (!address.empty() &&
                    address[0] == sampleOp.getSymbolAttr()) {
                  if (address.size() != 1) {
                    sampleOp.emitError(
                        "Impulse: distribution function cannot have composite "
                        "constrained address");
                    return WalkResult::interrupt();
                  }
                  isConstrained = true;
                  constrainedOffset = computeOffsetForSampleInSelection(
                      CI, fn, constrainedAddresses, sampleOp.getSymbolAttr(),
                      symbolTable);
                  break;
                }
              }
            }

            Value source = isConstrained ? constraint : prevTrace;
            int64_t extractOffset =
                isConstrained ? constrainedOffset : sampleOffset;
            sampledValues.resize(sampleOp.getNumResults());
            sampledValues[0] = sampleOp.getOperand(0); // RNG state

            for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
              auto resultType =
                  cast<RankedTensorType>(sampleOp.getResult(i).getType());
              int64_t numElements = computeTensorElementCount(resultType);
              if (numElements < 0) {
                sampleOp.emitError(
                    "Impulse: dynamic tensor dimensions not supported");
                return WalkResult::interrupt();
              }

              auto sliceType = RankedTensorType::get(
                  {1, numElements}, resultType.getElementType());
              auto sliced = impulse::SliceOp::create(
                  rewriter, sampleOp.getLoc(), sliceType, source,
                  rewriter.getDenseI64ArrayAttr({0, extractOffset}),
                  rewriter.getDenseI64ArrayAttr(
                      {1, extractOffset + numElements}),
                  rewriter.getDenseI64ArrayAttr({1, 1}));
              auto extracted = impulse::ReshapeOp::create(
                  rewriter, sampleOp.getLoc(), resultType, sliced);
              sampledValues[i] = extracted.getResult();
              extractOffset += numElements;
            }
          }

          int64_t flipIdx = flipLatentIndex(sampleOp, isSelected);
          Value oldLive;
          if (flipIdx >= 0) {
            oldLive = oldLiveAt(sampleOp.getLoc(), flipIdx);
            auto distFn =
                cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                    sampleOp, sampleOp.getFnAttr()));
            auto distCall = func::CallOp::create(
                rewriter, sampleOp.getLoc(), distFn.getName(),
                distFn.getResultTypes(), sampleOp.getInputs());
            for (unsigned i = 1; i < sampledValues.size(); ++i)
              sampledValues[i] = impulse::SelectOp::create(
                  rewriter, sampleOp.getLoc(), sampledValues[i].getType(),
                  oldLive, sampledValues[i], distCall.getResult(i));
            sampledValues[0] = distCall.getResult(0);
          }
          if (hastings && !deltaMode && (isSelected || flipIdx >= 0))
            for (unsigned i = 1; i < sampledValues.size(); ++i)
              hTaint.insert(sampledValues[i]);

          auto logpdfFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getLogpdfAttr()));

          SmallVector<Value> logpdfOperands;
          for (unsigned i = 1; i < sampledValues.size(); ++i) {
            logpdfOperands.push_back(sampledValues[i]);
          }
          for (unsigned i = 1; i < sampleOp.getNumOperands(); ++i) {
            logpdfOperands.push_back(sampleOp.getOperand(i));
          }

          if (logpdfOperands.size() != logpdfFn.getNumArguments()) {
            sampleOp.emitError("Impulse: failed to construct logpdf call; "
                               "logpdf function has wrong number of arguments");
            return WalkResult::interrupt();
          }

          bool termDirty = isSelected;
          if (deltaMode && !termDirty)
            for (unsigned i = 1; i < sampleOp.getNumOperands(); ++i)
              if (tainted.contains(sampleOp.getOperand(i))) {
                termDirty = true;
                break;
              }

          if (deltaMode && isSelected)
            for (unsigned i = 1; i < sampledValues.size(); ++i)
              tainted.insert(sampledValues[i]);

          Value oldScore;
          if (scoreTailSize > 0) {
            auto s11 = RankedTensorType::get({1, 1}, rewriter.getF64Type());
            auto sliced = impulse::SliceOp::create(
                rewriter, sampleOp.getLoc(), s11, prevTrace,
                rewriter.getDenseI64ArrayAttr({0, positionSize + siteOrdinal}),
                rewriter.getDenseI64ArrayAttr(
                    {1, positionSize + siteOrdinal + 1}),
                rewriter.getDenseI64ArrayAttr({1, 1}));
            oldScore =
                impulse::ReshapeOp::create(
                    rewriter, sampleOp.getLoc(),
                    RankedTensorType::get({}, rewriter.getF64Type()), sliced)
                    .getResult();
          }

          if (scoring) {
            auto it = siteIndex.find(sampleOp.getSymbolAttr());
            if (it != siteIndex.end()) {
              auto logpdf = func::CallOp::create(
                  rewriter, sampleOp.getLoc(), logpdfFn.getName(),
                  logpdfFn.getResultTypes(), logpdfOperands);
              weightAccumulator =
                  recordSiteScore(sampleOp.getLoc(), weightAccumulator,
                                  it->second, logpdf.getResult(0));
            }
          } else if (!deltaMode || termDirty) {
            auto logpdf = func::CallOp::create(
                rewriter, sampleOp.getLoc(), logpdfFn.getName(),
                logpdfFn.getResultTypes(), logpdfOperands);
            Value contribution = logpdf.getResult(0);
            if (deltaMode) {
              contribution = arith::SubFOp::create(rewriter, sampleOp.getLoc(),
                                                   contribution, oldScore);
            }
            weightAccumulator = arith::AddFOp::create(
                rewriter, sampleOp.getLoc(), weightAccumulator, contribution);
            if (scoreTailSize > 0)
              siteScores.push_back(logpdf.getResult(0));
            if (hastings && isSelected) {
              // A prior draw's proposal density cancels its own term.
              carry.corr = arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                                 carry.corr, contribution);
              if (!deltaMode && !siteIndex.count(sampleOp.getSymbolAttr())) {
                // Parameters unchanged by the move: score the old value inline.
                SmallVector<Value> oldOperands =
                    sliceOldValues(sampleOp, sampleOffset);
                oldOperands.erase(oldOperands.begin());
                for (unsigned i = 1; i < sampleOp.getNumOperands(); ++i)
                  oldOperands.push_back(sampleOp.getOperand(i));
                auto oldLp = func::CallOp::create(
                    rewriter, sampleOp.getLoc(), logpdfFn.getName(),
                    logpdfFn.getResultTypes(), oldOperands);
                carry.corr =
                    arith::SubFOp::create(rewriter, sampleOp.getLoc(),
                                          carry.corr, oldLp.getResult(0));
              }
            } else if (flipIdx >= 0) {
              // Kept: old density, cancelled via oldInfo; fresh: new density.
              Value term = arith::SelectOp::create(
                  rewriter, sampleOp.getLoc(), oldLive,
                  scalarAt(rewriter, sampleOp.getLoc(), oldInfo, {1, flipIdx}),
                  contribution);
              carry.corr = arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                                 carry.corr, term);
            }
          } else {
            siteScores.push_back(oldScore);
          }
          ++siteOrdinal;

          bool inSelection = false;
          for (auto addr : selection) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              inSelection = true;
              break;
            }
          }

          if (inSelection) {
            for (unsigned i = 1; i < sampledValues.size(); ++i) {
              auto sampleValue = sampledValues[i];
              auto sampleType = cast<RankedTensorType>(sampleValue.getType());
              int64_t numElements = computeTensorElementCount(sampleType);
              if (numElements < 0) {
                sampleOp.emitError(
                    "Impulse: dynamic tensor dimensions not supported");
                return WalkResult::interrupt();
              }
              if (!deltaMode || isSelected) {
                auto flatSampleType = RankedTensorType::get(
                    {1, numElements}, sampleType.getElementType());
                auto flatSample = impulse::ReshapeOp::create(
                    rewriter, sampleOp.getLoc(), flatSampleType, sampleValue);
                auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
                auto row0 = arith::ConstantOp::create(
                    rewriter, sampleOp.getLoc(), i64S,
                    DenseElementsAttr::get(i64S,
                                           rewriter.getI64IntegerAttr(0)));
                auto colOff = arith::ConstantOp::create(
                    rewriter, sampleOp.getLoc(), i64S,
                    DenseElementsAttr::get(
                        i64S, rewriter.getI64IntegerAttr(currentTraceOffset)));
                currTrace = impulse::DynamicUpdateSliceOp::create(
                                rewriter, sampleOp.getLoc(), traceType,
                                currTrace, flatSample, ValueRange{row0, colOff})
                                .getResult();
              }
              currentTraceOffset += numElements;
            }
          }
        } else {
          // B. Generative function: recursively regenerate the nested function
          auto genFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getFnAttr()));

          if (genFn.getFunctionBody().empty()) {
            sampleOp.emitError(
                "Impulse: generative function body is empty; "
                "if this is a distribution, add a logpdf attribute");
            return WalkResult::interrupt();
          }

          ArrayAttr subSelection =
              buildSubSelection(rewriter, selection, sampleOp.getSymbolAttr());
          ArrayAttr subRegenerateAddrs =
              buildSubSelection(rewriter, CI.getRegenerateAddressesAttr(),
                                sampleOp.getSymbolAttr());
          ArrayAttr subConstrainedAddrs =
              (constraint && constrainedAddresses)
                  ? buildSubSelection(rewriter, constrainedAddresses,
                                      sampleOp.getSymbolAttr())
                  : ArrayAttr();

          if (subSelection.empty()) {
            auto genCall = func::CallOp::create(
                rewriter, sampleOp.getLoc(), genFn.getName(),
                genFn.getResultTypes(), sampleOp.getInputs());
            sampledValues.append(genCall.getResults().begin(),
                                 genCall.getResults().end());
          } else {
            int64_t subPositionSize = computePositionSizeForSelection(
                sampleOp, genFn, subSelection, symbolTable);
            if (subPositionSize <= 0) {
              sampleOp.emitError(
                  "Impulse: failed to compute sub-position size");
              return WalkResult::interrupt();
            }

            int64_t mergeOffset = computeOffsetForNestedSample(
                sampleOp, fn, selection, sampleOp.getSymbolAttr(), symbolTable);
            if (mergeOffset < 0) {
              sampleOp.emitError("Impulse: failed to compute merge offset");
              return WalkResult::interrupt();
            }

            auto subTraceType = RankedTensorType::get({1, subPositionSize},
                                                      rewriter.getF64Type());
            Value subPrevTrace = impulse::SliceOp::create(
                rewriter, sampleOp.getLoc(), subTraceType, prevTrace,
                rewriter.getDenseI64ArrayAttr({0, mergeOffset}),
                rewriter.getDenseI64ArrayAttr(
                    {1, mergeOffset + subPositionSize}),
                rewriter.getDenseI64ArrayAttr({1, 1}));

            Value subConstraint;
            ArrayAttr subConstrainedForOp;
            if (constraint && subConstrainedAddrs &&
                !subConstrainedAddrs.empty()) {
              int64_t subConstraintSize = computePositionSizeForSelection(
                  sampleOp, genFn, subConstrainedAddrs, symbolTable);
              int64_t subConstraintOffset = computeOffsetForNestedSample(
                  sampleOp, fn, constrainedAddresses, sampleOp.getSymbolAttr(),
                  symbolTable);
              if (subConstraintSize > 0 && subConstraintOffset >= 0) {
                auto subConstraintType = RankedTensorType::get(
                    {1, subConstraintSize}, rewriter.getF64Type());
                subConstraint = impulse::SliceOp::create(
                    rewriter, sampleOp.getLoc(), subConstraintType, constraint,
                    rewriter.getDenseI64ArrayAttr({0, subConstraintOffset}),
                    rewriter.getDenseI64ArrayAttr(
                        {1, subConstraintOffset + subConstraintSize}),
                    rewriter.getDenseI64ArrayAttr({1, 1}));
                subConstrainedForOp = subConstrainedAddrs;
              }
            }

            if (hastings) {
              bool genHasIf = false;
              genFn.getFunctionBody().walk(
                  [&](impulse::IfOp) { genHasIf = true; });
              bool inputsTainted =
                  llvm::any_of(llvm::drop_begin(sampleOp.getInputs()),
                               [&](Value v) { return hTaint.contains(v); });
              if (inFlippableBranch(sampleOp) ||
                  (inputsTainted &&
                   (!subRegenerateAddrs.empty() || genHasIf))) {
                sampleOp.emitError(
                    "Impulse: MH over a generative call whose sites can "
                    "change liveness or parameters under the move is not "
                    "supported");
                return WalkResult::interrupt();
              }
            }

            // Build result types: (new_trace, weight, original_returns...)
            auto scalarTy = RankedTensorType::get({}, rewriter.getF64Type());
            SmallVector<Type> regenResultTypes;
            regenResultTypes.push_back(subTraceType);
            regenResultTypes.push_back(
                hastings ? RankedTensorType::get({2}, rewriter.getF64Type())
                         : scalarTy);
            for (auto t : genFn.getResultTypes())
              regenResultTypes.push_back(t);

            auto nestedRegenerate = impulse::RegenerateOp::create(
                rewriter, sampleOp.getLoc(), regenResultTypes,
                sampleOp.getFnAttr(), sampleOp.getInputs(), subPrevTrace,
                subConstraint, subSelection, subRegenerateAddrs,
                subConstrainedForOp, rewriter.getStringAttr(""));

            Value subTrace = nestedRegenerate.getNewTrace();
            Value subWeight = nestedRegenerate.getWeight();

            if (hastings) {
              nestedRegenerate->setAttr(kMHHastingsAttr,
                                        rewriter.getUnitAttr());
              auto [w, c] =
                  unpackHastingsWeight(rewriter, sampleOp.getLoc(), subWeight);
              weightAccumulator = arith::AddFOp::create(
                  rewriter, sampleOp.getLoc(), weightAccumulator, w);
              carry.corr = arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                                 carry.corr, c);
            } else if (!scoring) {
              weightAccumulator = arith::AddFOp::create(
                  rewriter, sampleOp.getLoc(), weightAccumulator, subWeight);
            }

            auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
            auto row0 = arith::ConstantOp::create(
                rewriter, sampleOp.getLoc(), i64S,
                DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(0)));
            auto colOff = arith::ConstantOp::create(
                rewriter, sampleOp.getLoc(), i64S,
                DenseElementsAttr::get(
                    i64S, rewriter.getI64IntegerAttr(mergeOffset)));
            currTrace = impulse::DynamicUpdateSliceOp::create(
                            rewriter, sampleOp.getLoc(), traceType, currTrace,
                            subTrace, ValueRange{row0, colOff})
                            .getResult();
            currentTraceOffset =
                std::max(currentTraceOffset, mergeOffset + subPositionSize);

            for (auto output : nestedRegenerate.getOutputs())
              sampledValues.push_back(output);
          }
        }

        sampleOp.replaceAllUsesWith(sampledValues);
        toErase.push_back(sampleOp);
        return WalkResult::advance();
      };

      auto result =
          processImpulseBlock(NewF.getFunctionBody().front(), rewriter, carry,
                              traceType, processSample, toErase);
      currTrace = carry.currTrace;
      weightAccumulator = carry.weight;
      currentTraceOffset = carry.offset;
      Value corrAccumulator = carry.corr;

      for (Operation *op : toErase)
        rewriter.eraseOp(op);

      if (failed(result)) {
        CI.emitError("Impulse: failed to walk sample ops");
        return failure();
      }

      NewF.walk([&](func::ReturnOp retOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(retOp);

        SmallVector<Value> newRetVals;
        Value outTrace = currTrace;
        if (scoreTailSize > 0) {
          auto fullType =
              RankedTensorType::get({1, declaredWidth}, rewriter.getF64Type());
          auto zeroFull =
              arith::ConstantOp::create(rewriter, retOp.getLoc(), fullType,
                                        DenseElementsAttr::get(fullType, 0.0));
          auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
          auto c0 = arith::ConstantOp::create(
              rewriter, retOp.getLoc(), i64S,
              DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(0)));
          Value acc = impulse::DynamicUpdateSliceOp::create(
                          rewriter, retOp.getLoc(), fullType, zeroFull,
                          currTrace, ValueRange{c0, c0})
                          .getResult();
          for (auto [i, s] : llvm::enumerate(siteScores)) {
            auto s11 = impulse::ReshapeOp::create(
                rewriter, retOp.getLoc(),
                RankedTensorType::get({1, 1}, rewriter.getF64Type()), s);
            auto cOff = arith::ConstantOp::create(
                rewriter, retOp.getLoc(), i64S,
                DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(
                                                 positionSize + (int64_t)i)));
            acc = impulse::DynamicUpdateSliceOp::create(
                      rewriter, retOp.getLoc(), fullType, acc, s11,
                      ValueRange{c0, cOff})
                      .getResult();
          }
          outTrace = acc;
        }
        newRetVals.push_back(outTrace);
        if (hastings) {
          Value corr = corrAccumulator;
          if (oldInfo)
            for (int64_t k = 0, e = cast<RankedTensorType>(oldInfo.getType())
                                        .getShape()[1];
                 k < e; ++k)
              corr = arith::SubFOp::create(
                  rewriter, retOp.getLoc(), corr,
                  scalarAt(rewriter, retOp.getLoc(), oldInfo, {1, k}));
          newRetVals.push_back(packHastingsWeight(rewriter, retOp.getLoc(),
                                                  weightAccumulator, corr));
        } else {
          newRetVals.push_back(weightAccumulator);
        }
        newRetVals.append(retOp.getOperands().begin(),
                          retOp.getOperands().end());

        func::ReturnOp::create(rewriter, retOp.getLoc(), newRetVals);
        rewriter.eraseOp(retOp);
      });

      if (hastings || scoring) {
        SmallVector<Type> resultTypes;
        NewF.walk([&](func::ReturnOp retOp) {
          resultTypes.assign(retOp.getOperandTypes().begin(),
                             retOp.getOperandTypes().end());
        });
        NewF.setType(FunctionType::get(CI.getContext(), NewF.getArgumentTypes(),
                                       resultTypes));
      }

      rewriter.setInsertionPoint(CI);
      SmallVector<Value> operands;
      operands.push_back(CI.getOriginalTrace());
      if (hasConstraint)
        operands.push_back(CI.getConstraint());
      operands.append(CI.getInputs().begin(), CI.getInputs().end());
      auto newCI = func::CallOp::create(rewriter, CI.getLoc(), NewF.getName(),
                                        NewF.getResultTypes(), operands);

      rewriter.replaceOp(CI, newCI.getResults());

      delete putils;

      return success();
    }
  };
};

} // end anonymous namespace

void ExpandImpulsePass::runOnOperation() {
  // Must run before the per-op MH patterns lower these moves.
  MHRunBatcher::run(getOperation(), &getContext());

  RewritePatternSet patterns(&getContext());
  patterns.add<LowerUntracedCallPattern, LowerSimulatePattern,
               LowerGeneratePattern, LowerMHPattern, LowerRegeneratePattern>(
      &getContext());
  patterns.add<LowerMCMCPattern<impulse::InferOp>,
               LowerMCMCPattern<impulse::InferRegionOp>>(&getContext(),
                                                         debugDump);
  patterns.add<LowerVIPattern<impulse::InferOp>,
               LowerVIPattern<impulse::InferRegionOp>>(&getContext(),
                                                       debugDump);
  patterns.add<LowerMHInferPattern<impulse::InferOp>,
               LowerMHInferPattern<impulse::InferRegionOp>>(&getContext());
  patterns.add<LowerGibbsRegionPattern, RejectFlatGibbsPattern>(&getContext());

  mlir::GreedyRewriteConfig config;

  if (failed(
          applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
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
