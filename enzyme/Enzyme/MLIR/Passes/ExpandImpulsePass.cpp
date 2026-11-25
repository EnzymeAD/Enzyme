//===- ExpandImpulsePass.cpp - Lower the Impulse inference ops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the expand-impulse pass, which lowers the Impulse
// inference ops (simulate, generate, regenerate, untraced_call, mh, and infer
// and infer_region with an HMC, NUTS, VI, MH or Gibbs configuration) into
// calls, arith and math ops and lower-level Impulse ops.
//===----------------------------------------------------------------------===//

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Impulse/InferStateLayout.h"
#include "Dialect/Ops.h"
#include "Interfaces/DistBuilder.h"
#include "Interfaces/HMCUtils.h"
#include "Interfaces/ImpulseUtils.h"
#include "Interfaces/VIUtils.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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

/// Sample sites by symbol.
using SampleOpMap = DenseMap<Attribute, impulse::SampleOp>;

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

/// Returns the site of `map` with symbol `targetSymbol`, or null.
static impulse::SampleOp findSampleBySymbol(const SampleOpMap &map,
                                            Attribute targetSymbol) {
  auto it = map.find(targetSymbol);
  return it != map.end() ? it->second : nullptr;
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

  auto sampleOp = findSampleBySymbol(sampleMap, address[0]);
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

/// Returns the first column of the first address starting with `targetSymbol`
/// on a trace that lays out the sites of `fn` at the addresses in `selection`
/// in order, or -1 when there is none or an address before it has no site.
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

namespace {
/// Where the sites of a model at a list of addresses sit on a trace that lays
/// out their values one after another, in list order.
struct FlatTraceLayout {
  /// The columns of the site at one address.
  struct Entry {
    ArrayAttr address;
    int64_t column;
    int64_t size;
  };
  /// The sample sites of the model, nested ones included, by symbol.
  SampleOpMap siteBySymbol;
  /// One entry per address, in list order.
  SmallVector<Entry> entries;
  /// Columns taken by all the addresses.
  int64_t positionSize = 0;
  /// First column of the first address that starts with each symbol.
  DenseMap<Attribute, int64_t> columnOf;
  /// Columns taken by the addresses that continue past each symbol.
  DenseMap<Attribute, int64_t> nestedSizeOf;

  /// Returns the first column of the first address that starts with
  /// `symbol`, or -1 when no address does.
  int64_t firstColumn(Attribute symbol) const {
    auto it = columnOf.find(symbol);
    return it == columnOf.end() ? -1 : it->second;
  }
};
} // namespace

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

/// Lays out the sites of `fn` at `addresses` on a trace, in that order. Fails
/// after emitting on `op` when an address is empty or `getAddressSize` fails.
static FailureOr<FlatTraceLayout>
getFlatTraceLayout(Operation *op, FunctionOpInterface fn, ArrayAttr addresses,
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

/// Fails after emitting on `op` when `support`, the support of the site at
/// `address`, lacks a bound or the event size that unconstraining reads.
static LogicalResult verifyUnconstrainable(Operation *op, Attribute address,
                                           impulse::SupportAttr support) {
  if (!support)
    return success();
  impulse::SupportKind kind = support.getKind();
  SmallVector<StringRef> missing;
  if ((kind == impulse::SupportKind::INTERVAL ||
       kind == impulse::SupportKind::GREATER_THAN) &&
      !support.getLowerBound())
    missing.push_back("`lower_bound`");
  if ((kind == impulse::SupportKind::INTERVAL ||
       kind == impulse::SupportKind::LESS_THAN) &&
      !support.getUpperBound())
    missing.push_back("`upper_bound`");
  if (kind == impulse::SupportKind::SIMPLEX && !support.getEventSize())
    missing.push_back("`event_size`");
  if (missing.empty())
    return success();
  return op->emitError("Impulse: the ")
         << impulse::stringifySupportKind(kind) << " support of the site at "
         << address << " has no " << llvm::join(missing, " and ")
         << ", which unconstraining the site needs";
}

/// Checks the supports of the sites of `fn` at the top-level addresses in
/// `selection`.
static LogicalResult verifySelectedSupports(Operation *op,
                                            FunctionOpInterface fn,
                                            ArrayAttr selection) {
  SampleOpMap sampleMap = buildSampleOpMap(fn);
  for (Attribute addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (address.size() != 1)
      continue;
    impulse::SampleOp site = findSampleBySymbol(sampleMap, address[0]);
    if (site && failed(verifyUnconstrainable(op, addr, site.getSupportAttr())))
      return failure();
  }
  return success();
}

/// Checks the supports of the sites of `regionOp` at the addresses in its
/// `selection`.
static LogicalResult verifySelectedSupports(impulse::InferRegionOp regionOp) {
  ArrayAttr selection = regionOp.getSelectionAttr();
  DenseSet<Attribute> selected(selection.begin(), selection.end());
  WalkResult result =
      regionOp.getSampling().walk([&](impulse::SampleRegionOp site) {
        Attribute address = site.getAddressAttr();
        if (address && selected.contains(address) &&
            failed(verifyUnconstrainable(regionOp, address,
                                         site.getSupportAttr())))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
  return failure(result.wasInterrupted());
}

/// The selected addresses in `all_addresses` order if
/// `positionDeclarationOrder`, else in `selection` order, each mapped through
/// `key`; an address that `key` maps to null is skipped.
static SmallVector<Attribute>
orderSelected(ArrayAttr selection, ArrayAttr allAddresses,
              bool positionDeclarationOrder,
              function_ref<Attribute(Attribute)> key) {
  SmallVector<Attribute> ordered;
  if (!positionDeclarationOrder) {
    for (Attribute address : selection)
      if (Attribute selected = key(address))
        ordered.push_back(selected);
    return ordered;
  }
  llvm::SmallDenseSet<Attribute, 4> selected;
  for (Attribute address : selection)
    if (Attribute keyed = key(address))
      selected.insert(keyed);
  for (Attribute address : allAddresses) {
    Attribute keyed = key(address);
    if (keyed && selected.contains(keyed))
      ordered.push_back(keyed);
  }
  return ordered;
}

/// Appends to `supports` a site of `size` constrained values on `support` at
/// trace column `traceOffset`, placed in the position after the sites already
/// in `supports`.
static void appendSupport(SmallVectorImpl<impulse::SupportInfo> &supports,
                          int64_t traceOffset, int64_t size,
                          impulse::SupportAttr support) {
  int64_t constrainedOffset = 0;
  int64_t unconstrainedOffset = 0;
  if (!supports.empty()) {
    const impulse::SupportInfo &last = supports.back();
    constrainedOffset = last.constrainedOffset + last.constrainedSize;
    unconstrainedOffset = last.unconstrainedOffset + last.unconstrainedSize;
  }
  int64_t unconstrainedSize =
      support ? support.getUnconstrainedSize(size) : size;
  supports.emplace_back(constrainedOffset, traceOffset, size,
                        unconstrainedOffset, unconstrainedSize, support);
}

/// Support info for the top-level addresses in `selection`. Positions follow
/// `all_addresses` order when `positionDeclarationOrder` is set, else
/// `selection` order.
static SmallVector<impulse::SupportInfo>
collectSupportInfoForSelection(Operation *op, FunctionOpInterface fn,
                               ArrayAttr selection, ArrayAttr allAddresses,
                               SymbolTableCollection &symbolTable,
                               bool positionDeclarationOrder) {
  auto sampleMap = buildSampleOpMap(fn);
  auto topLevelSymbol = [](Attribute address) -> Attribute {
    auto path = cast<ArrayAttr>(address);
    return path.size() == 1 ? path[0] : Attribute();
  };
  SmallVector<impulse::SupportInfo> supports;
  for (Attribute targetSymbol : orderSelected(
           selection, allAddresses, positionDeclarationOrder, topLevelSymbol)) {
    auto sampleOp = findSampleBySymbol(sampleMap, targetSymbol);
    if (!sampleOp)
      continue;

    int64_t sampleSize = computeSampleElementCount(op, sampleOp);
    if (sampleSize < 0)
      continue;

    int64_t traceOffset = computeOffsetForSampleInSelection(
        op, fn, allAddresses, targetSymbol, symbolTable);
    if (traceOffset < 0) {
      op->emitError("Symbol in selection not found in all_addresses - cannot "
                    "determine trace offset for scattered selection");
      return {};
    }

    appendSupport(supports, traceOffset, sampleSize, sampleOp.getSupportAttr());
  }
  return supports;
}

/// Returns the number of trace columns the values of `site` take.
static int64_t sampleRegionSize(impulse::SampleRegionOp site) {
  int64_t size = 0;
  for (Value result : llvm::drop_begin(site->getResults()))
    size += cast<RankedTensorType>(result.getType()).getNumElements();
  return size;
}

namespace {
/// Where the sites of a region-form infer op sit on its trace.
struct RegionTraceLayout {
  DenseMap<Attribute, impulse::SampleRegionOp> siteByAddress;
  /// First trace column of each address in `all_addresses`.
  DenseMap<Attribute, int64_t> columnOf;
  /// Columns taken by the values of all addresses, before any score tail.
  int64_t positionSize = 0;
};
} // namespace

/// Lays out the `all_addresses` of `regionOp` on its trace, in that order.
/// Sites are looked up in the whole sampling region, or with `topLevelOnly`
/// in its entry block only. Fails when an address has no site.
static FailureOr<RegionTraceLayout>
getRegionTraceLayout(impulse::InferRegionOp regionOp, bool topLevelOnly) {
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

namespace {
/// The position layout of the selected sites of a region-form infer op.
struct RegionPositionLayout {
  SmallVector<impulse::SupportInfo> supports;
  /// Position column of each value of the selected sites, in `selection`
  /// order.
  SmallVector<int64_t> argOffsets;
};
} // namespace

/// Lays out the selected sites of `regionOp` in `all_addresses` order if
/// `positionDeclarationOrder`, else in `selection` order; trace offsets follow
/// `all_addresses`. Fails when an address has no site in the sampler.
static FailureOr<RegionPositionLayout>
collectRegionLayout(impulse::InferRegionOp regionOp,
                    bool positionDeclarationOrder) {
  ArrayAttr selection = regionOp.getSelectionAttr();
  ArrayAttr allAddresses = regionOp.getAllAddressesAttr();
  if (!selection || !allAddresses || regionOp.getSampling().empty())
    return failure();
  // Branch-local sites also hold trace columns, so walk the whole sampler.
  FailureOr<RegionTraceLayout> trace =
      getRegionTraceLayout(regionOp, /*topLevelOnly=*/false);
  if (failed(trace))
    return failure();

  RegionPositionLayout layout;
  DenseMap<Attribute, int64_t> positionColumn;
  for (Attribute address :
       orderSelected(selection, allAddresses, positionDeclarationOrder,
                     [](Attribute address) { return address; })) {
    impulse::SampleRegionOp site = trace->siteByAddress.lookup(address);
    if (!site || !trace->columnOf.count(address))
      return failure();
    appendSupport(layout.supports, trace->columnOf.lookup(address),
                  sampleRegionSize(site), site.getSupportAttr());
    positionColumn[address] = layout.supports.back().constrainedOffset;
  }

  for (Attribute address : selection) {
    impulse::SampleRegionOp site = trace->siteByAddress.lookup(address);
    if (!site)
      return failure();
    int64_t column = positionColumn.lookup(address);
    for (Value result : llvm::drop_begin(site->getResults())) {
      layout.argOffsets.push_back(column);
      column += cast<RankedTensorType>(result.getType()).getNumElements();
    }
  }
  return layout;
}

namespace {
/// HMC/NUTS warmup adaptation state, in `state_schema` slot order from
/// `da_log_step_size` through `window_idx`. `welford` is unset while the warmup
/// does not adapt the mass matrix.
struct AdaptationState {
  DualAveragingState dualAveraging;
  std::optional<WelfordState> welford;
  Value windowIdx;

  SmallVector<Value> toValues() const {
    SmallVector<Value> values = dualAveraging.toValues();
    if (welford)
      llvm::append_range(values, welford->toValues());
    values.push_back(windowIdx);
    return values;
  }
  SmallVector<Type> getTypes() const {
    return llvm::to_vector(ValueRange(toValues()).getTypes());
  }
  /// Reads the state, with Welford accumulators if `hasWelford`, from the
  /// front of `values` and drops what it read.
  static AdaptationState consume(ArrayRef<Value> &values, bool hasWelford) {
    AdaptationState state;
    state.dualAveraging = DualAveragingState::fromValues(
        values.take_front(DualAveragingState::kNumValues));
    values = values.drop_front(DualAveragingState::kNumValues);
    if (hasWelford) {
      state.welford =
          WelfordState::fromValues(values.take_front(WelfordState::kNumValues));
      values = values.drop_front(WelfordState::kNumValues);
    }
    state.windowIdx = values.front();
    values = values.drop_front();
    return state;
  }
};

/// Values carried by the HMC/NUTS warmup loop.
struct WarmupCarry {
  ChainPoint chain;
  Value stepSize;
  Value invMass;
  Value massMatrixSqrt;
  AdaptationState adaptation;

  SmallVector<Value> toValues() const {
    SmallVector<Value> values = {
        chain.q,  chain.grad, chain.potentialEnergy, chain.rng,
        stepSize, invMass,    massMatrixSqrt};
    llvm::append_range(values, adaptation.toValues());
    return values;
  }
  /// The position, gradient, potential energy and step size are typed by
  /// `positionType` and `scalarType`.
  SmallVector<Type> getTypes(RankedTensorType positionType,
                             RankedTensorType scalarType) const {
    SmallVector<Type> types = {positionType,
                               positionType,
                               scalarType,
                               chain.rng.getType(),
                               scalarType,
                               invMass.getType(),
                               massMatrixSqrt.getType()};
    llvm::append_range(types, adaptation.getTypes());
    return types;
  }
  static WarmupCarry fromValues(ValueRange range, bool hasWelford) {
    SmallVector<Value> storage = llvm::to_vector(range);
    ArrayRef<Value> values = storage;
    WarmupCarry carry;
    carry.chain = {values[0], values[1], values[2], values[3]};
    carry.stepSize = values[4];
    carry.invMass = values[5];
    carry.massMatrixSqrt = values[6];
    values = values.drop_front(7);
    carry.adaptation = AdaptationState::consume(values, hasWelford);
    return carry;
  }
};

/// The results of one HMC/NUTS chain, in the order of the op's results.
struct ChainResult {
  Value trace;
  Value diagnostics;
  Value logDensities;
  Value rng;
  SmallVector<Value> stateOut;

  SmallVector<Value> toValues() const {
    SmallVector<Value> values = {trace, diagnostics, logDensities, rng};
    llvm::append_range(values, stateOut);
    return values;
  }
};

/// Emits the HMC/NUTS chains of one `impulse.infer` or `impulse.infer_region`.
/// The constructor reads what all chains share and creates the kernel's
/// constant operands; `emitChains` then emits the chains one after another.
class MCMCChainEmitter {
public:
  template <typename OpT>
  MCMCChainEmitter(OpBuilder &builder, OpT op, ModelContext model);

  /// Emits every chain. Several chains each start from their own key: the
  /// input key split once per chain, or on resume the keys saved in the `rng`
  /// slot.
  SmallVector<ChainResult> emitChains(Value rng);

private:
  /// Constants of the warmup adaptation schedule.
  struct WarmupSchedule {
    Value zero;
    Value one;
    SmallVector<Value> windowEnds;
    Value lastWindow;
    Value lastIteration;
  };

  /// Emits chain `chainIdx`, which draws from key `rng`.
  ChainResult emitChain(Value rng, int64_t chainIdx);

  /// The `state_in` value of slot `name`, or null on a cold start.
  Value stateInSlot(StringRef name) const;
  /// Like `stateInSlot`, but only chain `chainIdx` of a chain-major slot.
  Value chainStateInSlot(StringRef name, int64_t chainIdx);
  /// Takes element `chainIdx` of the leading dimension of `value`.
  Value sliceChain(Value value, int64_t chainIdx);

  /// The HMC context of `model` with the given mass matrix and step size.
  HMCContext makeHMCContext(Value invMass, Value massMatrixSqrt,
                            Value stepSize) const;
  /// The NUTS context of `model` with the given mass matrix and step size,
  /// for a transition from potential energy `potentialEnergy`.
  NUTSContext makeNUTSContext(Value invMass, Value massMatrixSqrt,
                              Value stepSize, Value potentialEnergy) const;
  /// One HMC or NUTS transition from `state`.
  MCMCKernelResult runKernel(const ChainPoint &state, Value stepSize,
                             Value invMass, Value massMatrixSqrt);

  /// The chain's resumed state, or the initial state computed by InitHMC.
  ChainPoint initChainState(Value rng, int64_t chainIdx, Value stepSize,
                            Value invMass, Value massMatrixSqrt);
  /// Returns `saved` when set, else a fresh adaptation state for `stepSize`;
  /// a fresh state gets Welford accumulators only if `withWelford`.
  AdaptationState
  loadOrInitAdaptation(Value stepSize, bool withWelford,
                       const std::optional<AdaptationState> &saved);
  /// Emits the warmup loop and returns its results.
  WarmupCarry emitWarmup(const ChainPoint &start, Value stepSize, Value invMass,
                         Value massMatrixSqrt,
                         const std::optional<AdaptationState> &saved,
                         int64_t chainIdx);
  /// Emits one warmup iteration at global iteration `iteration` and returns
  /// the values it carries to the next one.
  WarmupCarry emitWarmupStep(const WarmupCarry &carry, Value iteration,
                             const WarmupSchedule &schedule);
  /// Emits the end of a middle adaptation window: when `atMiddleWindowEnd`
  /// holds, the mass matrix is re-estimated from the Welford accumulators of
  /// `next` and the adapted accumulators restart. Updates `next` to the
  /// selected values.
  void emitWindowEndRestart(WarmupCarry &next, Value atMiddleWindowEnd);
  /// Emits the sampling loop from `state`, leaves the chain's final state in
  /// `state` and returns the collected samples and diagnostics.
  ChainResult emitSampling(ChainPoint &state, Value stepSize, Value invMass,
                           Value massMatrixSqrt);
  /// The chain's `state_out` values in the order of the canonical
  /// `state_schema`, which the op verifier enforces: the chain state, the
  /// step size and the inverse mass matrix, then with `adaptation` its values
  /// and the warmup offset.
  SmallVector<Value>
  packStateOut(const ChainPoint &state, Value stepSize, Value invMass,
               const std::optional<AdaptationState> &adaptation,
               int64_t chainIdx);

  OpBuilder &builder;
  Location loc;
  ModelContext model;

  int64_t numWarmup;
  int64_t totalWarmup;
  int64_t numSamples;
  int64_t thinning;
  int64_t numChains;

  ArrayAttr schema;
  ValueRange stateIn;
  Value initPosition;
  bool exposeAdaptation;
  std::optional<AdaptationState> savedAdaptation;
  /// Step size and inverse mass matrix every chain starts from, unless a
  /// chain-major `state_in` slot overrides them per chain.
  Value initStepSize;
  Value initInvMass;
  /// Whether the inverse mass matrix and its Welford estimate are diagonal.
  bool diagonal = true;
  WelfordConfig welfordConfig;

  bool isHMC;
  bool isNUTS;
  bool adaptStepSize = false;
  bool adaptMassMatrix = false;
  Value trajectoryLength;
  int64_t numSteps = 0;
  Value maxDeltaEnergy;
  int64_t maxTreeDepth = 0;

  Type elemType;
  RankedTensorType positionType;
  RankedTensorType scalarType;
  RankedTensorType i64TensorType;
};
} // namespace

/// Stacks the per-chain results into the op's chain-major results. The output
/// RNG state is the last chain's key; with a state schema, the keys of all
/// chains also go into the trailing `rng` slot.
static SmallVector<Value> stackChainResults(OpBuilder &builder, Location loc,
                                            ArrayRef<ChainResult> chains,
                                            bool hasStateSchema) {
  Value zeroIdx = emitI64ScalarConstant(builder, loc, 0);
  int64_t numChains = chains.size();

  auto stack = [&](function_ref<Value(const ChainResult &)> pick,
                   bool isRng) -> Value {
    auto type = cast<RankedTensorType>(pick(chains.front()).getType());
    SmallVector<int64_t> stackedShape = {numChains};
    llvm::append_range(stackedShape, type.getShape());
    auto stackedType =
        RankedTensorType::get(stackedShape, type.getElementType());
    SmallVector<int64_t> oneShape = {1};
    llvm::append_range(oneShape, type.getShape());
    auto oneType = RankedTensorType::get(oneShape, type.getElementType());
    Value buffer;
    // ui64 has no arith.constant, so the key buffer starts from chain 0's key.
    if (isRng)
      buffer = enzyme::BroadcastOp::create(
          builder, loc, stackedType, pick(chains.front()),
          builder.getDenseI64ArrayAttr({numChains}));
    else
      buffer = arith::ConstantOp::create(builder, loc, stackedType,
                                         builder.getZeroAttr(stackedType));
    for (auto [chainIdx, chain] : llvm::enumerate(chains)) {
      Value reshaped =
          impulse::ReshapeOp::create(builder, loc, oneType, pick(chain));
      SmallVector<Value> indices = {
          emitI64ScalarConstant(builder, loc, chainIdx)};
      indices.append(type.getRank(), zeroIdx);
      buffer = impulse::DynamicUpdateSliceOp::create(builder, loc, stackedType,
                                                     buffer, reshaped, indices);
    }
    return buffer;
  };

  SmallVector<Value> results;
  results.push_back(stack([](const ChainResult &chain) { return chain.trace; },
                          /*isRng=*/false));
  results.push_back(
      stack([](const ChainResult &chain) { return chain.diagnostics; },
            /*isRng=*/false));
  results.push_back(
      stack([](const ChainResult &chain) { return chain.logDensities; },
            /*isRng=*/false));
  results.push_back(chains.back().rng);
  for (size_t i = 0, e = chains.front().stateOut.size(); i < e; ++i)
    results.push_back(
        stack([i](const ChainResult &chain) { return chain.stateOut[i]; },
              /*isRng=*/false));
  if (hasStateSchema)
    results.push_back(stack([](const ChainResult &chain) { return chain.rng; },
                            /*isRng=*/true));
  return results;
}

/// Reads the operands that the HMC, NUTS and VI lowerings share; the caller
/// lays out the position.
template <typename OpT>
static ModelContext readInferModel(OpT op, bool debugDump) {
  ModelContext model;
  model.logpdfFn = op.getLogpdfFnAttr();
  if (!model.logpdfFn) {
    model.fn = op.getFnAttr();
    model.originalTrace = op.getOriginalTrace();
  }
  model.fnInputs = llvm::to_vector(op.getInputs().drop_front());
  model.allAddresses = op.getAllAddressesAttr();
  model.autodiffAttrs = op.getAutodiffAttrsAttr();
  model.debugDump = debugDump;
  if constexpr (std::is_same_v<OpT, impulse::InferRegionOp>)
    model.logpdfRegion = &op.getLogpdf();
  return model;
}

/// Lays out the top-level sites in the `selection` of `op` in
/// `model.supports`, in `all_addresses` order if `positionDeclarationOrder`,
/// else in `selection` order, and reads the result types of the model `fn` of
/// `op`. Returns the number of values the selected sites hold; fails when
/// there is none or a site cannot be laid out.
template <typename OpT>
static FailureOr<int64_t>
resolveFlatModel(OpT op, bool positionDeclarationOrder,
                 SymbolTableCollection &symbolTable, ModelContext &model) {
  auto fn = cast<FunctionOpInterface>(
      symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr()));
  ArrayAttr selection = op.getSelectionAttr();
  int64_t numValues =
      computePositionSizeForSelection(op, fn, selection, symbolTable);
  if (numValues <= 0)
    return failure();
  if (failed(verifySelectedSupports(op, fn, selection)))
    return failure();
  model.supports =
      collectSupportInfoForSelection(op, fn, selection, model.allAddresses,
                                     symbolTable, positionDeclarationOrder);
  auto fnType = cast<FunctionType>(fn.getFunctionType());
  model.fnResultTypes.assign(fnType.getResults().begin(),
                             fnType.getResults().end());
  return numValues;
}

template <typename OpT>
MCMCChainEmitter::MCMCChainEmitter(OpBuilder &builder, OpT op,
                                   ModelContext model)
    : builder(builder), loc(op.getLoc()), model(std::move(model)),
      numWarmup(op.getNumWarmup()), totalWarmup(op.getNumWarmup()),
      numSamples(op.getNumSamples()), thinning(op.getThinning()),
      numChains(op.getNumChains()), schema(op.getStateSchemaAttr()),
      stateIn(op.getStateIn()), initPosition(op.getInitPosition()),
      initStepSize(op.getInitStepSize()),
      initInvMass(op.getInitInverseMassMatrix()),
      isHMC(op.getHmcConfig().has_value()),
      isNUTS(op.getNutsConfig().has_value()) {
  if (auto total = op.getTotalWarmup())
    totalWarmup = *total;

  exposeAdaptation =
      impulse::slotIndex(schema, state_slot::kWindowIdx).has_value();
  if (std::optional<unsigned> first =
          impulse::slotIndex(schema, state_slot::kDaLogStepSize);
      first && *first < stateIn.size()) {
    SmallVector<Value> slots = llvm::to_vector(stateIn.drop_front(*first));
    ArrayRef<Value> values = slots;
    savedAdaptation = AdaptationState::consume(values, /*hasWelford=*/true);
  }

  // Multi-chain slots are chain-major; emitChain reads them per chain.
  if (numChains <= 1) {
    if (Value savedStepSize = stateInSlot(state_slot::kStepSize))
      initStepSize = savedStepSize;
    if (Value savedInvMass = stateInSlot(state_slot::kInverseMassMatrix))
      initInvMass = savedInvMass;
  }

  elemType = cast<RankedTensorType>(initStepSize.getType()).getElementType();
  this->model.elementType = elemType;
  positionType = RankedTensorType::get({1, this->model.positionSize}, elemType);
  scalarType = RankedTensorType::get({}, elemType);
  i64TensorType = RankedTensorType::get({}, builder.getI64Type());

  auto f64TensorType = RankedTensorType::get({}, builder.getF64Type());
  if (isHMC) {
    auto hmcConfig = op.getHmcConfig().value();
    numSteps = hmcConfig.getNumSteps();
    trajectoryLength =
        emitSplatConstant(builder, loc, f64TensorType,
                          hmcConfig.getTrajectoryLength().getValueAsDouble());
    adaptStepSize = hmcConfig.getAdaptStepSize();
    adaptMassMatrix = hmcConfig.getAdaptMassMatrix();
  } else {
    auto nutsConfig = op.getNutsConfig().value();
    maxTreeDepth = nutsConfig.getMaxTreeDepth();
    adaptStepSize = nutsConfig.getAdaptStepSize();
    adaptMassMatrix = nutsConfig.getAdaptMassMatrix();
    double maxDeltaEnergyVal =
        nutsConfig.getMaxDeltaEnergy()
            ? nutsConfig.getMaxDeltaEnergy().getValueAsDouble()
            : 1000.0;
    maxDeltaEnergy =
        emitSplatConstant(builder, loc, f64TensorType, maxDeltaEnergyVal);
  }

  if (initInvMass)
    diagonal = impulse::isDiagonalInverseMass(
        cast<RankedTensorType>(initInvMass.getType()));
  welfordConfig.diagonal = diagonal;
}

SmallVector<ChainResult> MCMCChainEmitter::emitChains(Value rng) {
  if (numChains <= 1)
    return {emitChain(rng, 0)};

  Value savedKeys = stateInSlot(state_slot::kRng);
  impulse::RandomSplitOp splitOp;
  if (!savedKeys) {
    SmallVector<Type> chainKeyTypes(numChains, rng.getType());
    splitOp = impulse::RandomSplitOp::create(builder, loc, chainKeyTypes, rng);
  }

  SmallVector<ChainResult> chains;
  chains.reserve(numChains);
  for (int64_t chainIdx = 0; chainIdx < numChains; ++chainIdx) {
    Value chainRng = savedKeys ? sliceChain(savedKeys, chainIdx)
                               : splitOp.getResult(chainIdx);
    chains.push_back(emitChain(chainRng, chainIdx));
  }
  return chains;
}

ChainResult MCMCChainEmitter::emitChain(Value rng, int64_t chainIdx) {
  Value stepSize = initStepSize;
  Value invMass = initInvMass;
  std::optional<AdaptationState> saved = savedAdaptation;
  if (numChains > 1) {
    if (Value savedStepSize = stateInSlot(state_slot::kStepSize))
      stepSize = sliceChain(savedStepSize, chainIdx);
    if (Value savedInvMass = stateInSlot(state_slot::kInverseMassMatrix))
      invMass = sliceChain(savedInvMass, chainIdx);
    if (saved) {
      SmallVector<Value> slots = saved->toValues();
      for (Value &slot : slots)
        slot = sliceChain(slot, chainIdx);
      ArrayRef<Value> values = slots;
      saved = AdaptationState::consume(values, /*hasWelford=*/true);
    }
  }

  Value massMatrixSqrt =
      computeMassMatrixSqrt(builder, loc, invMass, positionType);
  ChainPoint state =
      initChainState(rng, chainIdx, stepSize, invMass, massMatrixSqrt);

  if (!invMass) {
    invMass = emitSplatConstant(builder, loc, positionType, 1.0);
    massMatrixSqrt = emitSplatConstant(builder, loc, positionType, 1.0);
  }

  std::optional<AdaptationState> finalAdaptation;
  if (numWarmup > 0) {
    WarmupCarry warmedUp =
        emitWarmup(state, stepSize, invMass, massMatrixSqrt, saved, chainIdx);
    state = warmedUp.chain;
    invMass = warmedUp.invMass;
    massMatrixSqrt = warmedUp.massMatrixSqrt;
    stepSize = conditionalDump(builder, loc, warmedUp.stepSize,
                               "MCMC: adapted step size after warmup",
                               model.debugDump);
    if (adaptMassMatrix) {
      invMass = conditionalDump(
          builder, loc, invMass,
          "MCMC: adapted inverse mass matrix after warmup", model.debugDump);
    }
    if (exposeAdaptation) {
      finalAdaptation = warmedUp.adaptation;
      if (!adaptMassMatrix)
        finalAdaptation->welford =
            initWelford(builder, loc, model.positionSize, diagonal);
    }
  } else if (exposeAdaptation) {
    finalAdaptation =
        loadOrInitAdaptation(stepSize, /*withWelford=*/true, saved);
  }

  ChainResult result = emitSampling(state, stepSize, invMass, massMatrixSqrt);
  result.trace = conditionalDump(builder, loc, result.trace,
                                 "MCMC: collected samples", model.debugDump);
  if (schema && !schema.empty())
    result.stateOut =
        packStateOut(state, stepSize, invMass, finalAdaptation, chainIdx);
  return result;
}

Value MCMCChainEmitter::stateInSlot(StringRef name) const {
  return impulse::stateInSlot(schema, stateIn, name);
}

Value MCMCChainEmitter::chainStateInSlot(StringRef name, int64_t chainIdx) {
  Value value = stateInSlot(name);
  if (value && numChains > 1)
    value = sliceChain(value, chainIdx);
  return value;
}

Value MCMCChainEmitter::sliceChain(Value value, int64_t chainIdx) {
  auto type = cast<RankedTensorType>(value.getType());
  ArrayRef<int64_t> shape = type.getShape();
  SmallVector<int64_t> start(shape.size(), 0),
      limit(shape.begin(), shape.end()), stride(shape.size(), 1);
  start[0] = chainIdx;
  limit[0] = chainIdx + 1;
  SmallVector<int64_t> oneShape(shape.begin(), shape.end());
  oneShape[0] = 1;
  auto oneType = RankedTensorType::get(oneShape, type.getElementType());
  Value slice = impulse::SliceOp::create(builder, loc, oneType, value,
                                         builder.getDenseI64ArrayAttr(start),
                                         builder.getDenseI64ArrayAttr(limit),
                                         builder.getDenseI64ArrayAttr(stride));
  auto dropType =
      RankedTensorType::get(shape.drop_front(), type.getElementType());
  return impulse::ReshapeOp::create(builder, loc, dropType, slice);
}

HMCContext MCMCChainEmitter::makeHMCContext(Value invMass, Value massMatrixSqrt,
                                            Value stepSize) const {
  return HMCContext{model,          stepSize,         invMass,
                    massMatrixSqrt, trajectoryLength, numSteps};
}

NUTSContext MCMCChainEmitter::makeNUTSContext(Value invMass,
                                              Value massMatrixSqrt,
                                              Value stepSize,
                                              Value potentialEnergy) const {
  return NUTSContext{makeHMCContext(invMass, massMatrixSqrt, stepSize),
                     potentialEnergy, maxDeltaEnergy, maxTreeDepth};
}

MCMCKernelResult MCMCChainEmitter::runKernel(const ChainPoint &state,
                                             Value stepSize, Value invMass,
                                             Value massMatrixSqrt) {
  if (isHMC)
    return SampleHMC(builder, loc, state,
                     makeHMCContext(invMass, massMatrixSqrt, stepSize));
  return SampleNUTS(builder, loc, state,
                    makeNUTSContext(invMass, massMatrixSqrt, stepSize,
                                    state.potentialEnergy));
}

ChainPoint MCMCChainEmitter::initChainState(Value rng, int64_t chainIdx,
                                            Value stepSize, Value invMass,
                                            Value massMatrixSqrt) {
  Value position = chainStateInSlot(state_slot::kPosition, chainIdx);
  Value grad = chainStateInSlot(state_slot::kGradient, chainIdx);
  Value potentialEnergy =
      chainStateInSlot(state_slot::kPotentialEnergy, chainIdx);
  if (position && grad && potentialEnergy)
    return {position, grad, potentialEnergy, rng};

  HMCContext ctx = makeHMCContext(invMass, massMatrixSqrt, stepSize);
  Value chainInitPosition;
  if (model.logpdfFn) {
    chainInitPosition = initPosition;
    if (numChains > 1)
      chainInitPosition = sliceChain(chainInitPosition, chainIdx);
  }
  return InitHMC(builder, loc, rng, ctx, chainInitPosition);
}

AdaptationState MCMCChainEmitter::loadOrInitAdaptation(
    Value stepSize, bool withWelford,
    const std::optional<AdaptationState> &saved) {
  if (saved)
    return *saved;
  AdaptationState fresh;
  fresh.dualAveraging =
      initDualAveraging(builder, loc, stepSize, /*logOfProduct=*/true);
  if (withWelford)
    fresh.welford = initWelford(builder, loc, model.positionSize, diagonal);
  fresh.windowIdx = emitI64ScalarConstant(builder, loc, 0);
  return fresh;
}

WarmupCarry
MCMCChainEmitter::emitWarmup(const ChainPoint &start, Value stepSize,
                             Value invMass, Value massMatrixSqrt,
                             const std::optional<AdaptationState> &saved,
                             int64_t chainIdx) {
  WarmupSchedule schedule;
  schedule.zero = emitI64ScalarConstant(builder, loc, 0);
  schedule.one = emitI64ScalarConstant(builder, loc, 1);
  Value numWarmupConst = emitI64ScalarConstant(builder, loc, numWarmup);

  SmallVector<AdaptWindow> windows = buildAdaptationSchedule(totalWarmup);
  int64_t numWindows = static_cast<int64_t>(windows.size());
  for (const auto &window : windows)
    schedule.windowEnds.push_back(
        emitI64ScalarConstant(builder, loc, window.end));
  schedule.lastWindow = emitI64ScalarConstant(builder, loc, numWindows - 1);
  schedule.lastIteration = emitI64ScalarConstant(builder, loc, totalWarmup - 1);

  Value initialStepSize =
      conditionalDump(builder, loc, stepSize,
                      "MCMC: initial step size before warmup", model.debugDump);
  AdaptationState adaptation =
      loadOrInitAdaptation(initialStepSize, adaptMassMatrix, saved);
  if (!adaptMassMatrix)
    adaptation.welford.reset();

  WarmupCarry init;
  init.chain = start;
  init.stepSize = initialStepSize;
  init.invMass = invMass;
  init.massMatrixSqrt = massMatrixSqrt;
  init.adaptation = adaptation;
  SmallVector<Type> warmupLoopTypes = init.getTypes(positionType, scalarType);

  auto warmupLoop =
      impulse::ForOp::create(builder, loc, warmupLoopTypes, schedule.zero,
                             numWarmupConst, schedule.one, init.toValues());

  Block *warmupBody = builder.createBlock(&warmupLoop.getRegion());
  warmupBody->addArgument(i64TensorType, loc);
  for (Type type : warmupLoopTypes)
    warmupBody->addArgument(type, loc);

  builder.setInsertionPointToStart(warmupBody);

  Value iteration = warmupBody->getArgument(0);
  if (Value warmupOffset =
          chainStateInSlot(state_slot::kWarmupOffset, chainIdx))
    iteration = arith::AddIOp::create(builder, loc, warmupOffset, iteration);
  WarmupCarry carry = WarmupCarry::fromValues(
      warmupBody->getArguments().drop_front(), adaptMassMatrix);
  WarmupCarry next = emitWarmupStep(carry, iteration, schedule);
  impulse::YieldOp::create(builder, loc, next.toValues());

  builder.setInsertionPointAfter(warmupLoop);
  return WarmupCarry::fromValues(warmupLoop.getResults(), adaptMassMatrix);
}

WarmupCarry MCMCChainEmitter::emitWarmupStep(const WarmupCarry &carry,
                                             Value iteration,
                                             const WarmupSchedule &schedule) {
  auto sample = runKernel(carry.chain, carry.stepSize, carry.invMass,
                          carry.massMatrixSqrt);

  DualAveragingConfig daConfig;
  DualAveragingState updatedDaState;
  Value currentStepSizeFromDA;
  Value finalStepSizeFromDA;

  if (adaptStepSize) {
    updatedDaState =
        updateDualAveraging(builder, loc, carry.adaptation.dualAveraging,
                            sample.acceptProb, daConfig);
    currentStepSizeFromDA =
        getStepSizeFromDualAveraging(builder, loc, updatedDaState);
    finalStepSizeFromDA =
        getStepSizeFromDualAveraging(builder, loc, updatedDaState, true);
  } else {
    updatedDaState = carry.adaptation.dualAveraging;
    currentStepSizeFromDA = carry.stepSize;
    finalStepSizeFromDA = carry.stepSize;
  }

  auto isLastIter =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, iteration,
                            schedule.lastIteration);
  Value adaptedStepSizeInLoop =
      impulse::SelectOp::create(builder, loc, scalarType, isLastIter,
                                finalStepSizeFromDA, currentStepSizeFromDA);

  const auto &floatSemantics = cast<FloatType>(elemType).getFloatSemantics();
  auto tinyConst = arith::ConstantOp::create(
      builder, loc, scalarType,
      DenseElementsAttr::get(
          scalarType, FloatAttr::get(elemType, llvm::APFloat::getSmallest(
                                                   floatSemantics))));
  auto maxConst = arith::ConstantOp::create(
      builder, loc, scalarType,
      DenseElementsAttr::get(
          scalarType,
          FloatAttr::get(elemType, llvm::APFloat::getLargest(floatSemantics))));
  adaptedStepSizeInLoop =
      arith::MaximumFOp::create(builder, loc, adaptedStepSizeInLoop, tinyConst);
  adaptedStepSizeInLoop =
      arith::MinimumFOp::create(builder, loc, adaptedStepSizeInLoop, maxConst);

  auto windowIdxGtZero =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt,
                            carry.adaptation.windowIdx, schedule.zero);
  auto windowIdxLtLast =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                            carry.adaptation.windowIdx, schedule.lastWindow);
  auto isMiddleWindow =
      arith::AndIOp::create(builder, loc, windowIdxGtZero, windowIdxLtLast);

  WelfordState conditionalWelford;
  if (adaptMassMatrix) {
    auto sampleType1D = RankedTensorType::get({model.positionSize}, elemType);
    Value sample1D =
        impulse::ReshapeOp::create(builder, loc, sampleType1D, sample.next.q);
    const WelfordState &welford = *carry.adaptation.welford;
    WelfordState updatedWelfordAfterSample =
        updateWelford(builder, loc, welford, sample1D, welfordConfig);

    conditionalWelford.mean = impulse::SelectOp::create(
        builder, loc, welford.mean.getType(), isMiddleWindow,
        updatedWelfordAfterSample.mean, welford.mean);
    conditionalWelford.m2 = impulse::SelectOp::create(
        builder, loc, welford.m2.getType(), isMiddleWindow,
        updatedWelfordAfterSample.m2, welford.m2);
    conditionalWelford.n = impulse::SelectOp::create(
        builder, loc, welford.n.getType(), isMiddleWindow,
        updatedWelfordAfterSample.n, welford.n);
  }

  Value atWindowEnd = emitI1ScalarConstant(builder, loc, false);

  for (auto [window, windowEnd] : llvm::enumerate(schedule.windowEnds)) {
    auto inWindow = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, carry.adaptation.windowIdx,
        emitI64ScalarConstant(builder, loc, window));
    auto atEnd = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                       iteration, windowEnd);
    auto matchesThisWindow =
        arith::AndIOp::create(builder, loc, inWindow, atEnd);
    atWindowEnd =
        arith::OrIOp::create(builder, loc, atWindowEnd, matchesThisWindow);
  }

  Value newWindowIdx = arith::AddIOp::create(
      builder, loc, carry.adaptation.windowIdx, schedule.one);
  Value windowIdxAfterIncrement =
      impulse::SelectOp::create(builder, loc, i64TensorType, atWindowEnd,
                                newWindowIdx, carry.adaptation.windowIdx);

  auto atMiddleWindowEnd =
      arith::AndIOp::create(builder, loc, atWindowEnd, isMiddleWindow);

  WarmupCarry next;
  next.chain = sample.next;
  next.stepSize = adaptedStepSizeInLoop;
  next.invMass = carry.invMass;
  next.massMatrixSqrt = carry.massMatrixSqrt;
  next.adaptation.dualAveraging = updatedDaState;
  if (adaptMassMatrix)
    next.adaptation.welford = conditionalWelford;
  next.adaptation.windowIdx = windowIdxAfterIncrement;

  emitWindowEndRestart(next, atMiddleWindowEnd);
  return next;
}

void MCMCChainEmitter::emitWindowEndRestart(WarmupCarry &next,
                                            Value atMiddleWindowEnd) {
  auto windowEndValues = [](const WarmupCarry &c) {
    SmallVector<Value> values = {c.invMass, c.massMatrixSqrt};
    if (c.adaptation.welford)
      llvm::append_range(values, c.adaptation.welford->toValues());
    llvm::append_range(values, c.adaptation.dualAveraging.toValues());
    return values;
  };
  SmallVector<Value> keptValues = windowEndValues(next);
  auto ifOp = impulse::IfOp::create(
      builder, loc, ValueRange(keptValues).getTypes(), atMiddleWindowEnd);

  builder.setInsertionPointToStart(builder.createBlock(&ifOp.getTrueBranch()));
  WarmupCarry restarted = next;
  if (adaptMassMatrix) {
    Value newInvMass =
        finalizeWelford(builder, loc, *next.adaptation.welford, welfordConfig);
    Value newMassMatrixSqrt =
        computeMassMatrixSqrt(builder, loc, newInvMass, positionType);
    // finalizeWelford returns rank-1 diagonals; match the carried type.
    if (newInvMass.getType() != next.invMass.getType())
      newInvMass = impulse::ReshapeOp::create(
          builder, loc, next.invMass.getType(), newInvMass);
    if (newMassMatrixSqrt.getType() != next.massMatrixSqrt.getType())
      newMassMatrixSqrt = impulse::ReshapeOp::create(
          builder, loc, next.massMatrixSqrt.getType(), newMassMatrixSqrt);
    restarted.invMass = newInvMass;
    restarted.massMatrixSqrt = newMassMatrixSqrt;
    restarted.adaptation.welford =
        initWelford(builder, loc, model.positionSize, diagonal);
  }
  if (adaptStepSize)
    restarted.adaptation.dualAveraging =
        initDualAveraging(builder, loc, next.stepSize);
  impulse::YieldOp::create(builder, loc, windowEndValues(restarted));

  builder.setInsertionPointToStart(builder.createBlock(&ifOp.getFalseBranch()));
  impulse::YieldOp::create(builder, loc, keptValues);

  builder.setInsertionPointAfter(ifOp);
  SmallVector<Value> results = llvm::to_vector_of<Value>(ifOp.getResults());
  ArrayRef<Value> updated = results;
  next.invMass = updated[0];
  next.massMatrixSqrt = updated[1];
  updated = updated.drop_front(2);
  if (adaptMassMatrix) {
    next.adaptation.welford =
        WelfordState::fromValues(updated.take_front(WelfordState::kNumValues));
    updated = updated.drop_front(WelfordState::kNumValues);
  }
  next.adaptation.dualAveraging = DualAveragingState::fromValues(updated);
}

ChainResult MCMCChainEmitter::emitSampling(ChainPoint &state, Value stepSize,
                                           Value invMass,
                                           Value massMatrixSqrt) {
  int64_t collectionSize = numSamples / thinning;
  int64_t startIdx = numSamples % thinning;

  auto samplesBufferType = RankedTensorType::get(
      {collectionSize, model.getConstrainedPositionSize()}, elemType);
  auto diagnosticsBufferType =
      RankedTensorType::get({collectionSize, 2}, builder.getI1Type());
  auto logDensityBufferType = RankedTensorType::get({collectionSize}, elemType);

  Value samplesBuffer = emitSplatConstant(builder, loc, samplesBufferType, 0.0);
  auto diagnosticsBuffer = arith::ConstantOp::create(
      builder, loc, diagnosticsBufferType,
      DenseElementsAttr::get(diagnosticsBufferType,
                             builder.getBoolAttr(isNUTS)));
  Value logDensityBuffer =
      emitSplatConstant(builder, loc, logDensityBufferType, 0.0);

  Value zero = emitI64ScalarConstant(builder, loc, 0);
  Value one = emitI64ScalarConstant(builder, loc, 1);
  Value numSamplesConst = emitI64ScalarConstant(builder, loc, numSamples);
  Value startIdxConst = emitI64ScalarConstant(builder, loc, startIdx);
  Value thinningConst = emitI64ScalarConstant(builder, loc, thinning);

  if (collectionSize == 0)
    return {samplesBuffer, diagnosticsBuffer, logDensityBuffer, state.rng, {}};

  SmallVector<Type> loopResultTypes = {
      positionType,        positionType,      scalarType,
      state.rng.getType(), samplesBufferType, diagnosticsBufferType,
      logDensityBufferType};
  auto forLoopOp = impulse::ForOp::create(
      builder, loc, loopResultTypes, zero, numSamplesConst, one,
      ValueRange{state.q, state.grad, state.potentialEnergy, state.rng,
                 samplesBuffer, diagnosticsBuffer, logDensityBuffer});

  Block *loopBody = builder.createBlock(&forLoopOp.getRegion());
  loopBody->addArgument(i64TensorType, loc);
  for (Type type : loopResultTypes)
    loopBody->addArgument(type, loc);

  builder.setInsertionPointToStart(loopBody);
  Value iterIdx = loopBody->getArgument(0);
  ChainPoint chainLoop = {loopBody->getArgument(1), loopBody->getArgument(2),
                          loopBody->getArgument(3), loopBody->getArgument(4)};
  Value samplesBufferLoop = loopBody->getArgument(5);
  Value diagnosticsBufferLoop = loopBody->getArgument(6);
  Value logDensityBufferLoop = loopBody->getArgument(7);

  auto sample = runKernel(chainLoop, stepSize, invMass, massMatrixSqrt);
  auto qConstrained =
      impulse::constrainPosition(builder, loc, sample.next.q, model.supports);

  auto iMinusStart =
      arith::SubIOp::create(builder, loc, iterIdx, startIdxConst);
  auto storageIdx =
      arith::DivSIOp::create(builder, loc, iMinusStart, thinningConst);

  auto geStartIdx = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::sge, iterIdx, startIdxConst);
  auto modThinning =
      arith::RemSIOp::create(builder, loc, iMinusStart, thinningConst);
  auto modIsZero = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                         modThinning, zero);
  auto shouldStore = arith::AndIOp::create(builder, loc, geStartIdx, modIsZero);

  Value zeroCol = emitI64ScalarConstant(builder, loc, 0);
  auto updatedSamplesBuffer = impulse::DynamicUpdateSliceOp::create(
      builder, loc, samplesBufferType, samplesBufferLoop, qConstrained,
      ValueRange{storageIdx, zeroCol});
  auto selectedSamplesBuffer =
      impulse::SelectOp::create(builder, loc, samplesBufferType, shouldStore,
                                updatedSamplesBuffer, samplesBufferLoop);

  Value oneCol = emitI64ScalarConstant(builder, loc, 1);
  auto accepted1x1 = impulse::ReshapeOp::create(
      builder, loc, RankedTensorType::get({1, 1}, builder.getI1Type()),
      sample.accepted);
  auto divergent1x1 = impulse::ReshapeOp::create(
      builder, loc, RankedTensorType::get({1, 1}, builder.getI1Type()),
      sample.divergent);
  auto updatedDiagnosticsBuffer = impulse::DynamicUpdateSliceOp::create(
      builder, loc, diagnosticsBufferType, diagnosticsBufferLoop, accepted1x1,
      ValueRange{storageIdx, zeroCol});
  updatedDiagnosticsBuffer = impulse::DynamicUpdateSliceOp::create(
      builder, loc, diagnosticsBufferType, updatedDiagnosticsBuffer,
      divergent1x1, ValueRange{storageIdx, oneCol});
  auto selectedDiagnosticsBuffer = impulse::SelectOp::create(
      builder, loc, diagnosticsBufferType, shouldStore,
      updatedDiagnosticsBuffer, diagnosticsBufferLoop);

  auto negU = arith::NegFOp::create(builder, loc, sample.next.potentialEnergy);
  auto logDensity1D = impulse::ReshapeOp::create(
      builder, loc, RankedTensorType::get({1}, elemType), negU);
  auto updatedLogDensityBuffer = impulse::DynamicUpdateSliceOp::create(
      builder, loc, logDensityBufferType, logDensityBufferLoop, logDensity1D,
      ValueRange{storageIdx});
  auto selectedLogDensityBuffer =
      impulse::SelectOp::create(builder, loc, logDensityBufferType, shouldStore,
                                updatedLogDensityBuffer, logDensityBufferLoop);

  impulse::YieldOp::create(
      builder, loc,
      ValueRange{sample.next.q, sample.next.grad, sample.next.potentialEnergy,
                 sample.next.rng, selectedSamplesBuffer,
                 selectedDiagnosticsBuffer, selectedLogDensityBuffer});

  builder.setInsertionPointAfter(forLoopOp);
  state = {forLoopOp.getResult(0), forLoopOp.getResult(1),
           forLoopOp.getResult(2), forLoopOp.getResult(3)};
  return {forLoopOp.getResult(4),
          forLoopOp.getResult(5),
          forLoopOp.getResult(6),
          state.rng,
          {}};
}

SmallVector<Value> MCMCChainEmitter::packStateOut(
    const ChainPoint &state, Value stepSize, Value invMass,
    const std::optional<AdaptationState> &adaptation, int64_t chainIdx) {
  Value warmupOffsetIn = chainStateInSlot(state_slot::kWarmupOffset, chainIdx);
  if (!warmupOffsetIn)
    warmupOffsetIn = emitI64ScalarConstant(builder, loc, 0);
  Value warmupOffsetOut =
      arith::AddIOp::create(builder, loc, warmupOffsetIn,
                            emitI64ScalarConstant(builder, loc, numWarmup));

  SmallVector<Value> stateOut = {state.q, state.grad, state.potentialEnergy,
                                 stepSize, invMass};
  if (adaptation) {
    llvm::append_range(stateOut, adaptation->toValues());
    stateOut.push_back(warmupOffsetOut);
  }
  return stateOut;
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

/// Unit attribute asking an `impulse.regenerate` for the weight vector
/// `[weight, log q(fresh) - log q(replaced)]`, which carries the Hastings
/// correction of the regenerated sites next to the weight.
static constexpr llvm::StringLiteral kMHHastingsAttr = "impulse.mh_hastings";
/// Array of top-level addresses asking an `impulse.regenerate` for a `2 x N`
/// weight instead: column `k` holds 1 and the log density of the site at
/// address `k` where that site runs, else zeros.
static constexpr llvm::StringLiteral kMHSiteScoresAttr =
    "impulse.mh_site_scores";

/// Returns the element of `tensor` at `index` as a rank-0 tensor.
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

/// Returns entry `index` of `vec` along its leading dimension, with that
/// dimension dropped.
static Value takeLeading(OpBuilder &builder, Location loc, Value vec,
                         int64_t index) {
  auto vecType = cast<RankedTensorType>(vec.getType());
  SmallVector<int64_t> start(vecType.getRank(), 0), limit(vecType.getShape()),
      strides(vecType.getRank(), 1), shape(vecType.getShape());
  start[0] = index;
  limit[0] = index + 1;
  shape[0] = 1;
  Value slice = impulse::SliceOp::create(
      builder, loc, RankedTensorType::get(shape, vecType.getElementType()), vec,
      builder.getDenseI64ArrayAttr(start), builder.getDenseI64ArrayAttr(limit),
      builder.getDenseI64ArrayAttr(strides));
  auto elementType = RankedTensorType::get(vecType.getShape().drop_front(),
                                           vecType.getElementType());
  return impulse::ReshapeOp::create(builder, loc, elementType, slice);
}

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

/// Writes `update` into the rank-2 `tensor` starting at row 0, column
/// `column`.
static Value updateColumns(OpBuilder &builder, Location loc, Value tensor,
                           Value update, int64_t column) {
  Value row = emitI64ScalarConstant(builder, loc, 0);
  Value col = emitI64ScalarConstant(builder, loc, column);
  return impulse::DynamicUpdateSliceOp::create(
      builder, loc, tensor.getType(), tensor, update, ValueRange{row, col});
}

/// Flattens `value` to a single row and writes it into `trace` starting at
/// column `column`.
static Value writeTraceColumns(OpBuilder &builder, Location loc, Value trace,
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

/// Draws fresh values for `site`: inline from its distribution when that
/// sampler can be emitted, else by calling its `fn`. Returns replacements for
/// all results of `site`, the rng first.
static SmallVector<Value> emitSiteDraw(OpBuilder &builder,
                                       SymbolTableCollection &symbolTable,
                                       impulse::SampleOp site) {
  Location loc = site.getLoc();
  if (impulse::DistAttr dist = site.getDistAttr();
      dist && impulse::distSamplerEmittable(dist))
    return impulse::emitDistSampler(
        builder, loc, dist, site.getInputs().front(), siteParams(site),
        cast<RankedTensorType>(site.getResult(1).getType()));
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

/// Returns the slice of row 0 of the rank-2 `tensor` of type `sliceType`, a
/// `1 x n` tensor, that takes every `stride`-th column from column `start` on.
static Value emitColumnSlice(OpBuilder &builder, Location loc,
                             RankedTensorType sliceType, Value tensor,
                             int64_t start, int64_t stride = 1) {
  int64_t end = start + (sliceType.getDimSize(1) - 1) * stride + 1;
  return impulse::SliceOp::create(builder, loc, sliceType, tensor,
                                  builder.getDenseI64ArrayAttr({0, start}),
                                  builder.getDenseI64ArrayAttr({1, end}),
                                  builder.getDenseI64ArrayAttr({1, stride}));
}

/// Reads a value of `type` from row 0 of `trace`, starting at column `column`.
static Value sliceTraceValue(OpBuilder &builder, Location loc, Value trace,
                             int64_t column, RankedTensorType type) {
  auto rowType =
      RankedTensorType::get({1, type.getNumElements()}, type.getElementType());
  Value row = emitColumnSlice(builder, loc, rowType, trace, column);
  return impulse::ReshapeOp::create(builder, loc, type, row);
}

/// Reads consecutive values of `types` from row 0 of `trace`, starting at
/// column `offset`.
static SmallVector<Value> sliceTraceValues(OpBuilder &builder, Location loc,
                                           Value trace, int64_t offset,
                                           TypeRange types) {
  SmallVector<Value> values;
  for (Type type : types) {
    auto tensorType = cast<RankedTensorType>(type);
    values.push_back(sliceTraceValue(builder, loc, trace, offset, tensorType));
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
        sliceTraceValue(builder, site.getLoc(), source, offset, type));
    offset += type.getNumElements();
  }
  return values;
}

/// Writes each rank-0 score of `cells` into row 0 of the rank-2 `trace` at
/// the column paired with it; `row0` is the index 0.
static Value writeScoreCells(OpBuilder &builder, Location loc, Value trace,
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

/// Splits the weight vector of a Hastings regenerate into the weight and the
/// Hastings correction.
static std::pair<Value, Value> unpackHastingsWeight(OpBuilder &b, Location loc,
                                                    Value packed) {
  return {scalarAt(b, loc, packed, {0}), scalarAt(b, loc, packed, {1})};
}

/// Half-open [start, end) column ranges of a trace.
using ColumnWindows = SmallVector<std::pair<int64_t, int64_t>>;

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

/// Returns the columns of a `width`-wide trace laid out by `layout` that an MH
/// move regenerating `regenerateAddresses` can change: those of the
/// regenerated sites and of the score tail, and on a trace without score tail
/// also those of sites in branches (their liveness can flip) and sub-calls.
static ColumnWindows mhChangedColumnsFlat(const FlatTraceLayout &layout,
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
/// A proposed MH move. `weight` is the new joint density, or its change for a
/// score-tail trace; `correction` is subtracted from the joint ratio to form
/// log alpha unless `logRatio` gives log alpha directly.
struct MHProposal {
  Value trace;
  Value weight;
  Value correction;
  Value rng;
  Value logRatio;
};

/// The MH accept test of a proposal: whether it is accepted, the weight it
/// carries when accepted, and the rng after the uniform draw.
struct MHDecision {
  Value accepted;
  Value acceptWeight;
  Value rng;
};

/// The trace, weight and rng after one MH step, and whether it accepted.
struct MHStep {
  Value trace;
  Value weight;
  Value rng;
  Value accepted;
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

/// Emits the MH step of `proposal` from `trace` and `weight`: the accept test,
/// and the trace and weight of the proposal where it is accepted. Only the
/// columns in `windows` take the proposed values.
static MHStep emitMHStep(OpBuilder &builder, Location loc, Value trace,
                         Value weight, const MHProposal &proposal,
                         bool hasScoreTail,
                         ArrayRef<std::pair<int64_t, int64_t>> windows) {
  MHDecision decision =
      emitMHDecision(builder, loc, weight, proposal, hasScoreTail);
  Value newTrace = writeBackMHWindows(builder, loc, trace, proposal.trace,
                                      decision.accepted, windows);
  Value newWeight = arith::SelectOp::create(builder, loc, decision.accepted,
                                            decision.acceptWeight, weight);
  return {newTrace, newWeight, decision.rng, decision.accepted};
}

/// Returns whether `op` or an op nested in it has an operand in `values`.
static bool usesAnyOf(Operation *op, const DenseSet<Value> &values) {
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

/// Clones the body of the logpdf block `lpBlock` with its arguments bound to
/// `args` and returns the cloned yields: the joint density, then one density
/// per site.
static SmallVector<Value> cloneLogpdfBody(OpBuilder &builder, Block &lpBlock,
                                          ValueRange args) {
  IRMapping map;
  for (auto [arg, value] : llvm::zip(lpBlock.getArguments(), args))
    map.map(arg, value);
  return cloneBlockBody(builder, lpBlock, map);
}

/// Adds `term` to `sum`, or makes `term` the sum while `sum` is null.
static void accumulate(OpBuilder &builder, Location loc, Value &sum,
                       Value term) {
  sum = sum ? arith::AddFOp::create(builder, loc, sum, term).getResult() : term;
}

/// Returns `sum`, or a zero of `type` when nothing was accumulated into it.
static Value sumOrZero(OpBuilder &builder, Location loc, Value sum,
                       RankedTensorType type) {
  return sum ? sum : emitSplatConstant(builder, loc, type, 0.0);
}

/// Proposes fresh values for the selected sites of `op` from its sampler,
/// reads the other sites from `trace` and scores the result with the logpdf
/// region. `drawOverride` gives the values of replaced sampler draws.
/// `changedColumns` receives the trace columns the move can change. With
/// `delta`, the weight is the sum of new-minus-old scores over the terms the
/// move can change, else the new joint density; with `withLogRatio` too, a
/// move of several sites on a score-tail trace also returns log alpha.
static MHProposal
proposeRegionMove(OpBuilder &b, Location loc, impulse::InferRegionOp op,
                  const RegionTraceLayout &layout, Value trace, Value rng,
                  bool delta, bool withLogRatio,
                  SmallVectorImpl<std::pair<int64_t, int64_t>> &changedColumns,
                  const DenseMap<Operation *, Value> *drawOverride = nullptr) {
  changedColumns.clear();
  ArrayAttr selection = op.getSelectionAttr();
  Region &sampler = op.getSampling();
  Region &logpdf = op.getLogpdf();
  Block &samplerBlock = sampler.front();
  auto traceType = cast<RankedTensorType>(trace.getType());
  int64_t positionSize = layout.positionSize;
  int64_t width = traceType.getShape()[1];
  bool hasScoreTail = width > positionSize;

  DenseSet<Attribute> selectedSet(selection.begin(), selection.end());

  IRMapping map;
  map.map(samplerBlock.getArgument(0), rng);
  for (unsigned i = 1, e = samplerBlock.getNumArguments(); i < e; ++i)
    map.map(samplerBlock.getArgument(i), op.getInputs()[i]);

  Value curRng = rng;
  Value newTrace = trace;
  DenseMap<Attribute, SmallVector<Value>> proposedSelected;
  // Sites count in program order, the order of the per-site yields and of the
  // score-tail columns.
  SmallVector<int64_t> selectedOrdinals;
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
    if (selectedSet.contains(addr)) {
      selectedOrdinals.push_back(siteOrdinal);
      changedColumns.push_back(
          {siteColumn, siteColumn + sampleRegionSize(site)});
      Block &sub = site.getSampling().front();
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
      Operation *samplerYield = sub.getTerminator();
      curRng = subMap.lookupOrDefault(samplerYield->getOperand(0));
      map.map(site.getResult(0), curRng);
      SmallVector<Value> values;
      int64_t column = siteColumn;
      for (unsigned k = 1, e = site.getNumResults(); k < e; ++k) {
        Value v = subMap.lookupOrDefault(samplerYield->getOperand(k));
        map.map(site.getResult(k), v);
        values.push_back(v);
        newTrace = writeTraceColumns(b, loc, newTrace, v, column);
        column += cast<RankedTensorType>(v.getType()).getNumElements();
      }
      proposedSelected[addr] = values;
    } else {
      map.map(site.getResult(0), curRng);
      map.map(site.getResults().drop_front(),
              sliceTraceValues(b, loc, trace, siteColumn,
                               TypeRange(site->getResultTypes()).drop_front()));
    }
  }

  Block &lpBlock = logpdf.front();
  SmallVector<Value> newPositions;
  for (Attribute addr : selection)
    llvm::append_range(newPositions, proposedSelected[addr]);
  SmallVector<Value> newYields = cloneLogpdfBody(b, lpBlock, newPositions);
  Value newWeight = newYields[0];
  SmallVector<bool> dirty =
      delta ? taintedSiteYields(lpBlock, lpBlock.getArguments())
            : SmallVector<bool>(newYields.size() - 1, true);

  if (hasScoreTail) {
    Value row0 = emitI64ScalarConstant(b, loc, 0);
    SmallVector<std::pair<int64_t, Value>> cells;
    for (auto [j, isDirty] : llvm::enumerate(dirty))
      if (isDirty)
        cells.push_back({positionSize + (int64_t)j, newYields[1 + j]});
    newTrace = writeScoreCells(b, loc, newTrace, row0, cells);
  }

  SmallVector<Value> oldYields;
  if (width == positionSize) {
    SmallVector<Value> oldPositions;
    for (Attribute addr : selection)
      llvm::append_range(
          oldPositions,
          sliceTraceValues(b, loc, trace, layout.columnOf.lookup(addr),
                           ValueRange(proposedSelected[addr]).getTypes()));
    oldYields = cloneLogpdfBody(b, lpBlock, oldPositions);
  }
  Value correction;
  Value logRatio;
  for (int64_t j : selectedOrdinals) {
    Value newScore = newYields[1 + j];
    Value oldScore = width == positionSize
                         ? oldYields[1 + j]
                         : scalarAt(b, loc, trace, {0, positionSize + j});
    accumulate(b, loc, correction,
               arith::SubFOp::create(b, loc, newScore, oldScore));
  }

  if (delta) {
    SmallVector<int64_t> dirtyTerms;
    for (auto [j, isDirty] : llvm::enumerate(dirty))
      if (isDirty)
        dirtyTerms.push_back(j);
    // Selected sites' own terms cancel against the proposal density, so log
    // alpha is the sum of the remaining dirty terms.
    bool splitRatio =
        withLogRatio && selectedOrdinals.size() >= 2 && hasScoreTail;
    llvm::SmallDenseSet<int64_t> ownTerms(selectedOrdinals.begin(),
                                          selectedOrdinals.end());
    Value ratio;
    Value sum;
    for (int64_t j : dirtyTerms) {
      Value newScore = newYields[1 + j];
      Value oldScore = scalarAt(b, loc, trace, {0, positionSize + (int64_t)j});
      Value change = arith::SubFOp::create(b, loc, newScore, oldScore);
      accumulate(b, loc, sum, change);
      if (splitRatio && !ownTerms.contains(j))
        accumulate(b, loc, ratio, change);
    }
    auto scalarType = RankedTensorType::get({}, b.getF64Type());
    if (splitRatio)
      logRatio = sumOrZero(b, loc, ratio, scalarType);
    newWeight = sumOrZero(b, loc, sum, scalarType);
  }

  // Only dirty score columns can change; write back just those.
  if (delta) {
    for (auto [j, isDirty] : llvm::enumerate(dirty))
      if (isDirty)
        changedColumns.push_back(
            {positionSize + (int64_t)j, positionSize + (int64_t)j + 1});
  } else if (hasScoreTail) {
    changedColumns.push_back({positionSize, width});
  }
  changedColumns = mergeColumnWindows(std::move(changedColumns));

  return {newTrace, newWeight, correction, curRng, logRatio};
}

/// Proposes the move of the flat-form MH op `op`, an `impulse.mh` or an MH
/// `impulse.infer`, from `trace` with a Hastings `impulse.regenerate` named
/// `name` that calls the model, whose results have `fnResultTypes`, on
/// `inputs`. The weight is the change of the joint density on a score-tail
/// trace, else the new joint density.
template <typename OpT>
static MHProposal proposeFlatMove(OpBuilder &b, Location loc, OpT op,
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

/// Creates the body block of `forOp`, whose arguments are the iteration index
/// and the carried values, and moves the insertion point to its start.
static Block *createForBody(OpBuilder &builder, impulse::ForOp forOp) {
  SmallVector<Type> types{forOp.getLowerBound().getType()};
  llvm::append_range(types, forOp.getResultTypes());
  SmallVector<Location> locs(types.size(), forOp.getLoc());
  return builder.createBlock(&forOp.getRegion(), {}, types, locs);
}

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
  bool ok = true;
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
/// traces. Clears `ok` when such an op has memory effects or feeds a move, an
/// intermediate trace has another reader, or a reader result used after the
/// run would differ on the final trace.
static InterleavedOps analyzeInterleavedOps(MutableArrayRef<MHMove> moves) {
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
      if (starts.size() != 2) {
        interleaved.ok = false;
        return interleaved;
      }
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
    if (!isMemoryEffectFree(o)) {
      interleaved.ok = false;
      return interleaved;
    }
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
      if (!anc || !interleaved.readerSet.contains(anc)) {
        interleaved.ok = false;
        return interleaved;
      }
    }
  for (Operation *o : interleaved.readers)
    for (Value r : o->getResults())
      for (OpOperand &u : r.getUses()) {
        Operation *anc = block->findAncestorOpInBlock(*u.getOwner());
        if (!anc || (moveOps.contains(anc) && anc == u.getOwner())) {
          interleaved.ok = false; // an operand of a move, not a region capture
          return interleaved;
        }
        if (last->isBeforeInBlock(anc)) {
          if (!finalSafe.lookup(o)) {
            interleaved.ok = false;
            return interleaved;
          }
          interleaved.liveOut.insert(r);
        }
      }
  return interleaved;
}

/// Returns whether the run `moves` can be emitted as one batched move.
static bool isBatchable(MutableArrayRef<MHMove> moves) {
  unsigned numMoves = moves.size();
  Block *block = moves.front().op->getBlock();
  Operation *last = moves.back().op;
  impulse::RandomOp firstDraw = moves.front().draw;
  for (MHMove &m : moves) {
    if (m.draw.getRngDistribution() != firstDraw.getRngDistribution() ||
        m.draw.getResult().getType() != firstDraw.getResult().getType() ||
        !sameValueOrConstant(m.draw.getA(), firstDraw.getA()) ||
        !sameValueOrConstant(m.draw.getB(), firstDraw.getB()))
      return false;
  }
  ColumnWindows all;
  for (MHMove &m : moves)
    for (auto w : m.windows)
      all.push_back(w);
  llvm::sort(all);
  for (unsigned i = 1; i < all.size(); ++i)
    if (all[i].first < all[i - 1].second)
      return false;
  for (MHMove &m : moves)
    for (Value v : {m.op.getDiagnostics(), m.op.getLogDensities()})
      for (OpOperand &u : v.getUses()) {
        Operation *anc = block->findAncestorOpInBlock(*u.getOwner());
        if (!anc || !last->isBeforeInBlock(anc))
          return false;
      }
  InterleavedOps interleaved = analyzeInterleavedOps(moves);
  if (!interleaved.ok)
    return false;
  DenseMap<Value, unsigned> &intermediate = interleaved.intermediate;
  DenseMap<Value, llvm::BitVector> &moveDependence = interleaved.moveDependence;
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
      return false;
    Block &lpBlock = op.getLogpdf().front();
    lpDep.seed(lpBlock.getArgument(0), k);
    lpDep.run(lpBlock);
    for (Value y : llvm::drop_begin(lpBlock.getTerminator()->getOperands()))
      if (lpDep.get(y).count() > 1)
        return false;
    SiteDependence sDep(numMoves);
    if (!seedCaptured(sDep, op.getSampling()))
      return false;
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
        return false;
    }
  }
  return true;
}

/// Replaces the run `moves` with one draw of all proposals, a vector accept
/// test and the write-back of the accepted columns.
static void emitBatchedRun(MutableArrayRef<MHMove> moves) {
  unsigned numMoves = moves.size();
  impulse::InferRegionOp first = moves.front().op, last = moves.back().op;
  Location loc = last.getLoc();
  Value trace0 = first.getOriginalTrace();
  Value weight0 = first.getStateIn()[0];
  Value rng0 = first.getInputs()[0];
  auto traceType = cast<RankedTensorType>(trace0.getType());
  auto weightType = cast<RankedTensorType>(weight0.getType());
  Type rngType = rng0.getType();
  InterleavedOps interleaved = analyzeInterleavedOps(moves);
  DenseMap<Operation *, SmallVector<unsigned>> readsIntermediate;
  for (Operation *o : interleaved.readers)
    for (OpOperand &u : o->getOpOperands())
      if (interleaved.intermediate.count(u.get()))
        readsIntermediate[o].push_back(u.getOperandNumber());
  for (unsigned i = 0; i + 1 < numMoves; ++i)
    moves[i].op.getTrace().replaceUsesWithIf(trace0, [&](OpOperand &u) {
      return !(
          u.getOwner() == moves[i + 1].op.getOperation() &&
          u.getOperandNumber() ==
              moves[i + 1].op.getOriginalTraceMutable()[0].getOperandNumber());
    });

  OpBuilder b(last);

  impulse::RandomOp firstDraw = moves.front().draw;
  auto drawType = cast<RankedTensorType>(firstDraw.getResult().getType());
  SmallVector<int64_t> batchedShape{(int64_t)numMoves};
  batchedShape.append(drawType.getShape().begin(), drawType.getShape().end());
  auto proposals = impulse::RandomOp::create(
      b, loc,
      TypeRange{rngType,
                RankedTensorType::get(batchedShape, drawType.getElementType())},
      rng0, firstDraw.getA(), firstDraw.getB(),
      firstDraw.getRngDistributionAttr());
  auto vecType =
      RankedTensorType::get({(int64_t)numMoves}, weightType.getElementType());
  Value rngOut = proposals.getOutputRngState();
  Value uniform = emitUniformDraw(b, loc, rngOut, vecType);
  Value logRand = math::LogOp::create(b, loc, uniform);

  SmallVector<Value> newTraces, deltaCells, correctionCells;
  SmallVector<ColumnWindows> changedWindows(numMoves);
  auto elemType = traceType.getElementType();
  auto rowType = RankedTensorType::get({1, (int64_t)numMoves}, elemType);
  auto colType = RankedTensorType::get({1, 1}, elemType);
  auto unitSlice = [&](Value t, int64_t c) -> Value {
    return emitColumnSlice(b, loc, colType, t, c);
  };
  for (unsigned k = 0; k < numMoves; ++k) {
    DenseMap<Operation *, Value> drawOverride;
    drawOverride[moves[k].draw] = takeLeading(b, loc, proposals.getResult(), k);
    MHProposal proposal = proposeRegionMove(
        b, loc, moves[k].op, moves[k].layout, trace0, rng0, /*delta=*/true,
        /*withLogRatio=*/false, changedWindows[k], &drawOverride);
    newTraces.push_back(proposal.trace);
    deltaCells.push_back(proposal.weight);
    correctionCells.push_back(proposal.correction);
  }

  SmallVector<SmallVector<int64_t>> columns(numMoves);
  for (unsigned k = 0; k < numMoves; ++k)
    for (auto [windowStart, windowEnd] : changedWindows[k])
      for (int64_t c = windowStart; c < windowEnd; ++c)
        columns[k].push_back(c);
  // Slots line up when slot `s` holds a value column in every move or a score
  // column in every move, and each move's own score sits at slot `ownSlot`.
  size_t numSlots = columns.front().size();
  int64_t ownSlot = -1;
  bool alignedSlots = true;
  for (unsigned k = 0; k < numMoves && alignedSlots; ++k) {
    alignedSlots = columns[k].size() == numSlots;
    for (size_t s = 0; s < numSlots && alignedSlots; ++s)
      alignedSlots = (columns[k][s] >= moves[k].layout.positionSize) ==
                     (columns[0][s] >= moves[0].layout.positionSize);
    int64_t own = -1;
    for (size_t s = 0; s < numSlots && alignedSlots; ++s)
      if (columns[k][s] == moves[k].layout.positionSize + moves[k].ordinal)
        own = s;
    alignedSlots = alignedSlots && own >= 0 && (k == 0 || own == ownSlot);
    ownSlot = own;
  }
  SmallVector<Value> proposedAt(numSlots), currentAt(numSlots);
  Value deltaVec, logAlphaVec;
  if (alignedSlots) {
    for (size_t s = 0; s < numSlots; ++s) {
      SmallVector<Value> proposed;
      for (unsigned k = 0; k < numMoves; ++k)
        proposed.push_back(unitSlice(newTraces[k], columns[k][s]));
      proposedAt[s] = packScalars(b, loc, vecType, proposed);
      int64_t stride = columns[1][s] - columns[0][s];
      bool even = stride >= 1;
      for (unsigned k = 1; k < numMoves && even; ++k)
        even = columns[k][s] - columns[k - 1][s] == stride;
      if (even) {
        currentAt[s] = impulse::ReshapeOp::create(
            b, loc, vecType,
            emitColumnSlice(b, loc, rowType, trace0, columns[0][s], stride));
      } else {
        SmallVector<Value> incoming;
        for (unsigned k = 0; k < numMoves; ++k)
          incoming.push_back(unitSlice(trace0, columns[k][s]));
        currentAt[s] = packScalars(b, loc, vecType, incoming);
      }
    }
    for (size_t s = 0; s < numSlots; ++s) {
      if (columns[0][s] < moves[0].layout.positionSize)
        continue;
      accumulate(b, loc, deltaVec,
                 arith::SubFOp::create(b, loc, proposedAt[s], currentAt[s]));
    }
    Value correctionVec =
        arith::SubFOp::create(b, loc, proposedAt[ownSlot], currentAt[ownSlot]);
    logAlphaVec = arith::SubFOp::create(b, loc, deltaVec, correctionVec);
  } else {
    deltaVec = packScalars(b, loc, vecType, deltaCells);
    logAlphaVec = arith::SubFOp::create(
        b, loc, deltaVec, packScalars(b, loc, vecType, correctionCells));
  }
  Value accepted = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                         logRand, logAlphaVec);
  Value trace = trace0, weight = weight0;
  SmallVector<Value> acceptedOf;
  // In sweep order, matching the sequential weight updates.
  for (unsigned k = 0; k < numMoves; ++k) {
    Value acc = takeLeading(b, loc, accepted, k);
    acceptedOf.push_back(acc);
    Value acceptWeight =
        arith::AddFOp::create(b, loc, weight, takeLeading(b, loc, deltaVec, k));
    weight = arith::SelectOp::create(b, loc, acc, acceptWeight, weight);
  }

  if (!alignedSlots) {
    for (unsigned k = 0; k < numMoves; ++k)
      trace = writeBackMHWindows(b, loc, trace, newTraces[k], acceptedOf[k],
                                 changedWindows[k]);
  } else {
    SmallVector<Value> resultAt(numSlots);
    for (size_t s = 0; s < numSlots; ++s)
      resultAt[s] = arith::SelectOp::create(b, loc, accepted, proposedAt[s],
                                            currentAt[s]);
    // Slots within moveStride columns share a window, viewed as numMoves
    // rows of moveStride columns.
    int64_t moveStride = columns[1][0] - columns[0][0];
    bool stridedByMove = moveStride >= 1;
    for (size_t s = 0; s < numSlots && stridedByMove; ++s)
      for (unsigned k = 0; k < numMoves && stridedByMove; ++k)
        stridedByMove =
            columns[k][s] == columns[0][s] + (int64_t)k * moveStride;
    SmallVector<std::pair<int64_t, SmallVector<size_t>>> groups;
    if (stridedByMove) {
      SmallVector<size_t> order(numSlots);
      for (size_t i = 0; i < numSlots; ++i)
        order[i] = i;
      llvm::sort(order, [&](size_t x, size_t y) {
        return columns[0][x] < columns[0][y];
      });
      for (size_t s : order) {
        if (!groups.empty() && columns[0][s] < groups.back().first + moveStride)
          groups.back().second.push_back(s);
        else
          groups.push_back({columns[0][s], {s}});
      }
      for (size_t i = 0; i < groups.size() && stridedByMove; ++i) {
        int64_t end = groups[i].first + (int64_t)numMoves * moveStride;
        stridedByMove = end <= traceType.getShape()[1] &&
                        (i + 1 == groups.size() || end <= groups[i + 1].first);
      }
    }
    if (stridedByMove) {
      auto blockType =
          RankedTensorType::get({(int64_t)numMoves, moveStride}, elemType);
      auto blockColType =
          RankedTensorType::get({(int64_t)numMoves, 1}, elemType);
      auto winType =
          RankedTensorType::get({1, (int64_t)numMoves * moveStride}, elemType);
      for (auto &[groupStart, slots] : groups) {
        Value block = impulse::ReshapeOp::create(
            b, loc, blockType,
            emitColumnSlice(b, loc, winType, trace0, groupStart));
        for (size_t s : slots) {
          Value slotCol =
              impulse::ReshapeOp::create(b, loc, blockColType, resultAt[s]);
          block =
              updateColumns(b, loc, block, slotCol, columns[0][s] - groupStart);
        }
        Value blockWin = impulse::ReshapeOp::create(b, loc, winType, block);
        trace = updateColumns(b, loc, trace, blockWin, groupStart);
      }
    } else {
      for (size_t s = 0; s < numSlots; ++s) {
        Value row = impulse::ReshapeOp::create(b, loc, rowType, resultAt[s]);
        for (unsigned k = 0; k < numMoves; ++k) {
          Value unitCell = unitSlice(row, k);
          trace = updateColumns(b, loc, trace, unitCell, columns[k][s]);
        }
      }
    }
  }

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

  for (unsigned k = 0; k < numMoves; ++k) {
    impulse::InferRegionOp op = moves[k].op;
    op.getDiagnostics().replaceAllUsesWith(acceptedOf[k]);
    auto ldType = cast<ShapedType>(op.getLogDensities().getType());
    if (!op.getLogDensities().use_empty())
      op.getLogDensities().replaceAllUsesWith(
          arith::ConstantOp::create(b, loc, b.getZeroAttr(ldType)));
  }
  last.getTrace().replaceAllUsesWith(trace);
  last.getOutputRngState().replaceAllUsesWith(rngOut);
  last.getStateOut()[0].replaceAllUsesWith(weight);
  for (unsigned k = numMoves; k-- > 0;)
    moves[k].op->erase();
}

/// Replaces the runs of chained single-site MH moves under `root` with batched
/// moves, each run split into the longest batchable pieces from its start.
static void batchMHRuns(Operation *root) {
  SmallVector<SmallVector<MHMove>> runs;
  root->walk([&](Block *block) {
    SmallVector<MHMove> run;
    for (Operation &o : *block) {
      auto op = dyn_cast<impulse::InferRegionOp>(&o);
      if (!op)
        continue;
      std::optional<MHMove> m = analyzeMHMove(op);
      if (!m) {
        continue;
      }
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
  SmallVector<SmallVector<MHMove>> batches;
  for (SmallVector<MHMove> &run : runs) {
    size_t i = 0;
    while (i < run.size()) {
      size_t j = std::min(run.size(), i + kMaxBatchSize) - 1;
      for (; j > i; --j)
        if (isBatchable(MutableArrayRef<MHMove>(run).slice(i, j + 1 - i)))
          break;
      if (j > i)
        batches.push_back(
            SmallVector<MHMove>(run.begin() + i, run.begin() + j + 1));
      i = j + 1;
    }
  }
  for (SmallVector<MHMove> &batch : batches)
    emitBatchedRun(batch);
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
      auto call =
          func::CallOp::create(rewriter, op.getLoc(), newFn.getName(),
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
          if (failed(
                  scoreAndTraceSite(rewriter, sampleOp,
                                    ArrayRef<Value>(sampledValues).drop_front(),
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
            sampleOp.getFnAttr(), sampleOp.getInputs(),
            nested->subAllAddresses);
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

  /// Lowers an `impulse.infer` or `impulse.infer_region` with an HMC or NUTS
  /// config to its chains.
  template <typename MCMCOpT>
  struct LowerMCMCPattern : public mlir::OpRewritePattern<MCMCOpT> {
    static constexpr bool isRegionForm =
        std::is_same_v<MCMCOpT, impulse::InferRegionOp>;

    bool debugDump;

    LowerMCMCPattern(MLIRContext *context, bool debugDump,
                     PatternBenefit benefit = 1)
        : mlir::OpRewritePattern<MCMCOpT>(context, benefit),
          debugDump(debugDump) {}

    /// Reads the model of an HMC/NUTS op. Reports a match failure for other
    /// inference configs and emits an error for a malformed op.
    FailureOr<ModelContext> resolveModel(MCMCOpT mcmcOp,
                                         PatternRewriter &rewriter) const {
      SymbolTableCollection symbolTable;

      bool hasLogpdfFn = static_cast<bool>(mcmcOp.getLogpdfFnAttr());

      if constexpr (isRegionForm) {
        if (mcmcOp.getLogpdf().empty()) {
          mcmcOp.emitError("Impulse: infer_region requires a constructed "
                           "logpdf region (run sicm first)");
          return failure();
        }
        if (!hasLogpdfFn && mcmcOp.getSampling().empty()) {
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
      if (!mcmcOp.getHmcConfig() && !mcmcOp.getNutsConfig())
        return rewriter.notifyMatchFailure(
            mcmcOp, "no hmc_config/nuts_config; not an HMC/NUTS op");

      if (!mcmcOp.getInitStepSize()) {
        mcmcOp.emitError("Impulse: MCMC requires init_step_size parameter");
        return failure();
      }

      if (mcmcOp.getInputs().empty()) {
        mcmcOp.emitError("Impulse: MCMC requires at least rng_state input");
        return failure();
      }

      ModelContext model = readInferModel(mcmcOp, debugDump);
      if constexpr (isRegionForm) {
        // The logpdf region takes the position values in their order.
        int64_t column = 0;
        for (BlockArgument arg : mcmcOp.getLogpdf().front().getArguments()) {
          model.logpdfArgOffsets.push_back(column);
          column += cast<RankedTensorType>(arg.getType()).getNumElements();
        }
      }
      auto unconstrainedSizeOf = [](ArrayRef<impulse::SupportInfo> supports) {
        int64_t size = 0;
        for (const impulse::SupportInfo &info : supports)
          size += info.unconstrainedSize;
        return size;
      };

      if (hasLogpdfFn) {
        auto initPosType =
            cast<RankedTensorType>(mcmcOp.getInitPosition().getType());
        // `init_position` is `1xN` or chain-major `Cx1xN`; N is per chain.
        model.positionSize =
            initPosType.getNumElements() / mcmcOp.getNumChains();
      } else if constexpr (isRegionForm) {
        if (failed(verifySelectedSupports(mcmcOp)))
          return failure();
        FailureOr<RegionPositionLayout> layout =
            collectRegionLayout(mcmcOp, /*positionDeclarationOrder=*/false);
        if (succeeded(layout))
          model.supports = std::move(layout->supports);
        if (model.supports.empty()) {
          mcmcOp.emitError("Impulse: failed to collect per-site support info "
                           "from the infer_region sampler");
          return failure();
        }
        model.positionSize = unconstrainedSizeOf(model.supports);
        if (model.positionSize <= 0)
          return failure();
      } else {
        for (Attribute address : mcmcOp.getSelectionAttr()) {
          if (cast<ArrayAttr>(address).size() > 1) {
            mcmcOp.emitError("Impulse: HMC and NUTS on a model take only "
                             "top-level addresses in `selection`, not ")
                << address;
            return failure();
          }
        }
        if (failed(resolveFlatModel(mcmcOp, /*positionDeclarationOrder=*/false,
                                    symbolTable, model)))
          return failure();
        model.positionSize = unconstrainedSizeOf(model.supports);
      }
      return model;
    }

    LogicalResult matchAndRewrite(MCMCOpT mcmcOp,
                                  PatternRewriter &rewriter) const override {
      FailureOr<ModelContext> model = resolveModel(mcmcOp, rewriter);
      if (failed(model))
        return failure();

      MCMCChainEmitter emitter(rewriter, mcmcOp, std::move(*model));
      SmallVector<ChainResult> chains =
          emitter.emitChains(mcmcOp.getInputs().front());
      if (chains.size() == 1) {
        rewriter.replaceOp(mcmcOp, chains.front().toValues());
        return success();
      }
      ArrayAttr schema = mcmcOp.getStateSchemaAttr();
      rewriter.replaceOp(mcmcOp,
                         stackChainResults(rewriter, mcmcOp.getLoc(), chains,
                                           schema && !schema.empty()));
      return success();
    }
  };

  /// Lowers an `impulse.infer` or `impulse.infer_region` with a VI config to
  /// an SVI loop whose learning rate is the `init_step_size` operand.
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

      if (op.getInputs().empty()) {
        op.emitError("Impulse VI: requires at least rng_state input");
        return failure();
      }

      ModelContext model = readInferModel(op, debugDump);
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
        if (failed(verifySelectedSupports(op)))
          return failure();
        // Per-site draws consume the RNG in program order, so the position
        // follows `all_addresses`.
        FailureOr<RegionPositionLayout> layout =
            collectRegionLayout(op, /*positionDeclarationOrder=*/true);
        if (failed(layout)) {
          op.emitError("Impulse VI: failed to derive position layout from the "
                       "infer_region sampler");
          return failure();
        }
        model.supports = std::move(layout->supports);
        model.logpdfArgOffsets = std::move(layout->argOffsets);
        model.positionSize = model.getConstrainedPositionSize();
        if (model.positionSize <= 0)
          return failure();
      } else if (hasLogpdfFn) {
        model.positionSize =
            cast<RankedTensorType>(op.getInitPosition().getType())
                .getNumElements();
      } else {
        FailureOr<int64_t> numValues = resolveFlatModel(
            op, /*positionDeclarationOrder=*/true, symbolTable, model);
        if (failed(numValues))
          return failure();
        model.positionSize = *numValues;
      }

      return lowerVI(rewriter, op, std::move(model));
    }
  };

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
      auto i1TensorType = RankedTensorType::get({}, rewriter.getI1Type());

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
      // The region form fills this in during the proposal.
      ColumnWindows changedColumns;
      if constexpr (isRegionForm) {
        FailureOr<RegionTraceLayout> regionLayout =
            getRegionTraceLayout(op, /*topLevelOnly=*/true);
        if (failed(regionLayout)) {
          op.emitError("Impulse MH: all_addresses entry without a top-level "
                       "site");
          return failure();
        }
        layout = std::move(*regionLayout);
        for (auto site : op.getSampling()
                             .front()
                             .template getOps<impulse::SampleRegionOp>())
          fullPositionSize += sampleRegionSize(site);
      } else {
        FailureOr<FlatTraceLayout> flatLayout =
            getFlatTraceLayout(op, fn, allAddresses, symbolTable);
        if (failed(flatLayout))
          return failure();
        fullPositionSize = flatLayout->positionSize;
        changedColumns = mhChangedColumnsFlat(*flatLayout, selection,
                                              traceType.getShape()[1]);
      }
      bool hasScoreTail = traceType.getShape()[1] > fullPositionSize;

      // Carries the trace, the weight, the key and whether the step accepted.
      auto oneStep = [&](ValueRange carried) -> SmallVector<Value> {
        Value trace = carried[0], rng = carried[2];
        MHProposal proposal;
        if constexpr (isRegionForm) {
          proposal = proposeRegionMove(rewriter, loc, op, layout, trace, rng,
                                       hasScoreTail,
                                       /*withLogRatio=*/true, changedColumns);
        } else {
          SmallVector<Value> inputs(op.getInputs());
          inputs[0] = rng;
          proposal =
              proposeFlatMove(rewriter, loc, op, inputs, trace,
                              rewriter.getStringAttr(""), fn.getResultTypes());
        }
        MHStep step = emitMHStep(rewriter, loc, trace, carried[1], proposal,
                                 hasScoreTail, changedColumns);
        return {step.trace, step.weight, step.rng, step.accepted};
      };
      SmallVector<Value> out = emitRepeated(
          rewriter, loc, op.getNumSamples(), {initTrace, initWeight, initRng},
          [&]() -> SmallVector<Value> {
            Value notAccepted = arith::ConstantOp::create(
                rewriter, loc, i1TensorType,
                DenseElementsAttr::get(i1TensorType,
                                       rewriter.getBoolAttr(false)));
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

  /// Enumerative Gibbs over scalar BERNOULLI sites: new = u < logistic(s1 -
  /// s0). The batched sweep is sound only when each per-site yield depends on
  /// at most one selected site and no site's parameters depend on another.
  struct LowerGibbsRegionPattern
      : public mlir::OpRewritePattern<impulse::InferRegionOp> {
    using mlir::OpRewritePattern<impulse::InferRegionOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(impulse::InferRegionOp op,
                                  PatternRewriter &rewriter) const override {
      if (!op.getGibbsConfig().has_value())
        return rewriter.notifyMatchFailure(op, "no gibbs_config");
      Location loc = op.getLoc();
      if (op.getSampling().empty() || op.getLogpdf().empty()) {
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
      Block &samplerEntry = op.getSampling().front();
      Block &lpBlock = op.getLogpdf().front();
      Operation *lpTerm = lpBlock.getTerminator();
      unsigned numYields = lpTerm->getNumOperands() - 1;
      ArrayAttr selection = op.getSelectionAttr();

      FailureOr<RegionTraceLayout> layout =
          getRegionTraceLayout(op, /*topLevelOnly=*/false);
      if (failed(layout)) {
        op.emitError("Impulse Gibbs: all_addresses entry without a site");
        return failure();
      }
      DenseMap<Attribute, impulse::SampleRegionOp> &siteByAddress =
          layout->siteByAddress;
      int64_t positionSize = layout->positionSize;
      Value initTrace = op.getOriginalTrace();
      Value initWeight = op.getStateIn()[0];
      Value initRng = op.getInputs()[0];
      auto traceType = cast<RankedTensorType>(initTrace.getType());
      auto weightType = cast<RankedTensorType>(initWeight.getType());
      int64_t width = traceType.getShape()[1];
      bool hasScoreTail = width > positionSize;
      if (hasScoreTail && width != positionSize + (int64_t)numYields) {
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
        selectedCols.push_back(layout->columnOf.lookup(addr));
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
                {layout->columnOf.lookup(addr), off, sampleRegionSize(site)});
          off += sampleRegionSize(site);
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
        SiteDependence lpDep(numSites);
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
        SiteDependence samplerDep(numSites);
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

      auto writeCols = [&](OpBuilder &b, Value trace, int64_t col,
                           Value value) -> Value {
        auto vt = cast<RankedTensorType>(value.getType());
        auto flatType = RankedTensorType::get({1, vt.getNumElements()},
                                              traceType.getElementType());
        Value flat = impulse::ReshapeOp::create(b, loc, flatType, value);
        return updateColumns(b, loc, trace, flat, col);
      };

      // Readers of the trace take constrained branch-local sites from it, so
      // it holds their observed values.
      auto writeObserved = [&](OpBuilder &b, Value trace) -> Value {
        Value constraint = op.getConstraint();
        for (auto [col, off, n] : observedBranchCols) {
          Value obs = emitColumnSlice(
              b, loc, RankedTensorType::get({1, n}, traceType.getElementType()),
              constraint, off);
          trace = writeCols(b, trace, col, obs);
        }
        return trace;
      };

      auto batchedSweep = [&](OpBuilder &b, Value trace, Value weight,
                              Value rng) -> std::array<Value, 3> {
        int64_t n = numSites;
        Type elemType = traceType.getElementType();
        auto scalarType = RankedTensorType::get({}, elemType);
        auto vecType = RankedTensorType::get({n}, elemType);
        auto rowType = RankedTensorType::get({1, n}, elemType);
        auto colType = RankedTensorType::get({1, 1}, elemType);
        int64_t stride = n > 1 ? selectedCols[1] - selectedCols[0] : 1;
        bool evenlySpaced = stride >= 1;
        for (int64_t k = 1; k < n && evenlySpaced; ++k)
          evenlySpaced = selectedCols[k] - selectedCols[k - 1] == stride;

        Value curVec;
        if (evenlySpaced) {
          Value row =
              emitColumnSlice(b, loc, rowType, trace, selectedCols[0], stride);
          curVec = impulse::ReshapeOp::create(b, loc, vecType, row);
        } else {
          SmallVector<Value> cur;
          for (int64_t col : selectedCols)
            cur.push_back(scalarAt(b, loc, trace, {0, col}));
          curVec = packScalars(b, loc, vecType, cur);
        }

        // All sites at v at once: each yield sees at most one selected site.
        SmallVector<Value> yieldsAt[2];
        for (int v = 0; v < 2; ++v) {
          SmallVector<Value> positions;
          for (BlockArgument arg : lpBlock.getArguments())
            positions.push_back(emitSplatConstant(
                b, loc, cast<RankedTensorType>(arg.getType()), (double)v));
          llvm::append_range(yieldsAt[v], llvm::drop_begin(cloneLogpdfBody(
                                              b, lpBlock, positions)));
        }
        SmallVector<Value> logOdds(n);
        for (unsigned j = 0; j < numYields; ++j) {
          int k = siteOfYield[j];
          if (k < 0)
            continue;
          accumulate(
              b, loc, logOdds[k],
              arith::SubFOp::create(b, loc, yieldsAt[1][j], yieldsAt[0][j]));
        }
        for (Value &l : logOdds)
          l = sumOrZero(b, loc, l, scalarType);
        Value lVec = packScalars(b, loc, vecType, logOdds);

        Value p1 = impulse::LogisticOp::create(b, loc, vecType, lVec);
        Value uniform = emitUniformDraw(b, loc, rng, vecType);
        Value take1 = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                            uniform, p1);
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
          weight = arith::AddFOp::create(b, loc, weight,
                                         takeLeading(b, loc, delta, k));

        Value newRow = impulse::ReshapeOp::create(b, loc, rowType, newVals);
        if (evenlySpaced && stride == 1) {
          trace = updateColumns(b, loc, trace, newRow, selectedCols[0]);
        } else {
          for (auto [k, col] : llvm::enumerate(selectedCols)) {
            Value colSlice = emitColumnSlice(b, loc, colType, newRow, k);
            trace = updateColumns(b, loc, trace, colSlice, col);
          }
        }
        if (hasScoreTail)
          for (unsigned j = 0; j < numYields; ++j)
            if (siteOfYield[j] >= 0)
              trace = writeCols(b, trace, positionSize + j,
                                arith::SelectOp::create(
                                    b, loc,
                                    takeLeading(b, loc, take1, siteOfYield[j]),
                                    yieldsAt[1][j], yieldsAt[0][j]));
        return {writeObserved(b, trace), weight, rng};
      };

      auto oneSweep = [&](OpBuilder &b, Value trace, Value weight,
                          Value rng) -> std::array<Value, 3> {
        if (batched)
          return batchedSweep(b, trace, weight, rng);
        SmallVector<Value> cur;
        for (auto [k, col] : llvm::enumerate(selectedCols))
          cur.push_back(sliceTraceValue(
              b, loc, trace, col,
              cast<RankedTensorType>(lpBlock.getArgument(k).getType())));
        auto scalarType = RankedTensorType::get({}, traceType.getElementType());
        for (unsigned k = 0; k < selectedCols.size(); ++k) {
          auto argType =
              cast<RankedTensorType>(lpBlock.getArgument(k).getType());
          SmallVector<Value> yieldsAt[2];
          Value scoreAt[2];
          for (int v = 0; v < 2; ++v) {
            SmallVector<Value> positions(cur);
            positions[k] = emitSplatConstant(b, loc, argType, (double)v);
            SmallVector<Value> yields = cloneLogpdfBody(b, lpBlock, positions);
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
          Value uniform = emitUniformDraw(b, loc, rng, weightType);
          Value take1 = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
                                              uniform, p1);
          Value curScalar =
              impulse::ReshapeOp::create(b, loc, scalarType, cur[k]);
          Value curIs1 = arith::CmpFOp::create(
              b, loc, arith::CmpFPredicate::OGT, curScalar,
              emitSplatConstant(b, loc, scalarType, 0.5));
          Value sNew =
              arith::SelectOp::create(b, loc, take1, scoreAt[1], scoreAt[0]);
          Value sCur =
              arith::SelectOp::create(b, loc, curIs1, scoreAt[1], scoreAt[0]);
          weight = arith::AddFOp::create(
              b, loc, weight, arith::SubFOp::create(b, loc, sNew, sCur));
          Value oneArg = emitSplatConstant(b, loc, argType, 1.0);
          Value zeroArg = emitSplatConstant(b, loc, argType, 0.0);
          Value newVal =
              arith::SelectOp::create(b, loc, take1, oneArg, zeroArg);
          cur[k] = newVal;
          trace = writeCols(b, trace, selectedCols[k], newVal);
          if (hasScoreTail)
            for (unsigned j = 0; j < numYields; ++j)
              if (dirtyOf[k][j])
                trace = writeCols(b, trace, positionSize + j,
                                  arith::SelectOp::create(b, loc, take1,
                                                          yieldsAt[1][j],
                                                          yieldsAt[0][j]));
        }
        return {writeObserved(b, trace), weight, rng};
      };

      SmallVector<Value> out = emitRepeated(
          rewriter, loc, op.getNumSamples(), {initTrace, initWeight, initRng},
          /*loopInit=*/nullptr, [&](ValueRange carried) {
            std::array<Value, 3> r =
                oneSweep(rewriter, carried[0], carried[1], carried[2]);
            return SmallVector<Value>(r.begin(), r.end());
          });

      // Gibbs always accepts; `log_densities` is a zero placeholder.
      auto diagType = cast<ShapedType>(op.getDiagnostics().getType());
      Value diagnostics = arith::ConstantOp::create(
          rewriter, loc, diagType,
          DenseElementsAttr::get(diagType, rewriter.getBoolAttr(true)));
      Value logDensities = arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(op.getLogDensities().getType()));
      rewriter.replaceOp(op,
                         {out[0], diagnostics, logDensities, out[2], out[1]});
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

  /// Lowers `impulse.mh` to a Hastings `impulse.regenerate` of the selection
  /// and an accept test.
  struct LowerMHPattern : public mlir::OpRewritePattern<impulse::MHOp> {
    using mlir::OpRewritePattern<impulse::MHOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(impulse::MHOp mhOp,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
          mhOp, mhOp.getFnAttr());

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
      auto allAddresses = mhOp.getAllAddressesAttr();

      auto traceType = oldTrace.getType();

      auto nameAttr = mhOp.getNameAttr();
      if (!nameAttr)
        nameAttr = rewriter.getStringAttr("");

      auto selection = mhOp.getSelectionAttr();

      FailureOr<FlatTraceLayout> layout =
          getFlatTraceLayout(mhOp, fn, allAddresses, symbolTable);
      if (failed(layout))
        return failure();
      // On a score-tail trace, regenerate returns the dirty-term weight delta.
      int64_t width = cast<RankedTensorType>(traceType).getShape()[1];
      bool hasScoreTail = width > layout->positionSize;

      MHProposal proposal =
          proposeFlatMove(rewriter, loc, mhOp, mhOp.getInputs(), oldTrace,
                          nameAttr, fn.getResultTypes());
      MHStep step =
          emitMHStep(rewriter, loc, oldTrace, oldWeight, proposal, hasScoreTail,
                     mhChangedColumnsFlat(*layout, selection, width));
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
            FailureOr<SmallVector<Value>> read = readSiteValues(
                rewriter, sampleOp, constraint, constrainedOffset);
            if (failed(read))
              return failure();
            sampledValues = std::move(*read);
          } else {
            if (impulse::DistAttr dist = sampleOp.getDistAttr();
                dist && !impulse::distSamplerEmittable(dist))
              return sampleOp.emitError(
                  "Impulse: distribution sampler is not "
                  "emittable; the site must be constrained");
            sampledValues = emitSiteDraw(rewriter, symbolTable, sampleOp);
          }
          if (failed(scoreAndTraceSite(
                  rewriter, sampleOp,
                  ArrayRef<Value>(sampledValues).drop_front(), *clone, carry,
                  symbolTable,
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

        auto subConstraintType = RankedTensorType::get({1, subConstraintSize},
                                                       rewriter.getF64Type());
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
  /// the new trace and the weight: the joint log density, or its change on a
  /// score-tail trace, in the form `kMHHastingsAttr` or `kMHSiteScoresAttr`
  /// ask for.
  struct LowerRegeneratePattern
      : public mlir::OpRewritePattern<impulse::RegenerateOp> {
    using mlir::OpRewritePattern<impulse::RegenerateOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(impulse::RegenerateOp op,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;
      ArrayAttr allAddresses = op.getAllAddressesAttr();
      ArrayAttr constrainedAddresses = op.getConstrainedAddressesAttr();
      bool hasConstraint = static_cast<bool>(op.getConstraint());
      // Operand width, not the packed size, as in generate.
      int64_t constraintSize =
          hasConstraint ? cast<RankedTensorType>(op.getConstraint().getType())
                              .getShape()[1]
                        : -1;
      int64_t declaredWidth =
          cast<RankedTensorType>(op.getNewTrace().getType()).getShape()[1];
      FailureOr<TracedClone> clone = beginTracedClone(
          op, op.getFnAttr(), allAddresses, declaredWidth,
          ImpulseMode::Regenerate, constraintSize, "regenerate", symbolTable);
      if (failed(clone))
        return failure();
      FunctionOpInterface fn = clone->fn;
      FunctionOpInterface newFn = clone->newFn;
      int64_t positionSize = clone->positionSize();
      bool hasScoreTail = clone->hasScoreTail();
      int64_t siteOrdinal = 0;

      OpBuilder entryBuilder(clone->bodyStart);
      Location initLoc = clone->bodyStart->getLoc();
      auto scalarType = cast<RankedTensorType>(clone->zeroWeight.getType());
      auto traceType = cast<RankedTensorType>(clone->zeroTrace.getType());
      Value zeroWeight = clone->zeroWeight;
      Value initWeight = zeroWeight;
      Value initTrace = clone->zeroTrace;

      Value prevTrace = newFn.getArgument(0);
      Value constraint = hasConstraint ? newFn.getArgument(1) : Value();

      // On a score-tail trace the weight is the sum over dirty terms
      // (regenerated or dependent on one) of new minus old log density; clean
      // terms cancel in the MH ratio. Taint is computed up front; fresh draws
      // are added as they are created.
      DenseSet<Value> tainted;
      if (hasScoreTail) {
        if (!supportsScoreTail(newFn)) {
          op.emitError("Impulse: regenerate over a score-tail trace requires "
                       "a straight-line body of top-level distribution sites");
          return failure();
        }
        tainted = computeRegenerateTaint(newFn, op.getSelectionAttr());
        // Non-regenerated values are unchanged: start from the previous trace.
        initTrace = emitColumnSlice(entryBuilder, initLoc, traceType, prevTrace,
                                    /*start=*/0);
      }

      ArrayAttr selection = op.getSelectionAttr();
      bool hastings = op->hasAttr(kMHHastingsAttr);
      ArrayAttr siteScoresAttr =
          op->getAttrOfType<ArrayAttr>(kMHSiteScoresAttr);
      bool scoring = static_cast<bool>(siteScoresAttr);
      // Site symbol -> column of the scores vector or of `oldSiteScores`.
      DenseMap<Attribute, int64_t> siteIndex;
      Value oldSiteScores;
      DenseSet<Value> hastingsTaint;
      auto enclosingIfs = [&](Operation *site) {
        SmallVector<impulse::IfOp> ifs;
        for (Operation *p = site->getParentOp(); p && p != newFn.getOperation();
             p = p->getParentOp())
          if (auto ifOp = dyn_cast<impulse::IfOp>(p))
            ifs.push_back(ifOp);
        return ifs;
      };
      auto inFlippableBranch = [&](Operation *site) {
        return llvm::any_of(enclosingIfs(site), [&](impulse::IfOp ifOp) {
          return hastingsTaint.contains(ifOp.getPredicate());
        });
      };
      if (scoring) {
        for (auto [k, addr] : llvm::enumerate(siteScoresAttr))
          siteIndex[cast<ArrayAttr>(addr)[0]] = k;
        auto scoresType = RankedTensorType::get(
            {2, (int64_t)siteScoresAttr.size()}, entryBuilder.getF64Type());
        initWeight = emitSplatConstant(entryBuilder, initLoc, scoresType, 0.0);
      } else if (hastings && !hasScoreTail) {
        // Sites whose old density cannot be evaluated inline at the new state.
        hastingsTaint = computeRegenerateTaint(newFn, selection);
        SmallVector<Attribute> oldScoreAddrs;
        newFn.getFunctionBody().walk([&](impulse::SampleOp s) {
          if (!s.getDistAttr() && !s.getLogpdfFnAttr())
            return;
          bool needsOld;
          if (addressInList(s, selection))
            needsOld =
                !enclosingIfs(s).empty() ||
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
        if (!oldScoreAddrs.empty()) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&newFn.getFunctionBody().front());
          auto scoresType = RankedTensorType::get(
              {2, (int64_t)oldScoreAddrs.size()}, rewriter.getF64Type());
          SmallVector<Type> scoreTypes{prevTrace.getType(), scoresType};
          scoreTypes.append(fn.getResultTypes().begin(),
                            fn.getResultTypes().end());
          SmallVector<Value> scoreInputs(
              newFn.getArguments().drop_front(hasConstraint ? 2 : 1));
          auto scoreOp = impulse::RegenerateOp::create(
              rewriter, op.getLoc(), scoreTypes, op.getFnAttr(), scoreInputs,
              prevTrace, constraint, allAddresses, rewriter.getArrayAttr({}),
              hasConstraint ? constrainedAddresses : ArrayAttr(),
              rewriter.getStringAttr(""));
          scoreOp->setAttr(kMHSiteScoresAttr,
                           rewriter.getArrayAttr(oldScoreAddrs));
          oldSiteScores = scoreOp.getWeight();
        }
      }
      // In MH, constraint entries outside branches may be latent init values.
      auto readsConstraint = [&](impulse::SampleOp s) {
        if (!constraint || !constrainedAddresses)
          return false;
        return !(hastings || scoring) || !enclosingIfs(s).empty();
      };
      // `oldSiteScores` column of a flippable latent (kept if live, else
      // redrawn).
      auto flipLatentIndex = [&](impulse::SampleOp s, bool isSelected) {
        if (!hastings || isSelected)
          return (int64_t)-1;
        auto it = siteIndex.find(s.getSymbolAttr());
        return it == siteIndex.end() ? (int64_t)-1 : it->second;
      };
      auto oldLiveAt = [&](Location loc, int64_t k) -> Value {
        Value flag = scalarAt(rewriter, loc, oldSiteScores, {0, k});
        Value half = emitSplatConstant(rewriter, loc, scalarType, 0.5);
        return arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGT,
                                     flag, half);
      };
      auto recordSiteScore = [&](Location loc, Value acc, int64_t k,
                                 Value score) -> Value {
        auto t11 = RankedTensorType::get({1, 1}, rewriter.getF64Type());
        Value one = emitSplatConstant(rewriter, loc, t11, 1.0);
        Value scoreRow = impulse::ReshapeOp::create(rewriter, loc, t11, score);
        acc = impulse::DynamicUpdateSliceOp::create(
            rewriter, loc, acc.getType(), acc, one,
            ValueRange{emitI64ScalarConstant(rewriter, loc, 0),
                       emitI64ScalarConstant(rewriter, loc, k)});
        return impulse::DynamicUpdateSliceOp::create(
            rewriter, loc, acc.getType(), acc, scoreRow,
            ValueRange{emitI64ScalarConstant(rewriter, loc, 1),
                       emitI64ScalarConstant(rewriter, loc, k)});
      };

      clone->carry = {initTrace, initWeight, /*offset=*/0,
                      hastings ? Value(zeroWeight) : Value()};
      auto lowerSite = [&](impulse::SampleOp sampleOp,
                           SiteCarry &branch) -> FailureOr<SmallVector<Value>> {
        SmallVector<Value> sampledValues;
        if (sampleOp.getDistAttr() || sampleOp.getLogpdfFnAttr()) {
          Location loc = sampleOp.getLoc();
          if (failed(verifyTopLevelAddress(sampleOp, selection, "selected")))
            return failure();
          bool isSelected = addressInList(sampleOp, selection);
          int64_t sampleOffset =
              clone->layout.firstColumn(sampleOp.getSymbolAttr());

          int64_t flipIdx = flipLatentIndex(sampleOp, isSelected);
          Value oldLive;
          if (isSelected) {
            sampledValues = emitSiteDraw(rewriter, symbolTable, sampleOp);
          } else {
            bool isConstrained = false;
            if (readsConstraint(sampleOp)) {
              if (failed(verifyTopLevelAddress(sampleOp, constrainedAddresses,
                                               "constrained")))
                return failure();
              isConstrained = addressInList(sampleOp, constrainedAddresses);
            }
            int64_t readOffset =
                isConstrained ? computeOffsetForSampleInSelection(
                                    op, fn, constrainedAddresses,
                                    sampleOp.getSymbolAttr(), symbolTable)
                              : sampleOffset;
            FailureOr<SmallVector<Value>> read = readSiteValues(
                rewriter, sampleOp, isConstrained ? constraint : prevTrace,
                readOffset);
            if (failed(read))
              return failure();
            sampledValues = std::move(*read);
            if (flipIdx >= 0) {
              oldLive = oldLiveAt(loc, flipIdx);
              SmallVector<Value> drawn =
                  emitSiteDraw(rewriter, symbolTable, sampleOp);
              for (unsigned i = 1; i < sampledValues.size(); ++i)
                sampledValues[i] = impulse::SelectOp::create(
                    rewriter, loc, sampledValues[i].getType(), oldLive,
                    sampledValues[i], drawn[i]);
              sampledValues[0] = drawn[0];
            }
          }
          ArrayRef<Value> values = ArrayRef<Value>(sampledValues).drop_front();
          if (hastings && !hasScoreTail && (isSelected || flipIdx >= 0))
            hastingsTaint.insert(values.begin(), values.end());
          FailureOr<FunctionOpInterface> logpdfFn =
              verifySiteLogpdf(symbolTable, sampleOp, values.size());
          if (failed(logpdfFn))
            return failure();

          bool termDirty =
              isSelected ||
              (hasScoreTail && llvm::any_of(siteParams(sampleOp), [&](Value v) {
                 return tainted.contains(v);
               }));
          if (hasScoreTail && isSelected)
            tainted.insert(values.begin(), values.end());

          Value oldScore;
          if (hasScoreTail)
            oldScore = scalarAt(rewriter, loc, prevTrace,
                                {0, positionSize + siteOrdinal});

          if (scoring) {
            auto it = siteIndex.find(sampleOp.getSymbolAttr());
            if (it != siteIndex.end())
              branch.weight = recordSiteScore(
                  loc, branch.weight, it->second,
                  emitSiteLogpdf(rewriter, sampleOp, *logpdfFn, values));
          } else if (!hasScoreTail || termDirty) {
            Value logpdf =
                emitSiteLogpdf(rewriter, sampleOp, *logpdfFn, values);
            Value contribution =
                hasScoreTail
                    ? arith::SubFOp::create(rewriter, loc, logpdf, oldScore)
                          .getResult()
                    : logpdf;
            branch.weight = arith::AddFOp::create(rewriter, loc, branch.weight,
                                                  contribution);
            if (hasScoreTail)
              clone->siteScores.push_back(logpdf);
            if (hastings && isSelected) {
              // A prior draw's proposal density cancels its own term.
              branch.hastingsCorrection = arith::AddFOp::create(
                  rewriter, loc, branch.hastingsCorrection, contribution);
              if (!hasScoreTail && !siteIndex.count(sampleOp.getSymbolAttr())) {
                // Parameters unchanged by the move: score the old value inline.
                Value oldLogpdf = emitSiteLogpdf(
                    rewriter, sampleOp, *logpdfFn,
                    sliceTraceValues(
                        rewriter, loc, prevTrace, sampleOffset,
                        TypeRange(sampleOp->getResultTypes()).drop_front()));
                branch.hastingsCorrection = arith::SubFOp::create(
                    rewriter, loc, branch.hastingsCorrection, oldLogpdf);
              }
            } else if (flipIdx >= 0) {
              // Kept: old density, cancelled via oldSiteScores; fresh: new
              // density.
              Value term = arith::SelectOp::create(
                  rewriter, loc, oldLive,
                  scalarAt(rewriter, loc, oldSiteScores, {1, flipIdx}), logpdf);
              branch.hastingsCorrection = arith::AddFOp::create(
                  rewriter, loc, branch.hastingsCorrection, term);
            }
          } else {
            clone->siteScores.push_back(oldScore);
          }
          ++siteOrdinal;

          if (addressInList(sampleOp, allAddresses) &&
              failed(appendSiteValues(rewriter, sampleOp, values, branch.trace,
                                      branch.offset,
                                      /*write=*/!hasScoreTail || isSelected)))
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
        Location loc = sampleOp.getLoc();
        Value subPrevTrace =
            emitColumnSlice(rewriter, loc, nested->getTraceType(), prevTrace,
                            nested->mergeOffset);

        Attribute symbol = sampleOp.getSymbolAttr();
        ArrayAttr subSelection =
            buildSubSelection(rewriter, op.getSelectionAttr(), symbol);
        Value subConstraint;
        ArrayAttr subConstrainedForOp;
        if (constraint && constrainedAddresses) {
          ArrayAttr subConstrainedAddrs =
              buildSubSelection(rewriter, constrainedAddresses, symbol);
          if (!subConstrainedAddrs.empty()) {
            int64_t subConstraintSize = computePositionSizeForSelection(
                sampleOp, nested->genFn, subConstrainedAddrs, symbolTable);
            int64_t subConstraintOffset = computeOffsetForSampleInSelection(
                sampleOp, fn, constrainedAddresses, symbol, symbolTable);
            if (subConstraintSize > 0 && subConstraintOffset >= 0) {
              auto subConstraintType = RankedTensorType::get(
                  {1, subConstraintSize}, rewriter.getF64Type());
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
              llvm::any_of(llvm::drop_begin(sampleOp.getInputs()),
                           [&](Value v) { return hastingsTaint.contains(v); });
          if (inFlippableBranch(sampleOp) ||
              (inputsTainted && (!subSelection.empty() || genHasIf)))
            return sampleOp.emitError(
                "Impulse: MH over a generative call whose sites can "
                "change liveness or parameters under the move is not "
                "supported");
        }

        Type weightType =
            hastings ? RankedTensorType::get({2}, rewriter.getF64Type())
                     : scalarType;
        auto nestedRegenerate = impulse::RegenerateOp::create(
            rewriter, loc, nested->getResultTypes(weightType),
            sampleOp.getFnAttr(), sampleOp.getInputs(), subPrevTrace,
            subConstraint, nested->subAllAddresses, subSelection,
            subConstrainedForOp, rewriter.getStringAttr(""));
        Value subWeight = nestedRegenerate.getWeight();
        if (hastings) {
          nestedRegenerate->setAttr(kMHHastingsAttr, rewriter.getUnitAttr());
          auto [weight, correction] =
              unpackHastingsWeight(rewriter, loc, subWeight);
          branch.weight =
              arith::AddFOp::create(rewriter, loc, branch.weight, weight);
          branch.hastingsCorrection = arith::AddFOp::create(
              rewriter, loc, branch.hastingsCorrection, correction);
        } else if (!scoring) {
          branch.weight =
              arith::AddFOp::create(rewriter, loc, branch.weight, subWeight);
        }
        mergeNestedTrace(rewriter, loc, *nested, nestedRegenerate.getNewTrace(),
                         branch);
        return SmallVector<Value>(nestedRegenerate.getOutputs());
      };

      auto weightOut = [&](Location loc) -> Value {
        const SiteCarry &carry = clone->carry;
        if (!hastings)
          return carry.weight;
        Value correction = carry.hastingsCorrection;
        if (oldSiteScores)
          for (int64_t k = 0,
                       e = cast<RankedTensorType>(oldSiteScores.getType())
                               .getShape()[1];
               k < e; ++k)
            correction = arith::SubFOp::create(
                rewriter, loc, correction,
                scalarAt(rewriter, loc, oldSiteScores, {1, k}));
        return packHastingsWeight(rewriter, loc, carry.weight, correction);
      };
      SmallVector<Value> callOperands{op.getOriginalTrace()};
      if (hasConstraint)
        callOperands.push_back(op.getConstraint());
      llvm::append_range(callOperands, op.getInputs());
      return finishTracedClone(rewriter, op, *clone, lowerSite, weightOut,
                               callOperands);
    }
  };
};

} // end anonymous namespace

void ExpandImpulsePass::runOnOperation() {
  // Must run before the per-op MH patterns lower these moves.
  batchMHRuns(getOperation());

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

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
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
