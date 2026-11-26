//===- ExpandImpulse.h - Shared parts of expand-impulse ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The trace layouts, trace helpers and MH step that the translation units of
// the expand-impulse pass share, and the functions that add their patterns.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_PASSES_EXPAND_IMPULSE_H
#define ENZYME_MLIR_PASSES_EXPAND_IMPULSE_H

#include "Dialect/Impulse/Impulse.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {
namespace impulse {

/// Sample sites by symbol.
using SampleOpMap = DenseMap<Attribute, SampleOp>;

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

/// Lays out the sites of `fn` at `addresses` on a trace, in that order. Fails
/// after emitting on `op` when an address is empty, a symbol has no site, an
/// address continues past a site with a `logpdf_fn` or into an empty
/// function, or a value of a site is not a statically shaped tensor.
FailureOr<FlatTraceLayout>
getFlatTraceLayout(Operation *op, FunctionOpInterface fn, ArrayAttr addresses,
                   SymbolTableCollection &symbolTable);

/// Returns the first column of the first address starting with `targetSymbol`
/// on a trace that lays out the sites of `fn` at the addresses in `selection`
/// in order, or -1 when there is none or an address before it has no site.
int64_t computeOffsetForSampleInSelection(Operation *op, FunctionOpInterface fn,
                                          ArrayAttr selection,
                                          Attribute targetSymbol,
                                          SymbolTableCollection &symbolTable);

/// Returns the number of trace columns the values of `site` take.
int64_t sampleRegionSize(SampleRegionOp site);

/// Where the sites of a region-form infer op sit on its trace.
struct RegionTraceLayout {
  DenseMap<Attribute, SampleRegionOp> siteByAddress;
  /// First trace column of each address in `all_addresses`.
  DenseMap<Attribute, int64_t> columnOf;
  /// Columns taken by the values of all addresses, before any score tail.
  int64_t positionSize = 0;
};

/// Lays out the `all_addresses` of `regionOp` on its trace, in that order.
/// Sites are looked up in the whole sampling region, or with `topLevelOnly`
/// in its entry block only. Fails when an address has no site.
FailureOr<RegionTraceLayout> getRegionTraceLayout(InferRegionOp regionOp,
                                                  bool topLevelOnly);

/// Returns the element of `tensor` at `index` as a rank-0 tensor.
Value scalarAt(OpBuilder &b, Location loc, Value tensor,
               ArrayRef<int64_t> index);

/// Writes `update` into the rank-2 `tensor` starting at row 0, column
/// `column`.
Value updateColumns(OpBuilder &builder, Location loc, Value tensor,
                    Value update, int64_t column);

/// Flattens `value` to a single row and writes it into `trace` starting at
/// column `column`.
Value writeTraceColumns(OpBuilder &builder, Location loc, Value trace,
                        Value value, int64_t column);

/// Reads consecutive values of `types` from row 0 of `trace`, starting at
/// column `offset`.
SmallVector<Value> sliceTraceValues(OpBuilder &builder, Location loc,
                                    Value trace, int64_t offset,
                                    TypeRange types);

/// Writes each rank-0 score of `cells` into row 0 of the rank-2 `trace` at
/// the column paired with it; `row0` is the index 0.
Value writeScoreCells(OpBuilder &builder, Location loc, Value trace, Value row0,
                      ArrayRef<std::pair<int64_t, Value>> cells);

/// Returns whether `op` or an op nested in it has an operand in `values`.
bool usesAnyOf(Operation *op, const DenseSet<Value> &values);

/// Unit attribute asking an `impulse.regenerate` for the weight vector
/// `[weight, log q(fresh) - log q(replaced)]`, which carries the Hastings
/// correction of the regenerated sites next to the weight.
inline constexpr llvm::StringLiteral kMHHastingsAttr = "impulse.mh_hastings";

/// Splits the weight vector of a Hastings regenerate into the weight and the
/// Hastings correction.
std::pair<Value, Value> unpackHastingsWeight(OpBuilder &b, Location loc,
                                             Value packed);

/// Half-open [start, end) column ranges of a trace.
using ColumnWindows = SmallVector<std::pair<int64_t, int64_t>>;

/// A proposed MH move. `weight` is the new joint density, or its change for a
/// score-tail trace; `correction` is subtracted from the joint ratio to form
/// log alpha unless `logRatio` gives log alpha directly. Only the `changed`
/// columns of `trace` can differ from the current trace.
struct MHProposal {
  Value trace;
  Value weight;
  Value correction;
  Value rng;
  Value logRatio;
  ColumnWindows changed;
};

/// The trace, weight and rng after one MH step, and whether it accepted.
struct MHStep {
  Value trace;
  Value weight;
  Value rng;
  Value accepted;
};

/// Emits the MH step of `proposal` from `trace` and `weight`: the accept test,
/// and the trace and weight of the proposal where it is accepted.
MHStep emitMHStep(OpBuilder &builder, Location loc, Value trace, Value weight,
                  const MHProposal &proposal, bool hasScoreTail);

/// Proposes the move of the flat-form MH op `op`, an `impulse.mh` or an MH
/// `impulse.infer`, from `trace` with a Hastings `impulse.regenerate` named
/// `name` that calls the model, whose results have `fnResultTypes`, on
/// `inputs`. The weight is the change of the joint density on a score-tail
/// trace, else the new joint density.
template <typename OpT>
MHProposal proposeFlatMove(OpBuilder &b, Location loc, OpT op,
                           ValueRange inputs, Value trace, StringAttr name,
                           TypeRange fnResultTypes);

/// Returns the columns of a `width`-wide trace laid out by `layout` that an MH
/// move regenerating `regenerateAddresses` can change: those of the
/// regenerated sites and of the score tail, and on a trace without score tail
/// also those of sites in branches (their liveness can flip) and sub-calls.
ColumnWindows mhChangedColumnsFlat(const FlatTraceLayout &layout,
                                   ArrayAttr regenerateAddresses,
                                   int64_t width);

/// Adds the patterns that lower `impulse.infer` and `impulse.infer_region`
/// with an HMC, NUTS or VI config. With `debugDump` the lowerings dump
/// intermediate values.
void populateExpandImpulseHMCVIPatterns(RewritePatternSet &patterns,
                                        bool debugDump);

/// Replaces the runs of chained single-site MH moves under `root` with batched
/// moves, each run split into the longest batchable pieces from its start.
void batchMHRuns(Operation *root);

/// Adds the patterns that lower `impulse.infer` and `impulse.infer_region`
/// with an MH or Gibbs config.
void populateExpandImpulseMHGibbsPatterns(RewritePatternSet &patterns);

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_PASSES_EXPAND_IMPULSE_H
