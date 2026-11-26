//===- SampleDependenceAnalysis.h - Sample dependence of values -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_ANALYSIS_SAMPLEDEPENDENCEANALYSIS_H
#define ENZYME_MLIR_ANALYSIS_SAMPLEDEPENDENCEANALYSIS_H

#include "Dialect/Impulse/Impulse.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace mlir {
namespace impulse {

/// The region of an `impulse.infer_region` that an analysis covers.
enum class AnalysisTarget {
  /// `sampling`, whose sources are the results of the sample sites that have
  /// no address or whose address is in `selection`.
  Sampling,
  /// `logpdf`, whose sources are its block arguments.
  Logpdf,
};

/// The values of one region of an `impulse.infer_region` that depend on the
/// sampled values. A value is sample-dependent if it is a source of the
/// target, or a result of an op other than `impulse.sample_region` that has a
/// sample-dependent operand, or whose regions use a sample-dependent value or
/// hold a sample site with sample-dependent results.
class SampleDependenceAnalysis {
public:
  SampleDependenceAnalysis(InferRegionOp regionOp, AnalysisTarget target);

  bool isSampleDependent(Value value) const;
  /// True if a result of `op` is sample-dependent.
  bool isSampleDependent(Operation *op) const;

  InferRegionOp getRegionOp() const { return regionOp; }
  AnalysisTarget getTarget() const { return target; }

  /// True if `op` is in the target region, at any depth.
  bool isInTargetRegion(Operation *op);

  /// The region of `getRegionOp()` that `getTarget()` names.
  Region &getTargetRegion();

  /// Update the analysis after the IR change of the same name. They only add
  /// values, so the analysis holds a superset of what a fresh one computes.
  void notifyOperationInserted(Operation *op);
  void notifyOperationModified(Operation *op);
  void notifyOperationReplaced(Operation *op, ValueRange replacement);
  void notifyOperationErased(Operation *op);
  void notifyBlockInserted(Block *block);
  void notifyBlockErased(Block *block);

  /// True if `selection` or `logpdf_fn` of the op changed since the analysis
  /// was built; the analysis must then be rebuilt.
  bool sourcesChanged();

  const DenseSet<Value> &getSampleDependentValues() const {
    return sampleDependentValues;
  }

private:
  InferRegionOp regionOp;
  AnalysisTarget target;
  DenseSet<Value> sampleDependentValues;

  Attribute builtSelection;
  Attribute builtLogpdfFn;

  SmallVector<Value> raisedWorklist;

  void runSamplingAnalysis();
  void runLogpdfAnalysis();
  void markSampleDependent(Value value);
  void propagateDependence(Region &region);

  bool isSelectedSite(SampleRegionOp site);
  void raise(Value value);
  void raiseUser(Operation *user);
  void updateOperation(Operation *op);
  void propagateRaised();
};

/// One analysis per (infer_region, target), shared by everything that queries
/// sample dependence while the cache is active.
class SampleDependenceCache {
public:
  /// The analysis of the `target` region of `regionOp`, built at the first
  /// request.
  std::shared_ptr<SampleDependenceAnalysis> get(InferRegionOp regionOp,
                                                AnalysisTarget target);

  template <typename Fn> void forEach(Fn fn) {
    for (auto &entry : entries)
      fn(*entry.second);
  }

  /// Drops the analyses of `regionOp`, only those whose sources changed if
  /// `onlyIfSourcesChanged`.
  void drop(Operation *regionOp, bool onlyIfSourcesChanged);

private:
  SmallVector<std::pair<std::pair<Operation *, AnalysisTarget>,
                        std::shared_ptr<SampleDependenceAnalysis>>>
      entries;
};

/// Makes `cache` the active one on this thread for its lifetime.
class ScopedSampleDependenceCache {
public:
  explicit ScopedSampleDependenceCache(SampleDependenceCache &cache);
  ~ScopedSampleDependenceCache();

private:
  SampleDependenceCache *previous;
};

/// The analysis of the region of the innermost `impulse.infer_region` that
/// holds `op`, from the active cache when there is one and built otherwise.
/// Null when no infer_region holds `op`.
std::shared_ptr<SampleDependenceAnalysis> analyzeEnclosingRegion(Operation *op);

/// Same, for the region that holds `value`.
std::shared_ptr<SampleDependenceAnalysis> analyzeRegionHolding(Value value);

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_ANALYSIS_SAMPLEDEPENDENCEANALYSIS_H
