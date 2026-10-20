#ifndef ENZYME_MLIR_ANALYSIS_SAMPLEDEPENDENCEANALYSIS_H
#define ENZYME_MLIR_ANALYSIS_SAMPLEDEPENDENCEANALYSIS_H

#include "Dialect/Impulse/Impulse.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace impulse {

enum class AnalysisTarget {
  Sampler,
  Logpdf,
};

class SampleDependenceAnalysis {
public:
  explicit SampleDependenceAnalysis(InferRegionOp regionOp);

  SampleDependenceAnalysis(InferRegionOp regionOp, AnalysisTarget target);

  bool isSampleDependent(Value value) const;
  bool isSampleDependent(Operation *op) const;
  bool canHoist(Operation *op) const;

  ArrayRef<SampleRegionOp> getSampleOps() const { return sampleOps; }

  InferRegionOp getRegionOp() const { return regionOp; }
  AnalysisTarget getTarget() const { return target; }

  bool isInTargetRegion(Operation *op);

  Region &getTargetRegion();

  /// Updates only raise values: a superset of what a fresh analysis computes.
  void notifyOperationInserted(Operation *op);
  void notifyOperationModified(Operation *op);
  void notifyOperationReplaced(Operation *op, ValueRange replacement);
  void notifyOperationErased(Operation *op);
  void notifyBlockInserted(Block *block);
  void notifyBlockErased(Block *block);

  /// True if the source attributes changed; the analysis must then be rebuilt.
  bool sourcesChanged();

  const DenseSet<Value> &getSampleDependentValues() const {
    return sampleDependentValues;
  }

  struct UpdateStats {
    uint64_t raised = 0;        ///< values raised to sample-dependent
    uint64_t useVisits = 0;     ///< uses visited from raised values
    uint64_t ancestorSteps = 0; ///< enclosing operations visited
    uint64_t opsVisited = 0;    ///< operations whose transfer was evaluated
  };
  const UpdateStats &getUpdateStats() const { return stats; }

private:
  InferRegionOp regionOp;
  AnalysisTarget target;
  DenseSet<Value> sampleDependentValues;
  SmallVector<SampleRegionOp> sampleOps;

  Attribute builtSelection;
  Attribute builtLogpdfFn;
  int64_t builtNumPositionArgs = 0;

  SmallVector<Value> raisedWorklist;
  UpdateStats stats;

  void runSamplerAnalysis();
  void runLogpdfAnalysis();
  void markSampleDependent(Value value);
  void propagateDependence(Region &region);

  bool isSelectedSite(SampleRegionOp site);
  void raise(Value value);
  void raiseUser(Operation *user);
  void updateOperation(Operation *op);
  void propagateRaised();
};

bool hoistSampleInvariantOps(InferRegionOp regionOp);

bool hoistSampleInvariantOps(InferRegionOp regionOp, AnalysisTarget target);

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_ANALYSIS_SAMPLEDEPENDENCEANALYSIS_H
