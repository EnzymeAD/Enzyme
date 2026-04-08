#ifndef ENZYME_MLIR_ANALYSIS_SAMPLEDEPENDENCEANALYSIS_H
#define ENZYME_MLIR_ANALYSIS_SAMPLEDEPENDENCEANALYSIS_H

#include "Dialect/Ops.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace enzyme {

enum class AnalysisTarget {
  Sampler,
  Logpdf,
};

class SampleDependenceAnalysis {
public:
  explicit SampleDependenceAnalysis(MCMCRegionOp regionOp);

  SampleDependenceAnalysis(MCMCRegionOp regionOp, AnalysisTarget target);

  bool isSampleDependent(Value value) const;
  bool isSampleDependent(Operation *op) const;
  bool canHoist(Operation *op) const;

  ArrayRef<SampleRegionOp> getSampleOps() const { return sampleOps; }

  MCMCRegionOp getRegionOp() const { return regionOp; }
  AnalysisTarget getTarget() const { return target; }

  bool isInTargetRegion(Operation *op);

  Region &getTargetRegion();

private:
  MCMCRegionOp regionOp;
  AnalysisTarget target;
  DenseSet<Value> sampleDependentValues;
  SmallVector<SampleRegionOp> sampleOps;

  void runSamplerAnalysis();
  void runLogpdfAnalysis();
  void markSampleDependent(Value value);
  void propagateDependence(Region &region);
};

bool hoistSampleInvariantOps(MCMCRegionOp regionOp);

bool hoistSampleInvariantOps(MCMCRegionOp regionOp, AnalysisTarget target);

bool constructUnifiedLogpdf(MCMCRegionOp regionOp);

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_ANALYSIS_SAMPLEDEPENDENCEANALYSIS_H
