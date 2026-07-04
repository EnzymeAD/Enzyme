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

private:
  InferRegionOp regionOp;
  AnalysisTarget target;
  DenseSet<Value> sampleDependentValues;
  SmallVector<SampleRegionOp> sampleOps;

  void runSamplerAnalysis();
  void runLogpdfAnalysis();
  void markSampleDependent(Value value);
  void propagateDependence(Region &region);
};

bool hoistSampleInvariantOps(InferRegionOp regionOp);

bool hoistSampleInvariantOps(InferRegionOp regionOp, AnalysisTarget target);

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_ANALYSIS_SAMPLEDEPENDENCEANALYSIS_H
