#ifndef ENZYME_MLIR_ANALYSIS_SAMPLEDEPENDENCEANALYSIS_H
#define ENZYME_MLIR_ANALYSIS_SAMPLEDEPENDENCEANALYSIS_H

#include "Dialect/Ops.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace enzyme {

class SampleDependenceAnalysis {
public:
  explicit SampleDependenceAnalysis(MCMCRegionOp regionOp);

  bool isSampleDependent(Value value) const;
  bool isSampleDependent(Operation *op) const;
  bool canHoist(Operation *op) const;

  ArrayRef<SampleRegionOp> getSampleOps() const { return sampleOps; }

  MCMCRegionOp getRegionOp() const { return regionOp; }

private:
  MCMCRegionOp regionOp;
  DenseSet<Value> sampleDependentValues;
  SmallVector<SampleRegionOp> sampleOps;

  void runAnalysis();
  void markSampleDependent(Value value);
};

bool hoistSampleInvariantOps(MCMCRegionOp regionOp);

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_ANALYSIS_SAMPLEDEPENDENCEANALYSIS_H
