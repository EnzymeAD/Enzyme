#ifndef ENZYME_MLIR_ANALYSIS_DATAFLOW_ACTIVITYANALYSIS_H
#define ENZYME_MLIR_ANALYSIS_DATAFLOW_ACTIVITYANALYSIS_H

#include "mlir/IR/Block.h"

namespace mlir {
class FunctionOpInterface;

namespace enzyme {

enum class Activity : uint32_t;

void runDataFlowActivityAnalysis(FunctionOpInterface callee,
                                 ArrayRef<enzyme::Activity> argumentActivity,
                                 bool print = false, bool verbose = false,
                                 bool annotate = false);

} // namespace enzyme
} // namespace mlir

#endif
