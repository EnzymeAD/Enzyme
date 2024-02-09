#ifndef ENZYME_MLIR_ANALYSIS_ACTIVITYANNOTATIONS_H
#define ENZYME_MLIR_ANALYSIS_ACTIVITYANNOTATIONS_H

#include "mlir/IR/Block.h"

namespace mlir {
class FunctionOpInterface;

namespace enzyme {

void runActivityAnnotations(FunctionOpInterface callee);

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_ANALYSIS_ACTIVITYANNOTATIONS_H
