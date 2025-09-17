#ifndef ENZYME_BATCH_DIFF_PASS_H
#define ENZYME_BATCH_DIFF_PASS_H

#include "Dialect/Ops.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {
namespace enzyme {
namespace batchutils {

struct BatchFwdDiffKey {
  FunctionOpInterface function;
  SmallVector<int64_t> batchSizes;

  // for use in std::map:
  bool operator<(const BatchCacheKey &other) const {
    if (const_cast<FunctionOpInterface &>(function).getName() !=
        const_cast<FunctionOpInterface &>(other.function).getName())
      return const_cast<FunctionOpInterface &>(function).getName() <
             const_cast<FunctionOpInterface &>(other.function).getName();
    return batchSizes < other.batchSizes;
  }
};

} // namespace batchutils
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_BATCH_DIFF_PASS_H
