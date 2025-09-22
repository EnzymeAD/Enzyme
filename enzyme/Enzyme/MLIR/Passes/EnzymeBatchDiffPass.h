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

struct BatchDiffCacheKey {
  FunctionOpInterface function;
  SmallVector<mlir::Value> inputs;
  SmallVector<enzyme::Activity> inActivity;
  SmallVector<enzyme::Activity> retActivity;

  // for use in std::map:
  bool operator<(const BatchDiffCacheKey &other) const {
    auto lhs_name = const_cast<FunctionOpInterface &>(function).getName();
    auto rhs_name = const_cast<FunctionOpInterface &>(other.function).getName();

    if (lhs_name < rhs_name)
      return true;
    if (rhs_name < lhs_name)
      return false;
    if (inputs.size() < other.inputs.size())
      return true;
    if (other.inputs.size() < inputs.size())
      return false;

    // Sizes are equal, so compare elements
    for (auto i = 0; i < inputs.size(); ++i) {
      auto lhs_ptr = inputs[i].getAsOpaquePointer();
      auto rhs_ptr = other.inputs[i].getAsOpaquePointer();
      if (lhs_ptr < rhs_ptr)
        return true;
      if (rhs_ptr < lhs_ptr)
        return false;
    }

    if (inActivity < other.inActivity)
      return true;
    if (other.inActivity < inActivity)
      return false;
    return retActivity < other.retActivity;
  }
};

bool isReadOnly(Operation *op);

} // namespace batchutils
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_BATCH_DIFF_PASS_H
