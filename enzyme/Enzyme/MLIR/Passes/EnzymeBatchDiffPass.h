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
  SmallVector<enzyme::Activity> outActivity;

  // for use in std::map:
  bool operator<(const BatchDiffCacheKey &other) const {
    if (const_cast<FunctionOpInterface &>(function).getName() !=
        const_cast<FunctionOpInterface &>(other.function).getName()) {
      return const_cast<FunctionOpInterface &>(function).getName() <
             const_cast<FunctionOpInterface &>(other.function).getName();
    } else if (inputs != other.inputs) {
      if (inputs.size() != other.inputs.size()) {
        return inputs.size() < other.inputs.size();
      } else {
        bool b = false;
        for (auto [idx, inp_val, other_val] :
             llvm::enumerate(inputs, other.inputs)) {
          if (inp_val != other_val) {
            b = inp_val.getAsOpaquePointer() < other_val.getAsOpaquePointer();
            break;
          }
        }
        return b;
      }
    } else if(inActivity != other.inActivity){
      return inActivity < other.inActivity;
    } 

    return outActivity < other.outActivity; 
  }
};

} // namespace batchutils
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_BATCH_DIFF_PASS_H
