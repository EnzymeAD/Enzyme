#ifndef ENZYME_BATCH_DIFF_PASS_H
#define ENZYME_BATCH_DIFF_PASS_H

#include "Dialect/Ops.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include <cstdint>

namespace mlir {
namespace enzyme {
namespace batchutils {

struct BatchDiffCacheKey {
  FunctionOpInterface function;
  SmallVector<mlir::Value> inputs;
  SmallVector<enzyme::Activity> inActivity;
  SmallVector<enzyme::Activity> retActivity;
  Block *blk;

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
    if (retActivity < other.retActivity)
      return true;
    if (other.retActivity < retActivity)
      return false;

    return blk < other.blk;
  }
};

template <typename SourceOp>
BatchDiffCacheKey createDiffCacheKey(SourceOp uop, FunctionOpInterface fn) {
  // extract in_activity, ret_activity, in_args
  SmallVector<Activity> inActivity;
  SmallVector<Activity> retActivity;
  SmallVector<Value> in_args;

  auto in_idx = 0;

  for (auto [idx, act] : llvm::enumerate(uop.getActivity())) {
    auto iattr = cast<ActivityAttr>(act);
    auto val = iattr.getValue();
    inActivity.push_back(val);

    in_args.push_back(uop.getInputs()[in_idx]);
    ++in_idx;

    if (val == Activity::enzyme_dup || val == Activity::enzyme_dupnoneed) {
      ++in_idx;
    }
  }

  for (auto [idx, ract] : llvm::enumerate(uop.getRetActivity())) {
    auto iattr = cast<ActivityAttr>(ract);
    auto val = iattr.getValue();
    retActivity.push_back(val);
  }

  batchutils::BatchDiffCacheKey key{fn, in_args, inActivity, retActivity,
                                    uop->getBlock()};
  return key;
}

Value tensorizeArg(OpBuilder &builder, Location &loc,
                   SmallVector<Value> &argList);

Value extractArg(OpBuilder &builder, Location &loc, Type &argTy, Value &val,
                 int64_t index);

SmallVector<MemoryEffects::EffectInstance> collectFnEffects(
    std::map<FunctionOpInterface, SmallVector<MemoryEffects::EffectInstance>>
        &effectCache,
    FunctionOpInterface fnOp);

bool isFnArg(FunctionOpInterface fnOp, Value val);

bool mayAlias(MemoryEffects::EffectInstance &a,
              MemoryEffects::EffectInstance &b,
              mlir::AliasAnalysis &aliasAnalyzer);
} // namespace batchutils
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_BATCH_DIFF_PASS_H
