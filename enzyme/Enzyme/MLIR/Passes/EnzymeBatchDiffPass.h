#ifndef ENZYME_BATCH_DIFF_PASS_H
#define ENZYME_BATCH_DIFF_PASS_H

#include "Dialect/Ops.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Interfaces/Utils.h"
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

Type getConcatType(Value val, int64_t width);

Value getConcatValue(OpBuilder &builder, Location &loc,
                         SmallVector<Value> &argList);

Value getExtractValue(OpBuilder &builder, Location &loc, Type &argTy,
                        Value &val, int64_t index);

template <typename SourceOp>
SmallVector<SourceOp>
pruneDiffs(SmallVector<SourceOp> &allDiffs,
           SmallVector<MemoryEffects::EffectInstance> &callerEffects) {
  SmallVector<SourceOp, 2> prunedSources;

  // We first prune and check that all derivative arguments are defined before
  // the first diff in the same block. (ops in allDiffs are guaranteed to belong
  // to the same basic block)
  auto firstDiffOp = allDiffs[0];
  for (auto uop : allDiffs) {
    auto diffArgs = uop.getGradArgs();
    bool definedBeforeFirst = true;

    for (auto diffVal : diffArgs) {
      if (auto diffValOR = dyn_cast<OpResult>(diffVal)) {
        // check that defining op appears before the current op
        auto parentOp = diffValOR.getOwner();
        if (!parentOp->isBeforeInBlock(firstDiffOp.getOperation())) {
          definedBeforeFirst = false;
          break;
        }
      }
    }

    if (definedBeforeFirst) {
      prunedSources.push_back(uop);
    }
  }

  // Account for betweenEffects
  if (callerEffects.size() == 0) {
    // legal to merge since there is no effect overwrite
    return prunedSources;
  } else {

    SmallVector<SourceOp, 2> legalMerge;
    legalMerge.emplace_back(prunedSources[0]);
    SmallVector<MemoryEffects::EffectInstance, 4> betweenEffects;

    for (Operation *cur = prunedSources[0].getOperation();
         cur != prunedSources.back(); cur = cur->getNextNode()) {
      auto curOpEffects = oputils::collectOpEffects(cur);
      if (curOpEffects.has_value()) {
        betweenEffects.clear();
        betweenEffects.append(curOpEffects->begin(), curOpEffects->end());
      }

      bool foundConflict = false;

      for (auto eff : betweenEffects) {
        // read before write
        if (isa<MemoryEffects::Read>(eff.getEffect())) {
          for (auto fneff : callerEffects) {
            if (isa<MemoryEffects::Write>(fneff.getEffect())) {
              if (oputils::mayAlias(eff, fneff)) {
                foundConflict = true;
                break;
              }
            }
          }
        }
        // write before read
        if (isa<MemoryEffects::Write>(eff.getEffect())) {
          for (auto fneff : callerEffects) {
            if (isa<MemoryEffects::Read>(fneff.getEffect())) {
              if (oputils::mayAlias(eff, fneff)) {
                foundConflict = true;
                break;
              }
            }
          }
        }
      }

      if (foundConflict) {
        break;
      }

      auto curFnOp = dyn_cast<SourceOp>(cur);
      if (curFnOp && llvm::is_contained(allDiffs, curFnOp)) {
        legalMerge.push_back(cast<SourceOp>(cur));
      }
    }
    return legalMerge;
  }
}

} // namespace batchutils
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_BATCH_DIFF_PASS_H
