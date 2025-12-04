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
    for (size_t i = 0; i < inputs.size(); ++i) {
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

template <typename SourceOp,
          std::enable_if_t<
              llvm::is_one_of<SourceOp, ForwardDiffOp, AutoDiffOp>::value,
              bool> = true>
SmallVector<MemoryEffects::EffectInstance> findCallerEffects(
    SourceOp callerOp, FunctionOpInterface innerFnOp,
    const SmallVector<MemoryEffects::EffectInstance> &innerEffects) {
  SmallVector<MemoryEffects::EffectInstance> outerEffects;
  for (auto &eff : innerEffects) {

    Value effVal = eff.getValue();
    if (!effVal) {
      // unknown effect which isn't tied to a value, just add to result
      outerEffects.push_back(eff);
      continue;
    }

    // Find primal argument corresponding to effect value
    size_t primalArgPos = 0;
    bool foundPrimal = false;
    if (auto effBA = dyn_cast<BlockArgument>(effVal)) {
      if (llvm::is_contained(innerFnOp.getArguments(), effBA)) {
        foundPrimal = true;
        primalArgPos = effBA.getArgNumber();
      }
    }

    if (!foundPrimal) {
      // TODO: Handle this either as a global value, or a value which
      // is inside of the MLIR function(for inter-proc alias analysis) -
      // Just skip for now, since we don't have interprocedural alias-analysis
      // implemented yet.
      continue;
    }

    // Add primal effects to caller effect map for all ops
    Value primalVal = callerOp.getPrimalInputs()[primalArgPos];
    outerEffects.push_back(
        oputils::getEffectOfVal(primalVal, eff.getEffect(), eff.getResource()));

    // Add derivative effects(only if primal arg is dup)
    // read(primal) -> read(derivative)
    // write(primal) -> write(derivative)

    // find position of dup arg for primal
    bool primalIsDup =
        (cast<ActivityAttr>(callerOp.getActivity()[primalArgPos]).getValue() ==
         Activity::enzyme_dup) ||
        (cast<ActivityAttr>(callerOp.getActivity()[primalArgPos]).getValue() ==
         Activity::enzyme_dupnoneed);

    if (primalIsDup) {
      auto gradArgPos = 0;
      for (auto [idx, act] : llvm::enumerate(callerOp.getActivity())) {
        auto iattr = cast<ActivityAttr>(act);
        auto act_val = iattr.getValue();
        ++gradArgPos;

        if (idx == primalArgPos)
          break;

        if (act_val == Activity::enzyme_dup ||
            act_val == Activity::enzyme_dupnoneed) {
          ++gradArgPos;
        }
      }

      Value dVal = callerOp.getInputs()[gradArgPos];
      // specialze effects based on callerOp type
      if constexpr (std::is_same_v<SourceOp, ForwardDiffOp>) {
        outerEffects.push_back(
            oputils::getEffectOfVal(dVal, eff.getEffect(), eff.getResource()));
      } else {
        outerEffects.push_back(oputils::getEffectOfVal(
            dVal, MemoryEffects::Write::get(), eff.getResource()));
        outerEffects.push_back(oputils::getEffectOfVal(
            dVal, MemoryEffects::Read::get(), eff.getResource()));
      }
    }
  }
  return outerEffects;
}

template <typename SourceOp,
          std::enable_if_t<
              llvm::is_one_of<SourceOp, ForwardDiffOp, AutoDiffOp>::value,
              bool> = true>
llvm::SmallVector<SourceOp, 2> pruneGradDefs(BatchDiffCacheKey &key,
                                          SmallVector<SourceOp> &allDiffs) {
  SmallVector<SourceOp, 2> prunedSources;

  // We first prune and check that all derivative arguments are defined before
  // the first diff in the same block. (ops in allDiffs are guaranteed to belong
  // to the same basic block)
  auto firstDiffOp = allDiffs[0];
  for (auto uop : allDiffs) {

    auto diffArgs = uop.getShadows();
    if constexpr (std::is_same_v<SourceOp, AutoDiffOp>) {
      auto diffeRet = uop.getDifferentialReturns();
      diffArgs.append(diffeRet.begin(), diffeRet.end());
    }

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

  return prunedSources;
}

template <typename SourceOp,
          std::enable_if_t<
              llvm::is_one_of<SourceOp, ForwardDiffOp, AutoDiffOp>::value,
              bool> = true>
llvm::SmallVector<SourceOp> pruneMemoryEffects(
    SymbolTableCollection &symbolTable, BatchDiffCacheKey &key,
    SmallVector<SourceOp> &prunedSources,
    DenseMap<SourceOp, SmallVector<MemoryEffects::EffectInstance>>
        &callerEffectMap,
    llvm::DenseMap<FunctionOpInterface,
                   SmallVector<MemoryEffects::EffectInstance>>
        &innerEffectCache) {
  // Find a mergeable subset of diff operations, which do not violate memory
  // effects wrt reads and writes. Note that callerEffects only contains the
  // aliased set of primal effects, so we have to first map these primal effects
  // to corresponding derivative effects in `prunedSources`
  // TODO: Also handle global values, and non-primal values inside callerEffects
  // through inter-procedural alias analysis. Skip for now

  if (callerEffectMap.empty()) {
    // legal to merge since there is no effect overwrite in mergeable ops
    return prunedSources;
  }

  SmallVector<SourceOp> legalMerge;
  auto lastOp = prunedSources[0];

  SmallVector<MemoryEffects::EffectInstance, 4> betweenEffects;
  for (auto candidateOp : prunedSources) {
    // Update betweenEffects to include memory effects from lastOp to
    // candidateOp
    for (Operation *curr = lastOp.getOperation();
         curr != candidateOp.getOperation(); curr = curr->getNextNode()) {
      auto currSourceOp = dyn_cast<SourceOp>(curr);
      if (currSourceOp && callerEffectMap.contains(currSourceOp)) {
        // curr is/was a mergeable candidate, and we would have already computed
        // its memory effects in the effect map
        betweenEffects.append(callerEffectMap[currSourceOp]);
      } else if (auto currFwdOp = dyn_cast<ForwardDiffOp>(curr)) {
        // curr is a previously un-encountered fwddiff op
        auto fnOp = dyn_cast_or_null<FunctionOpInterface>(
            symbolTable.lookupNearestSymbolFrom(currFwdOp,
                                                currFwdOp.getFnAttr()));
        if (!fnOp)
          continue;

        if (!innerEffectCache.contains(fnOp)) {
          innerEffectCache[fnOp] = oputils::collectFnEffects(fnOp);
        }

        // map to outerEffects
        betweenEffects.append(batchutils::findCallerEffects(
            currFwdOp, fnOp, innerEffectCache[fnOp]));
      } else if (auto currBwdOp = dyn_cast<AutoDiffOp>(curr)) {
        // curr is a previously un-encountered revdiff op
        auto fnOp = dyn_cast_or_null<FunctionOpInterface>(
            symbolTable.lookupNearestSymbolFrom(currBwdOp,
                                                currBwdOp.getFnAttr()));
        if (!fnOp)
          continue;

        if (!innerEffectCache.contains(fnOp)) {
          innerEffectCache[fnOp] = oputils::collectFnEffects(fnOp);
        }

        // map to outerEffects
        betweenEffects.append(batchutils::findCallerEffects(
            currBwdOp, fnOp, innerEffectCache[fnOp]));
      } else {
        // TODO: move forwarddiff and revdiff effect collection specialization
        // from `findCallerEffects` into collectOpEffects(), accounting for
        // inter-procedural alias analysis
        SmallVector<MemoryEffects::EffectInstance> currOpEffects;
        (void)oputils::collectOpEffects(curr, currOpEffects);
        betweenEffects.append(currOpEffects);
      }
    }

    // Check conflicts between betweenEffects and candidateOp. Since the batched
    // version essentially "pushes up" the candidateOp, we ideally want to stop
    // this if it violates the final order of writes to the candidate op owned
    // value
    bool foundConflict = false;
    for (auto candidateEffect : callerEffectMap[candidateOp]) {
      for (auto prevEffect : betweenEffects) {
        // We will disable batching any candidiate operation which re-orders the
        // relative order of writes to the primal and derivative arguments. For
        // this, we alias the underlying effects in the preceding effects and
        // the current  candidate operation.
        if ((isa<MemoryEffects::Write>(prevEffect.getEffect()) &&
             isa<MemoryEffects::Read>(candidateEffect.getEffect())) ||
            (isa<MemoryEffects::Read>(prevEffect.getEffect()) &&
             isa<MemoryEffects::Write>(candidateEffect.getEffect())) ||
            (isa<MemoryEffects::Write>(prevEffect.getEffect()) &&
             isa<MemoryEffects::Write>(candidateEffect.getEffect()))) {

          // if the effects alias each other, then this is not a candidate for
          // merging
          if (oputils::mayAlias(candidateEffect, prevEffect)) {
            foundConflict = true;
            break;
          }
        }
      }
    }

    if (!foundConflict)
      legalMerge.push_back(candidateOp);

    // mark start of next range
    lastOp = candidateOp;
  }

  return legalMerge;
}

} // namespace batchutils
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_BATCH_DIFF_PASS_H
