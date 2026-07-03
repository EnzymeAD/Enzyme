//===- EnzymeAcitivtyOpt.cpp - Optimize activity for differentiation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/DataFlowActivityAnalysis.h"
#include "Analysis/DataFlowAliasAnalysis.h"
#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/EnzymeLogic.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/raw_ostream.h"

#include <Analysis/ActivityAnalysis.h>
#include <deque>

#define DEBUG_TYPE "enzyme-activity-opt"
#define ENZYME_DBGS llvm::dbgs() << "[" << DEBUG_TYPE << "]"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::dataflow;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_ENZYMEACTIVITYOPT
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

struct TaggedActivity {
  Attribute tag;
  bool isConstant;
};

struct ActivityOptimization {
  SmallVector<Activity> argumentActivity;
  SmallVector<Activity> resultActivity;
  SmallVector<TaggedActivity> taggedActivities;
};

static void
printTaggedActivityAnalysis(FunctionOpInterface callee,
                            ArrayRef<TaggedActivity> taggedActivities) {
  llvm::errs() << FlatSymbolRefAttr::get(callee) << ":\n";
  for (const TaggedActivity &activity : taggedActivities) {
    llvm::errs() << "  " << activity.tag << ": "
                 << (activity.isConstant ? "Constant" : "Active") << "\n";
  }
}

// NOTE: Currently, we only optimize for return activities in the classical
// activity analyzer. This is because the ActivityAnalyzer computes and runs
// activity analysis based on a fixed assignment to argument activities
static FailureOr<ActivityOptimization> computeClassicalActivityOptimization(
    Operation *diffOp, FunctionOpInterface callee,
    ArrayRef<Activity> argumentActivity, ArrayRef<Activity> resultActivity,
    SymbolTableCollection &symbolTable) {
  enzyme::MTypeResults TR;
  SmallPtrSet<Block *, 4> blocksNotForAnalysis;
  SmallPtrSet<mlir::Value, 1> constargvals;
  SmallPtrSet<mlir::Value, 1> activeargvals;

  // TODO: check if we need to specialize here for fwddiff and autodiff
  for (auto &&[arg, act] :
       llvm::zip(callee.getFunctionBody().getArguments(), argumentActivity)) {
    if (act == Activity::enzyme_const)
      constargvals.insert(arg);
    else
      activeargvals.insert(arg);
  }

  SmallVector<DIFFE_TYPE> ReturnActivity;
  for (auto act : resultActivity) {
    if (act != Activity::enzyme_const)
      ReturnActivity.push_back(DIFFE_TYPE::DUP_ARG);
    else
      ReturnActivity.push_back(DIFFE_TYPE::CONSTANT);
  }

  DenseMap<Operation *, bool> readOnlyCache;
  enzyme::ActivityAnalyzer activityAnalyzer(blocksNotForAnalysis, readOnlyCache,
                                            constargvals, activeargvals,
                                            ReturnActivity);

  // walk thru all return ops and set activity
  SmallVector<bool> isRetValueConstant(resultActivity.size(), true);
  callee.walk([&](Operation *op) {
    if (!op->hasTrait<OpTrait::ReturnLike>() ||
        op->getParentOp() != callee.getOperation())
      return;

    for (const auto &[idx, opret] : llvm::enumerate(op->getOperands())) {
      bool icv = activityAnalyzer.isConstantValue(TR, opret);
      isRetValueConstant[idx] &= icv;
    }
  });

  // Initialize activity results to be the same as that of the callsite
  ActivityOptimization optimized;
  optimized.argumentActivity.assign(argumentActivity.begin(),
                                    argumentActivity.end());
  optimized.resultActivity.assign(resultActivity.begin(), resultActivity.end());

  for (auto &&[idx, ract] : llvm::enumerate(optimized.resultActivity)) {
    if (isRetValueConstant[idx] && ract != Activity::enzyme_const &&
        ract != Activity::enzyme_constnoneed) {
      if (ract == Activity::enzyme_dup || ract == Activity::enzyme_active) {
        ract = Activity::enzyme_const;
      } else if (ract == Activity::enzyme_activenoneed ||
                 ract == Activity::enzyme_dupnoneed) {
        ract = Activity::enzyme_constnoneed;
      }
    }
  }

  // set return activity based on final inferred activity
  return optimized;
}

static FailureOr<ActivityOptimization> computeDataFlowActivityOptimization(
    Operation *diffOp, FunctionOpInterface callee,
    ArrayRef<Activity> argumentActivity, ArrayRef<Activity> resultActivity,
    SymbolTableCollection &symbolTable, bool collectActivityAnalysis = false) {

  Region &body = callee.getFunctionBody();
  if (body.empty() || body.front().empty())
    return failure();
  if (callee.getArguments().size() != argumentActivity.size()) {
    diffOp->emitError()
        << "callee argument count does not match activity count";
    return failure();
  }

  DataFlowSolver solver;
  solver.load<enzyme::PointsToPointerAnalysis>();
  solver.load<enzyme::AliasAnalysis>(callee.getContext());
  solver.load<enzyme::SparseForwardActivityAnalysis>();
  solver.load<enzyme::DenseForwardActivityAnalysis>(&body.front(),
                                                    argumentActivity);
  solver.load<enzyme::SparseBackwardActivityAnalysis>(symbolTable);
  solver.load<enzyme::DenseBackwardActivityAnalysis>(
      symbolTable, callee, argumentActivity, resultActivity);
  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();

  for (const auto &[arg, activity] :
       llvm::zip(callee.getArguments(), argumentActivity)) {
    // we should only initialize for enzyme_active, enzyme_dup, enzyme_dupnoneed
    if (activity != Activity::enzyme_active &&
        activity != Activity::enzyme_dup &&
        activity != Activity::enzyme_dupnoneed)
      continue;
    auto *argLattice =
        solver.getOrCreateState<enzyme::ForwardValueActivity>(arg);
    (void)argLattice->join(ValueActivity::getActiveVal());
  }

  SmallVector<Operation *> returnOps;
  bool invalidReturn = false;
  callee->walk(
      [&](Operation *op) {
        if (!op->hasTrait<OpTrait::ReturnLike>() ||
            op->getParentOp() != callee.getOperation())
          return;

        if (op->getNumOperands() != resultActivity.size()) {
          invalidReturn = true;
          diffOp->emitError()
              << "return operand count does not match ret_activity count";
          return;
        }

        returnOps.push_back(op);
        for (const auto &[operand, activity] :
             llvm::zip(op->getOperands(), resultActivity)) {
          // Seed scalar return activity here; dense memory activity handles
          // pointed-to data for duplicated memory returns.
          auto *returnLattice =
              solver.getOrCreateState<enzyme::BackwardValueActivity>(operand);
          if (activity == Activity::enzyme_active ||
              activity == Activity::enzyme_activenoneed ||
              activity == Activity::enzyme_dup ||
              activity == Activity::enzyme_dupnoneed) {
            (void)returnLattice->meet(ValueActivity::getActiveVal());
          }
        }
      });

  if (invalidReturn)
    return failure();

  if (returnOps.empty())
    return failure();

  Operation *analysisRoot = callee->getParentOfType<ModuleOp>();
  if (!analysisRoot)
    analysisRoot = callee.getOperation();
  if (failed(solver.initializeAndRun(analysisRoot))) {
    diffOp->emitError() << "dataflow activity analysis failed";
    return failure();
  }

  auto isActiveData = [&](Value value) {
    auto *forward = solver.lookupState<enzyme::ForwardValueActivity>(value);
    auto *backward = solver.lookupState<enzyme::BackwardValueActivity>(value);
    bool forwardActive = forward && forward->getValue().isActiveVal();
    bool backwardActive = backward && backward->getValue().isActiveVal();
    return forwardActive && backwardActive;
  };

  auto isConstantPointer = [&](Value value) {
    auto *backwardMemory = solver.lookupState<enzyme::BackwardMemoryActivity>(
        solver.getProgramPointBefore(&body.front().front()));
    auto *aliasClassLattice =
        solver.lookupState<enzyme::AliasClassLattice>(value);

    if (!backwardMemory || !aliasClassLattice ||
        aliasClassLattice->isUndefined() || aliasClassLattice->isUnknown())
      return false;

    auto scheduleVisit = [](std::deque<DistinctAttr> &frontier,
                            DenseSet<DistinctAttr> &visited,
                            const enzyme::AliasClassSet &aliasClasses) {
      bool known = true;
      (void)aliasClasses.foreachElement(
          [&](DistinctAttr neighbor, enzyme::AliasClassSet::State state) {
            if (state != enzyme::AliasClassSet::State::Defined || !neighbor) {
              known = false;
              return ChangeResult::NoChange;
            }
            if (!visited.contains(neighbor)) {
              visited.insert(neighbor);
              frontier.push_back(neighbor);
            }
            return ChangeResult::NoChange;
          });
      return known;
    };

    for (Operation *returnOp : returnOps) {
      auto *forwardMemory = solver.lookupState<enzyme::ForwardMemoryActivity>(
          solver.getProgramPointAfter(returnOp));
      const enzyme::PointsToSets *pointsToSets =
          solver.lookupState<enzyme::PointsToSets>(
              solver.getProgramPointAfter(returnOp));
      if (!forwardMemory || !pointsToSets)
        return false;

      std::deque<DistinctAttr> frontier;
      DenseSet<DistinctAttr> visited;
      if (!scheduleVisit(frontier, visited,
                         aliasClassLattice->getAliasClassesObject()))
        return false;

      while (!frontier.empty()) {
        DistinctAttr aliasClass = frontier.front();
        frontier.pop_front();

        if (forwardMemory->hasActiveData(aliasClass) &&
            backwardMemory->activeDataFlowsOut(aliasClass))
          return false;

        const enzyme::AliasClassSet &pointsTo =
            pointsToSets->getPointsTo(aliasClass);
        if (pointsTo.isUndefined() || pointsTo.isUnknown())
          return false;
        if (!scheduleVisit(frontier, visited, pointsTo))
          return false;
      }
    }

    return true;
  };

  auto isConstantValue = [&](Value value) {
    if (isa<LLVM::LLVMPointerType, MemRefType, UnrankedMemRefType>(
            value.getType()))
      return isConstantPointer(value);
    return !isActiveData(value);
  };

  // Initialize activity results to be the same as that of the callsite
  ActivityOptimization optimized;
  optimized.argumentActivity.assign(argumentActivity.begin(),
                                    argumentActivity.end());
  optimized.resultActivity.assign(resultActivity.begin(), resultActivity.end());

  for (const auto &[index, arg, activity] :
       llvm::enumerate(callee.getArguments(), argumentActivity)) {
    if (isConstantValue(arg)) {
      if (activity == Activity::enzyme_active ||
          activity == Activity::enzyme_dup ||
          activity == Activity::enzyme_dupnoneed)
        optimized.argumentActivity[index] = Activity::enzyme_const;
    }
  }

  // Conservatively combine all the activities of all ReturnLike ops as a
  // summary for ret_activity originating from the callsite autodiff op
  SmallVector<bool> allReturnValuesConstant(resultActivity.size(), true);
  for (Operation *returnOp : returnOps) {
    for (const auto &[index, operand] :
         llvm::enumerate(returnOp->getOperands())) {
      allReturnValuesConstant[index] =
          allReturnValuesConstant[index] && isConstantValue(operand);
    }
  }

  for (const auto &[index, activity] : llvm::enumerate(resultActivity)) {
    if (allReturnValuesConstant[index]) {
      if (activity == Activity::enzyme_active ||
          activity == Activity::enzyme_dup)
        optimized.resultActivity[index] = Activity::enzyme_const;
      else if (activity == Activity::enzyme_activenoneed ||
               activity == Activity::enzyme_dupnoneed)
        optimized.resultActivity[index] = Activity::enzyme_constnoneed;
    }
  }

  if (collectActivityAnalysis) {
    for (BlockArgument arg : callee.getArguments()) {
      if (Attribute tag = callee.getArgAttr(arg.getArgNumber(), "enzyme.tag"))
        optimized.taggedActivities.push_back({tag, isConstantValue(arg)});
    }

    callee.walk([&](Operation *op) {
      Attribute tag = op->getAttr("tag");
      if (!tag)
        return;
      for (OpResult result : op->getResults())
        optimized.taggedActivities.push_back({tag, isConstantValue(result)});
    });
  }

  return optimized;
}

class MinimizeForwardActivity : public OpRewritePattern<ForwardDiffOp> {
public:
  MinimizeForwardActivity(MLIRContext *context, bool dataflow,
                          bool printActivityAnalysis,
                          PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), dataflow(dataflow),
        printActivityAnalysis(printActivityAnalysis) {}

  LogicalResult matchAndRewrite(ForwardDiffOp op,
                                PatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    SymbolTableCollection symbolTable;
    FunctionOpInterface callee = dyn_cast_or_null<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr()));
    if (!callee)
      return failure();

    SmallVector<Activity> argumentActivity =
        llvm::to_vector(op.getActivity().getAsValueRange<ActivityAttr>());
    SmallVector<Activity> resultActivity =
        llvm::to_vector(op.getRetActivity().getAsValueRange<ActivityAttr>());

    FailureOr<ActivityOptimization> optimized =
        dataflow
            ? computeDataFlowActivityOptimization(op, callee, argumentActivity,
                                                  resultActivity, symbolTable,
                                                  printActivityAnalysis)
            : computeClassicalActivityOptimization(op, callee, argumentActivity,
                                                   resultActivity, symbolTable);

    if (failed(optimized))
      return failure();

    // debug printing activity
    if (printActivityAnalysis && dataflow)
      printTaggedActivityAnalysis(callee, optimized->taggedActivities);

    if (llvm::equal(argumentActivity, optimized->argumentActivity) &&
        llvm::equal(resultActivity, optimized->resultActivity))
      return failure();

    // reconstruct op according to modified activity
    SmallVector<Value> in_args;
    auto in_idx = 0;
    for (const auto &[idx, act, nact] :
         llvm::enumerate(argumentActivity, optimized->argumentActivity)) {
      Value inp = op.getInputs()[in_idx];
      if (act == nact) {
        in_args.push_back(inp);
        in_idx++;
        if (act == Activity::enzyme_dup || act == Activity::enzyme_dupnoneed) {
          inp = op.getInputs()[in_idx];
          in_args.push_back(inp);
          in_idx++;
        }
      } else {
        // Handle const demotion for enzyme_dup/enzyme_dupnoneed/enzyme_active
        in_args.push_back(inp);
        in_idx++;
        // we skip the duplicated derivative arg
        if ((act == Activity::enzyme_dup ||
             act == Activity::enzyme_dupnoneed) &&
            nact == Activity::enzyme_const)
          in_idx++;
      }
    }

    SmallVector<Value> out_args;
    auto out_idx = 0;
    for (const auto &[idx, act, nact] :
         llvm::enumerate(resultActivity, optimized->resultActivity)) {
      if (act == Activity::enzyme_constnoneed)
        continue;

      Value out = op.getOutputs()[out_idx];
      if (act == nact) {
        switch (act) {
        case Activity::enzyme_active:
        case Activity::enzyme_activenoneed:
        case Activity::enzyme_const:
          out_args.push_back(out);
          out_idx++;
          break;

        case Activity::enzyme_dup:
        case Activity::enzyme_dupnoneed:
          out_args.push_back(out);
          out_idx++;

          // handle derivative
          out = op.getOutputs()[out_idx];
          out_args.push_back(out);
          out_idx++;
          break;

        default:
          llvm_unreachable("unexpected return activity arg");
        }
      } else {
        // NOTE: These are the only possible activity demotions
        // enzyme_active -> enzyme_const
        // enzyme_dup -> enzyme_const
        // enzyme_activenoneed -> enzyme_constnoneed
        // enzyme_dupnoneed -> enzyme_constnoneed
        switch (act) {
        case Activity::enzyme_active:
        case Activity::enzyme_dup:
          out_args.push_back(out);
          // we skip the derivative
          out_idx += 2;
          break;

        case Activity::enzyme_dupnoneed:
        case Activity::enzyme_activenoneed:
          // the primal is already skipped. We also skip the derivative.
          out_idx++;
          break;

        default:
          llvm_unreachable("unexpected demotion for return activity");
        }
      }
    }

    TypeRange out_ty = ValueRange(out_args).getTypes();
    ArrayAttr newInActivity = ArrayAttr::get(
        ctx, llvm::map_to_vector(optimized->argumentActivity,
                                 [&](enzyme::Activity act) -> Attribute {
                                   return ActivityAttr::get(ctx, act);
                                 }));

    ArrayAttr newOutActivity = ArrayAttr::get(
        ctx, llvm::map_to_vector(optimized->resultActivity,
                                 [&](enzyme::Activity act) -> Attribute {
                                   return ActivityAttr::get(ctx, act);
                                 }));

    ForwardDiffOp newOp = ForwardDiffOp::create(
        rewriter, op->getLoc(), out_ty, op.getFnAttr(), in_args, newInActivity,
        newOutActivity, op.getWidthAttr(), op.getStrongZeroAttr());

    // Map old uses of op to newOp
    auto oldIdx = 0;
    auto newIdx = 0;
    for (auto [idx, old_val, new_val] :
         llvm::enumerate(resultActivity, optimized->resultActivity)) {

      if (old_val == new_val) {
        // don't index into op if its a const_noneed
        if (old_val == Activity::enzyme_constnoneed) {
          continue;
        }
        // replace use
        op.getOutputs()[oldIdx++].replaceAllUsesWith(
            newOp.getOutputs()[newIdx++]);
        if (old_val == Activity::enzyme_dup) {
          // 2nd replacement for derivative
          op.getOutputs()[oldIdx++].replaceAllUsesWith(
              newOp.getOutputs()[newIdx++]);
        }
      } else {
        // handle all substitutions
        if (new_val == Activity::enzyme_constnoneed &&
            old_val == Activity::enzyme_dupnoneed) {
          ++oldIdx; // skip gradient too
        } else if (new_val == Activity::enzyme_const &&
                   old_val == Activity::enzyme_dup) {
          op.getOutputs()[oldIdx++].replaceAllUsesWith(
              newOp.getOutputs()[newIdx++]);
          ++oldIdx; // skip derivative
        } else {
          llvm_unreachable("unexpected demotion for return activity");
        }
      }
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  bool dataflow;
  bool printActivityAnalysis;
};

class MinimizeReverseActivity : public OpRewritePattern<AutoDiffOp> {
public:
  MinimizeReverseActivity(MLIRContext *context, bool dataflow,
                          bool printActivityAnalysis,
                          PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), dataflow(dataflow),
        printActivityAnalysis(printActivityAnalysis) {}

  LogicalResult matchAndRewrite(AutoDiffOp op,
                                PatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    SymbolTableCollection symbolTable;
    FunctionOpInterface callee = dyn_cast_or_null<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr()));
    if (!callee)
      return failure();

    SmallVector<Activity> argumentActivity =
        llvm::to_vector(op.getActivity().getAsValueRange<ActivityAttr>());
    SmallVector<Activity> resultActivity =
        llvm::to_vector(op.getRetActivity().getAsValueRange<ActivityAttr>());

    FailureOr<ActivityOptimization> optimized =
        dataflow
            ? computeDataFlowActivityOptimization(op, callee, argumentActivity,
                                                  resultActivity, symbolTable,
                                                  printActivityAnalysis)
            : computeClassicalActivityOptimization(op, callee, argumentActivity,
                                                   resultActivity, symbolTable);

    if (failed(optimized))
      return failure();

    // debug printing activity
    if (printActivityAnalysis && dataflow)
      printTaggedActivityAnalysis(callee, optimized->taggedActivities);

    if (llvm::equal(argumentActivity, optimized->argumentActivity) &&
        llvm::equal(resultActivity, optimized->resultActivity))
      return failure();

    // Construct new autodiff op based on optimized activities
    SmallVector<Value> in_args;
    auto in_idx = 0;
    for (const auto &[idx, act, nact] :
         llvm::enumerate(argumentActivity, optimized->argumentActivity)) {
      mlir::Value res = op.getInputs()[in_idx];
      if (act == nact) {
        // no change in activity
        in_args.push_back(res);
        ++in_idx;

        if (act == Activity::enzyme_dup || act == Activity::enzyme_dupnoneed) {
          mlir::Value dres = op.getInputs()[in_idx];
          in_args.push_back(dres);
          ++in_idx;
        }
      } else {
        // we can only demote input args to enzyme_const. Anything else should
        // return in an error
        if (nact != Activity::enzyme_const)
          return failure();
        switch (act) {
        case Activity::enzyme_active:
          in_args.push_back(res);
          ++in_idx;
          break;
        case Activity::enzyme_dup:
        case Activity::enzyme_dupnoneed:
          in_args.push_back(res);
          // we skip over the derivative
          in_idx += 2;
          break;
        default:
          llvm_unreachable(
              "unexpected demotion for input activity in reverse mode");
        }
      }
    }

    // If the autodiff operation is non differentiable then we just stop any
    // more transforms
    if (in_idx == op.getInputs().size())
      return failure();

    SmallVector<Value> out_args;
    SmallVector<Type> out_ty;
    auto out_idx = 0;
    for (const auto &[idx, act, nact] :
         llvm::enumerate(resultActivity, optimized->resultActivity)) {
      // we do not index into any output if the return activity is constnoneed
      if (act == Activity::enzyme_constnoneed) {
        if (nact != act)
          return op->emitError() << "unknown activity demotion";
        continue;
      }

      if (act == nact) {
        // check if this retun value has an active differential return which is
        // an input to the autodiff op
        if (act == Activity::enzyme_active ||
            act == Activity::enzyme_activenoneed) {
          mlir::Value diffret = op.getInputs()[in_idx];
          in_args.push_back(diffret);
          ++in_idx;
        }

        // we do not index into the output if the return activity is
        // activenoneed
        if (act == Activity::enzyme_activenoneed) {
          continue;
        }

        // Reverse mode only returns primal values for return activities.
        // Return cotangents are inputs for active/activenoneed returns, not
        // shadow results for dup/dupnoneed returns.
        switch (act) {
        case Activity::enzyme_active:
        case Activity::enzyme_dup:
        case Activity::enzyme_const: {
          mlir::Value res = op.getOutputs()[out_idx];
          out_args.push_back(res);
          out_ty.push_back(res.getType());
          ++out_idx;
          break;
        }

        case Activity::enzyme_dupnoneed: {
          break;
        }

        default:
          llvm_unreachable("unknown reverse mode activity");
        }

      } else {
        // NOTE:handle return activity demotions
        // 1.activenoneed->constnoneed (in which case we will not have a diffret
        // input for in_args)
        // 2. active -> const, in which case we will again not have diffret
        // input, but we retain the output args
        // 3. dup -> const
        // 4. dupnn -> constnn
        // act cannot be const or constnn

        bool validDemotion =
            (act == Activity::enzyme_active &&
             nact == Activity::enzyme_const) ||
            (act == Activity::enzyme_activenoneed &&
             nact == Activity::enzyme_constnoneed) ||
            (act == Activity::enzyme_dup && nact == Activity::enzyme_const) ||
            (act == Activity::enzyme_dupnoneed &&
             nact == Activity::enzyme_constnoneed);
        if (!validDemotion)
          return failure();

        if (act == Activity::enzyme_activenoneed &&
            nact == Activity::enzyme_constnoneed) {
          ++in_idx;
          continue;
        }

        switch (act) {
        case Activity::enzyme_active: {
          ++in_idx; // skip diffret for active arg in inputs
          mlir::Value res = op.getOutputs()[out_idx];
          out_args.push_back(res);
          out_ty.push_back(res.getType());
          ++out_idx;
          break;
        }
        case Activity::enzyme_dup: {
          mlir::Value res = op.getOutputs()[out_idx];
          out_args.push_back(res);
          out_ty.push_back(res.getType());
          ++out_idx;
          break;
        }
        case Activity::enzyme_dupnoneed: {
          break;
        }

        default:
          llvm_unreachable("unknown return activity demotion");
        }
      }
    }

    // handle differential returns
    for (const auto &[idx, act, nact] :
         llvm::enumerate(argumentActivity, optimized->argumentActivity)) {
      if (act == nact) {
        if (act == Activity::enzyme_active) {
          mlir::Value res = op.getOutputs()[out_idx];
          out_args.push_back(res);
          out_ty.push_back(res.getType());
          ++out_idx;
        }
      } else {
        // we skip over the output if active has been demoted to const
        if (act == Activity::enzyme_active && nact == Activity::enzyme_const) {
          ++out_idx;
        }
      }
    }

    ArrayAttr newInActivity = ArrayAttr::get(
        ctx, llvm::map_to_vector(optimized->argumentActivity,
                                 [&](enzyme::Activity act) -> Attribute {
                                   return ActivityAttr::get(ctx, act);
                                 }));

    ArrayAttr newOutActivity = ArrayAttr::get(
        ctx, llvm::map_to_vector(optimized->resultActivity,
                                 [&](enzyme::Activity act) -> Attribute {
                                   return ActivityAttr::get(ctx, act);
                                 }));

    AutoDiffOp newOp = AutoDiffOp::create(
        rewriter, op->getLoc(), out_ty, op.getFnAttr(), in_args, newInActivity,
        newOutActivity, op.getWidthAttr(), op.getStrongZeroAttr());

    // Map old uses of op to newOp
    auto oldIdx = 0;
    auto newIdx = 0;
    for (auto [idx, old_val, new_val] :
         llvm::enumerate(resultActivity, optimized->resultActivity)) {

      if (old_val == new_val) {
        // These return activities have no reverse-mode op result.
        if (old_val == Activity::enzyme_constnoneed ||
            old_val == Activity::enzyme_activenoneed ||
            old_val == Activity::enzyme_dupnoneed) {
          continue;
        }
        // replace use
        op.getOutputs()[oldIdx++].replaceAllUsesWith(
            newOp.getOutputs()[newIdx++]);
      } else {
        // handle all substitutions
        if (new_val == Activity::enzyme_constnoneed &&
            old_val == Activity::enzyme_dupnoneed) {
          continue;
        } else if (new_val == Activity::enzyme_const &&
                   old_val == Activity::enzyme_dup) {
          op.getOutputs()[oldIdx++].replaceAllUsesWith(
              newOp.getOutputs()[newIdx++]);
        } else if (old_val == Activity::enzyme_active &&
                   new_val == Activity::enzyme_const) {
          op.getOutputs()[oldIdx++].replaceAllUsesWith(
              newOp.getOutputs()[newIdx++]);
        } else if (old_val == Activity::enzyme_activenoneed &&
                   new_val == Activity::enzyme_constnoneed) {
          continue;
        } else {
          llvm_unreachable("unexpected demotion for return activity");
        }
      }
    }

    // only replace active differential returns
    for (auto [idx, old_val, new_val] :
         llvm::enumerate(argumentActivity, optimized->argumentActivity)) {
      if (old_val == new_val) {
        if (old_val == Activity::enzyme_active) {
          op.getOutputs()[oldIdx++].replaceAllUsesWith(
              newOp.getOutputs()[newIdx++]);
        }
      } else {
        if (old_val == Activity::enzyme_active &&
            new_val == Activity::enzyme_const) {
          ++oldIdx; // skip diffret
        }
      }
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  bool dataflow;
  bool printActivityAnalysis;
};

struct EnzymeActivityOptPass
    : public enzyme::impl::EnzymeActivityOptBase<EnzymeActivityOptPass> {
  using EnzymeActivityOptBase::EnzymeActivityOptBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MinimizeForwardActivity, MinimizeReverseActivity>(
        &getContext(), dataflow, printActivityAnalysis);

    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

} // namespace
