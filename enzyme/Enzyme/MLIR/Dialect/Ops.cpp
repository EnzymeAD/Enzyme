//===- EnzymeOps.cpp - Enzyme dialect ops -----------------------*- C++ -*-===//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Ops.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IntegerSet.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace enzyme;
using namespace mlir::arith;

//===----------------------------------------------------------------------===//
// InitOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<MemorySlot> InitOp::getPromotableSlots() {
  auto Ty = this->getType();
  if (isa<CacheType>(Ty))
    return {};

  if (!getOperation()->getBlock()->isEntryBlock())
    return {};

  auto gTy = cast<GradientType>(Ty);
  MemorySlot slot = {this->getResult(), gTy.getBasetype()};

  return {slot};
}

Value InitOp::getDefaultValue(const MemorySlot &slot, OpBuilder &builder) {
  auto gTy = cast<GradientType>(this->getType());
  return cast<AutoDiffTypeInterface>(gTy.getBasetype())
      .createNullValue(builder, this->getLoc());
}

void InitOp::handleBlockArgument(const MemorySlot &slot, BlockArgument argument,
                                 OpBuilder &builder) {}

std::optional<mlir::PromotableAllocationOpInterface>
InitOp::handlePromotionComplete(const MemorySlot &slot, Value defaultValue,
                                OpBuilder &builder) {
  if (defaultValue && defaultValue.use_empty())
    defaultValue.getDefiningOp()->erase();
  this->erase();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// GetOp
//===----------------------------------------------------------------------===//

bool GetOp::loadsFrom(const MemorySlot &slot) {
  return this->getGradient() == slot.ptr;
}

bool GetOp::storesTo(const MemorySlot &slot) { return false; }

Value GetOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                       Value reachingDef, const DataLayout &dataLayout) {
  return {};
}

bool GetOp::canUsesBeRemoved(
    const MemorySlot &slot,
    const llvm::SmallPtrSetImpl<OpOperand *> &blockingUses,
    llvm::SmallVectorImpl<OpOperand *> &newBlockingUses,
    const mlir::DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;

  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getGradient() == slot.ptr;
}

DeletionKind GetOp::removeBlockingUses(
    const MemorySlot &slot,
    const llvm::SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder,
    Value reachingDefinition, const DataLayout &dataLayout) {
  this->getResult().replaceAllUsesWith(reachingDefinition);
  return DeletionKind::Delete;
}

llvm::LogicalResult GetOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, llvm::SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return success(slot.ptr == getGradient());
}

//===----------------------------------------------------------------------===//
// SetOp
//===----------------------------------------------------------------------===//

bool SetOp::loadsFrom(const MemorySlot &slot) { return false; }

bool SetOp::storesTo(const MemorySlot &slot) {
  return this->getGradient() == slot.ptr;
}

Value SetOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                       Value reachingDef, const DataLayout &dataLayout) {
  return this->getValue();
}

bool SetOp::canUsesBeRemoved(
    const MemorySlot &slot,
    const llvm::SmallPtrSetImpl<OpOperand *> &blockingUses,
    llvm::SmallVectorImpl<OpOperand *> &newBlockingUses,
    const mlir::DataLayout &dataLayout) {
  return true;
}

DeletionKind SetOp::removeBlockingUses(
    const MemorySlot &slot,
    const llvm::SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder,
    Value reachingDefinition, const DataLayout &dataLayout) {
  return DeletionKind::Delete;
}

llvm::LogicalResult SetOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, llvm::SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return success(slot.ptr == getGradient());
}

//===----------------------------------------------------------------------===//
// GetFuncOp
//===----------------------------------------------------------------------===//

LogicalResult
ForwardDiffOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // func.func op.
  auto global =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

//===----------------------------------------------------------------------===//
// ForwardDiffOp
//===----------------------------------------------------------------------===//

// Helper: check if any input is mutable.
static inline bool isMutable(Type type) {
  if (isa<mlir::MemRefType>(type) || isa<mlir::UnrankedMemRefType>(type) ||
      isa<mlir::LLVM::LLVMPointerType>(type)) {
    return true;
  }

  return false;
}

/**
 *
 * Modifies input activites for the FwdDiffOp
 * The activity promotion flow is as follows
 * (depending on variable use):
 *
 *           -----> enzyme_dupnoneed
 *          /              /
 * enzyme_dup            /
 *          \          v
 *           ------> enzyme_const
 *
 */
class FwdInpOpt final : public OpRewritePattern<ForwardDiffOp> {
public:
  using OpRewritePattern<ForwardDiffOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForwardDiffOp uop,
                                PatternRewriter &rewriter) const override {

    if (uop.getOutputs().size() == 0)
      return failure();

    auto retActivity = uop.getRetActivity();
    auto inActivity = uop.getActivity();

    auto in_idx = 0;
    SmallVector<mlir::Value, 2> in_args;
    SmallVector<ActivityAttr, 2> newInActivityArgs;
    bool changed = false;
    for (auto [idx, act] : llvm::enumerate(inActivity)) {
      auto iattr = cast<ActivityAttr>(act);
      auto val = iattr.getValue();

      // Forward mode Input activities can only take values {dup, dupnoneed,
      // const }

      mlir::Value inp = uop.getInputs()[in_idx];

      switch (val) {

      case mlir::enzyme::Activity::enzyme_const:
        in_args.push_back(inp);
        newInActivityArgs.push_back(iattr);
        break;

      case Activity::enzyme_dupnoneed: {
        // always pass in primal
        in_args.push_back(inp);
        in_idx++;

        // selectively push or skip directional derivative
        inp = uop.getInputs()[in_idx];
        auto ET = inp.getType();
        auto ETintf = dyn_cast<AutoDiffTypeInterface>(ET);

        if (ETintf && !isMutable(ET) && ETintf.isZero(inp).succeeded()) {
          // skip and promote to const
          auto new_const = mlir::enzyme::ActivityAttr::get(
              rewriter.getContext(), mlir::enzyme::Activity::enzyme_const);
          newInActivityArgs.push_back(new_const);
          changed = true;
        } else {
          // push derivative value
          in_args.push_back(inp);
          newInActivityArgs.push_back(iattr);
        }
        break;
      }

      case Activity::enzyme_dup: {
        // always pass in primal
        in_args.push_back(inp);
        in_idx++;

        // selectively push or skip directional derivative
        inp = uop.getInputs()[in_idx];
        auto ET = inp.getType();
        auto ETintf = dyn_cast<AutoDiffTypeInterface>(ET);

        if (ETintf && !isMutable(ET) && ETintf.isZero(inp).succeeded()) {
        // skip and promote to const
          auto new_const = mlir::enzyme::ActivityAttr::get(
              rewriter.getContext(), mlir::enzyme::Activity::enzyme_const);
          newInActivityArgs.push_back(new_const);
          changed = true;
        } else {
          // push derivative value
          in_args.push_back(inp);
          newInActivityArgs.push_back(iattr);
        }
        break;
      }
      default:
        llvm_unreachable("unexpected input activity arg");
      }

      in_idx++;
    }

    if (!changed)
      return failure();

    // create the new op
    ArrayAttr newInActivity =
        ArrayAttr::get(rewriter.getContext(),
                       llvm::ArrayRef<Attribute>(newInActivityArgs.begin(),
                                                 newInActivityArgs.end()));
    rewriter.replaceOpWithNewOp<ForwardDiffOp>(
        uop, uop->getResultTypes(), uop.getFnAttr(), in_args, newInActivity,
        uop.getRetActivityAttr(), uop.getWidthAttr(), uop.getStrongZeroAttr());
    return success();
  }
};
/**
 *
 * Modifies return activites for the FwdDiffOp
 * The activity promotion flow is as follows
 * (depending on variable use):
 *
 *           -----> enzyme_dupnoneed ----
 *          /                            \
 * enzyme_dup                             ---> enzyme_constnoneed
 *          \                           /
 *           ------> enzyme_const -----
 *
 */
class FwdRetOpt final : public OpRewritePattern<ForwardDiffOp> {
public:
  using OpRewritePattern<ForwardDiffOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForwardDiffOp uop,
                                PatternRewriter &rewriter) const override {

    if (uop.getOutputs().size() == 0)
      return failure();

    auto retActivity = uop.getRetActivity();
    auto out_idx = 0;
    SmallVector<mlir::Value, 2> outs_args;
    SmallVector<Type, 2> out_ty;
    SmallVector<ActivityAttr, 2> newRetActivityArgs;
    bool changed = false;

    for (auto [idx, act] : llvm::enumerate(retActivity)) {
      auto iattr = cast<ActivityAttr>(act);
      auto val = iattr.getValue();

      // const_noneed does not have a value associated with it
      // so we can't index into outputs.
      if (val == Activity::enzyme_constnoneed) {
        newRetActivityArgs.push_back(iattr);
        continue;
      }

      mlir::Value res = uop.getOutputs()[out_idx];

      switch (val) {
      case Activity::enzyme_active:
        outs_args.push_back(res);
        out_ty.push_back(res.getType());
        newRetActivityArgs.push_back(iattr);
        break;

      case mlir::enzyme::Activity::enzyme_const:
        if (!res.use_empty()) {
          outs_args.push_back(res);
          out_ty.push_back(res.getType());
          newRetActivityArgs.push_back(iattr);
        } else {
          changed = true;
          auto new_constnn = mlir::enzyme::ActivityAttr::get(
              rewriter.getContext(),
              mlir::enzyme::Activity::enzyme_constnoneed);
          newRetActivityArgs.push_back(new_constnn);
        }
        break;
      case Activity::enzyme_dupnoneed:

        if (!res.use_empty()) {
          outs_args.push_back(res);
          out_ty.push_back(res.getType());
          newRetActivityArgs.push_back(iattr);
        } else {
          if (!isMutable(res.getType())) {
            changed = true;
            auto new_constnn = mlir::enzyme::ActivityAttr::get(
                rewriter.getContext(),
                mlir::enzyme::Activity::enzyme_constnoneed);
            newRetActivityArgs.push_back(new_constnn);
          } else {
            outs_args.push_back(res);
            out_ty.push_back(res.getType());
            newRetActivityArgs.push_back(iattr);
          }
        }
        break;
      case Activity::enzyme_constnoneed:
        outs_args.push_back(res);
        out_ty.push_back(res.getType());
        newRetActivityArgs.push_back(iattr);
        break;
      case Activity::enzyme_activenoneed:
        outs_args.push_back(res);
        out_ty.push_back(res.getType());
        newRetActivityArgs.push_back(iattr);
        break;
      case Activity::enzyme_dup: {
        ActivityAttr new_dup = iattr;
        if (!res.use_empty()) {
          outs_args.push_back(res);
          out_ty.push_back(res.getType());
        } else {
          changed = true;
          // discard return, change attr
          new_dup = ActivityAttr::get(rewriter.getContext(),
                                      Activity::enzyme_dupnoneed);
        }

        out_idx++;

        // derivative
        res = uop.getOutputs()[out_idx];
        if (!res.use_empty()) {
          // activity arg doesn't update
          out_ty.push_back(res.getType());
          outs_args.push_back(res);
        } else {
          // no uses, can discard
          if (!isMutable(res.getType())) {
            changed = true;
            // check if primal is used
            if (new_dup.getValue() == Activity::enzyme_dupnoneed) {
              new_dup = ActivityAttr::get(rewriter.getContext(),
                                          Activity::enzyme_constnoneed);
            } else {
              new_dup = ActivityAttr::get(rewriter.getContext(),
                                          Activity::enzyme_const);
            }
          } else {
            out_ty.push_back(res.getType());
            outs_args.push_back(res);
          }
        }
        newRetActivityArgs.push_back(new_dup);
        break;
      }
      default:
        llvm_unreachable("unexpected activity arg");
      }

      out_idx++;
    }

    if (!changed)
      return failure();

    ArrayAttr newRetActivity =
        ArrayAttr::get(rewriter.getContext(),
                       llvm::ArrayRef<Attribute>(newRetActivityArgs.begin(),
                                                 newRetActivityArgs.end()));

    ForwardDiffOp newOp = rewriter.create<ForwardDiffOp>(
        uop.getLoc(), out_ty, uop.getFnAttr(), uop.getInputs(),
        uop.getActivityAttr(), newRetActivity, uop.getWidthAttr(),
        uop.getStrongZeroAttr());

    // Map old uses of uop to newOp
    auto oldIdx = 0;
    auto newIdx = 0;
    for (auto [idx, old_act, new_act] :
         llvm::enumerate(retActivity, newRetActivityArgs)) {

      auto iattr = cast<ActivityAttr>(old_act);
      auto old_val = iattr.getValue();
      auto new_val = new_act.getValue();

      if (old_val == new_val) {
        // don't index into op if its a const_noneed
        if (old_val == Activity::enzyme_constnoneed) {
          continue;
        }
        // replace use
        uop.getOutputs()[oldIdx++].replaceAllUsesWith(
            newOp.getOutputs()[newIdx++]);
        if (old_val == Activity::enzyme_dup) {
          // 2nd replacement for derivative
          uop.getOutputs()[oldIdx++].replaceAllUsesWith(
              newOp.getOutputs()[newIdx++]);
        }
      } else {
        // handle all substitutions
        if (new_val == Activity::enzyme_dupnoneed &&
            old_val == Activity::enzyme_dup) {
          ++oldIdx; // skip primal
          uop.getOutputs()[oldIdx++].replaceAllUsesWith(
              newOp.getOutputs()[newIdx++]);
        } else if (new_val == mlir::enzyme::Activity::enzyme_constnoneed &&
                   old_val == mlir::enzyme::Activity::enzyme_const) {
          ++oldIdx; // skip const
        } else if (new_val == mlir::enzyme::Activity::enzyme_constnoneed &&
                   old_val == mlir::enzyme::Activity::enzyme_dupnoneed) {
          ++oldIdx; // skip gradient too
        } else if (new_val == Activity::enzyme_const &&
                   old_val == Activity::enzyme_dup) {

          uop.getOutputs()[oldIdx++].replaceAllUsesWith(
              newOp.getOutputs()[newIdx++]);
          ++oldIdx; // skip derivative
        } else if (new_val == Activity::enzyme_constnoneed &&
                   old_val == Activity::enzyme_dup) {
          ++oldIdx; // skip primal
          ++oldIdx; // skip derivative
        }
      }
    }

    rewriter.eraseOp(uop);
    return success();
  }
};

void ForwardDiffOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {

  patterns.add<FwdRetOpt, FwdInpOpt>(context);
}

LogicalResult AutoDiffOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // func.func op.
  auto global = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      *this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

LogicalResult BatchOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // func.func op.
  auto global = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      *this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

void BroadcastOp::build(OpBuilder &builder, OperationState &result, Value input,
                        ArrayRef<int64_t> shape) {
  auto shapeAttr = builder.getDenseI64ArrayAttr(shape);
  auto resultTy = input.getType();
  for (auto s : llvm::reverse(shape)) {
    resultTy = cast<AutoDiffTypeInterface>(resultTy).getShadowType(s);
  }
  build(builder, result, resultTy, input, shapeAttr);
}

//===----------------------------------------------------------------------===//
// SampleOp
//===----------------------------------------------------------------------===//

LogicalResult SampleOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // func.func op.
  auto global =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

/**
 *
 * Modifies activities for the AutoDiffOp.
 *
 * This is also a nice place to understand the semantics of the rev-mode
 * autodiff op. At it's core, the reverse mode autodiff takes in a function
 *
 * f: def f (pInput):
 * 	...perform computation
 * 	return pOutput
 *
 * One can assign a very simple function signature to f here:
 * f: pInput -> pOutput
 *
 * When trying to differentiate this function (using autodiff op), Enzyme
 * creates a new function which also takes in the co-tangents of the outputs
 * (dOutput), and computes and returns both the output and the input co-tangent
 * (dInput). This is how the generated autodiff op eventually looks like:
 *
 * def revdiff_f(pInput, dOutput):
 * 	...perform computation to compute pOutput
 * 	...perform computation to compute dInput
 * 	return pOutput, dInput
 *
 * The new function signature now becomes
 * revdiff_f : (pInput', dOutput) -> (pOutput, dInput)
 *
 * I mention pInput' here because it is not exactly the input arguments to the
 * function we are differentiating, for example, if the input argument type is
 * an `enzyme_dup`, we will provide both tht primal value along with the
 * shadow(which we then accumulate into and return). So specifically,
 *
 * pInput' = 	pInput (if the activity is enzyme_active, enzyme_const)
 * 		| pInput, dInput (if the activity is enzyme_dup)
 * 		| dInput (if the activity is enzyme_dupnoneed)
 *
 * Now that we have fixed the codegen semantics, we can go ahead and optimize
 * for both the input return activities based on usage. Possible activity
 * promotion flow for the inputs can be as follows:
 * 1. enzyme_active --> enzyme_const (dInput is never used, so we simply don't
 * compute it)
 * 2. enzyme_activenoneed --> enzyme_constnoneed (It is the noneed equivalent
 * of the previous rule and semantically makes sense, although I can't think of
 * a function where you don't pass in the input but still compute the derivative
 * w.r.t that input)
 *
 * Similarly, one can define a similar activity promotion flow for the outputs:
 * 1. enzyme_active --> enzyme_activenoneed (we do need to pass in dOutput into
 * the function, but we can see that pOutput is never used, so let's just not
 * return it. This has the advantage of triggering some additional DCE inside
 * the generated derivative function)
 * 2. enzyme_const --> enzyme_constnoneed (same as above, but we now simply skip
 * over this output)
 *
 * One other thing to note here is that these optimizations preserve the input
 * function signature, and only modify the number of outputs.
 *
 */
class ReverseRetOpt final : public OpRewritePattern<AutoDiffOp> {
public:
  using OpRewritePattern<AutoDiffOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AutoDiffOp uop,
                                PatternRewriter &rewriter) const override {
    // early return if there are no outputs
    if (uop.getOutputs().size() == 0)
      return failure();

    auto inpActivity = uop.getActivity();
    auto retActivity = uop.getRetActivity();
    auto out_idx = 0;
    SmallVector<mlir::Value, 2> outs_args;
    SmallVector<Type, 2> out_ty;
    SmallVector<ActivityAttr, 2> newInActivityArgs;
    SmallVector<ActivityAttr, 2> newRetActivityArgs;

    bool changed = false;

    // handle pOutput
    for (auto [idx, act] : llvm::enumerate(retActivity)) {

      auto iattr = cast<ActivityAttr>(act);
      auto val = iattr.getValue();

      // skip primal return
      if (val == Activity::enzyme_constnoneed ||
          val == Activity::enzyme_activenoneed ||
          val == Activity::enzyme_dupnoneed) {
        newRetActivityArgs.push_back(iattr);
        continue;
      }

      mlir::Value res = uop.getOutputs()[out_idx];

      switch (val) {
      case Activity::enzyme_active:
        if (!res.use_empty()) {
          outs_args.push_back(res);
          out_ty.push_back(res.getType());
          newRetActivityArgs.push_back(iattr);
        } else {
          changed = true;
          auto new_activenn = ActivityAttr::get(rewriter.getContext(),
                                                Activity::enzyme_activenoneed);
          newRetActivityArgs.push_back(new_activenn);
        }
        break;

      case Activity::enzyme_const:
        if (!res.use_empty()) {
          outs_args.push_back(res);
          out_ty.push_back(res.getType());
          newRetActivityArgs.push_back(iattr);
        } else {
          changed = true;
          auto new_constnn = ActivityAttr::get(rewriter.getContext(),
                                               Activity::enzyme_constnoneed);
          newRetActivityArgs.push_back(new_constnn);
        }
        break;

      case Activity::enzyme_dup:
        // dont do anything here for now
        outs_args.push_back(res);
        out_ty.push_back(res.getType());
        newRetActivityArgs.push_back(iattr);
        break;

      case Activity::enzyme_activenoneed:
      case Activity::enzyme_constnoneed:
      case Activity::enzyme_dupnoneed:
        break;

      default:
        llvm_unreachable("unexpected activity arg");
      }

      ++out_idx;
    }

    // handle dInputs
    for (auto [idx, act] : llvm::enumerate(inpActivity)) {
      auto iattr = cast<ActivityAttr>(act);
      auto val = iattr.getValue();

      if (val == Activity::enzyme_active) {
        mlir::Value res = uop.getOutputs()[out_idx];
        if (!res.use_empty()) {
          out_ty.push_back(res.getType());
          outs_args.push_back(res);
          newInActivityArgs.push_back(iattr);
        } else {
          // TODO: check if we can relax immutability here
          if (!isMutable(res.getType())) {
            changed = true;
            auto new_const = ActivityAttr::get(rewriter.getContext(),
                                               Activity::enzyme_const);
            newInActivityArgs.push_back(new_const);
          } else {
            // noop even if its not used.
            out_ty.push_back(res.getType());
            outs_args.push_back(res);
            newInActivityArgs.push_back(iattr);
          }
        }

        ++out_idx;
      } else if (val == Activity::enzyme_activenoneed) {
        mlir::Value res = uop.getOutputs()[out_idx];
        out_ty.push_back(res.getType());
        outs_args.push_back(res);
        newInActivityArgs.push_back(iattr);
        ++out_idx;
        llvm_unreachable("unsupported arg activenoneed");
      } else {
        newInActivityArgs.push_back(iattr);
      }
    }

    if (!changed)
      return failure();

    ArrayAttr newInActivity =
        ArrayAttr::get(rewriter.getContext(),
                       llvm::ArrayRef<Attribute>(newInActivityArgs.begin(),
                                                 newInActivityArgs.end()));
    ArrayAttr newRetActivity =
        ArrayAttr::get(rewriter.getContext(),
                       llvm::ArrayRef<Attribute>(newRetActivityArgs.begin(),
                                                 newRetActivityArgs.end()));

    AutoDiffOp newOp = rewriter.create<AutoDiffOp>(
        uop.getLoc(), out_ty, uop.getFnAttr(), uop.getInputs(), newInActivity,
        newRetActivity, uop.getWidthAttr(), uop.getStrongZeroAttr());

    // Map old uses of uop to newOp
    auto oldIdx = 0;
    auto newIdx = 0;
    for (auto [idx, old_act, new_act] :
         llvm::enumerate(retActivity, newRetActivityArgs)) {

      auto iattr = cast<ActivityAttr>(old_act);
      auto old_val = iattr.getValue();
      auto new_val = new_act.getValue();

      if (old_val == new_val) {
        // don't index into op if no primal is returned
        if (old_val == Activity::enzyme_constnoneed ||
            old_val == Activity::enzyme_activenoneed ||
            old_val == Activity::enzyme_dupnoneed) {
          continue;
        }
        // replace current Primal
        uop.getOutputs()[oldIdx++].replaceAllUsesWith(
            newOp.getOutputs()[newIdx++]);
      } else {
        // handle all substitutions
        if (new_val == Activity::enzyme_activenoneed &&
            old_val == Activity::enzyme_active) {
          ++oldIdx; // skip active primal
        } else if (new_val == Activity::enzyme_constnoneed &&
                   old_val == Activity::enzyme_const) {
          ++oldIdx; // skip const primal
        }
      }
    }

    for (auto [idx, old_act, new_act] :
         llvm::enumerate(inpActivity, newInActivityArgs)) {
      auto iattr = cast<ActivityAttr>(old_act);
      auto old_val = iattr.getValue();
      auto new_val = new_act.getValue();

      if (old_val == new_val) {
        if (old_val == Activity::enzyme_active ||
            old_val == Activity::enzyme_activenoneed) {
          uop.getOutputs()[oldIdx++].replaceAllUsesWith(
              newOp.getOutputs()[newIdx++]);
        } else {
          continue;
        }
      } else {
        if (old_val == Activity::enzyme_active &&
            new_val == Activity::enzyme_const) {
          oldIdx++; // skip derivative
        }
      }
    }

    rewriter.eraseOp(uop);
    return success();
  }
};

void AutoDiffOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.add<ReverseRetOpt>(context);
}
