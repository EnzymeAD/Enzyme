//===- CoreDialectsAutoDiffImplementations.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains common utilities for the external model implementation of
// the automatic differentiation op interfaces for upstream MLIR dialects.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Interfaces/Utils.h"
#include "Passes/RemovalUtils.h"
#include "Passes/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"

#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::enzyme;

static llvm::cl::opt<bool> EnzymeMLIRFastMath(
    "enzyme-mlir-fast-math", llvm::cl::init(true), llvm::cl::Hidden,
    llvm::cl::desc("Build derivative expressions with fast-math enabled"));

void mlir::enzyme::setDerivativeFastMath(Operation *op) {
  if (!op || !EnzymeMLIRFastMath)
    return;
  if (auto iface = dyn_cast<arith::ArithFastMathInterface>(op))
    op->setAttr(iface.getFastMathAttrName(),
                arith::FastMathFlagsAttr::get(op->getContext(),
                                              arith::FastMathFlags::fast));
  else if (auto iface = dyn_cast<LLVM::FastmathFlagsInterface>(op))
    op->setAttr(iface.getFastmathAttrName(),
                LLVM::FastmathFlagsAttr::get(op->getContext(),
                                             LLVM::FastmathFlags::fast));
}

mlir::TypedAttr mlir::enzyme::getConstantAttr(mlir::Type type,
                                              llvm::StringRef value) {
  using namespace mlir;
  if (value == "0") {
    auto ATI = cast<AutoDiffTypeInterface>(type);
    return cast<TypedAttr>(ATI.createNullAttr());
  }
  if (auto T = dyn_cast<TensorType>(type)) {
    if (auto ET = dyn_cast<FloatType>(T.getElementType())) {
      APFloat values[] = {APFloat(ET.getFloatSemantics(), value)};
      return DenseElementsAttr::get(cast<ShapedType>(type),
                                    ArrayRef<APFloat>(values));
    } else if (auto CET = dyn_cast<ComplexType>(T.getElementType())) {
      auto ET = cast<FloatType>(CET.getElementType());
      mlir::Complex<APFloat> values[] = {
          mlir::Complex<APFloat>(APFloat(ET.getFloatSemantics(), value),
                                 APFloat(ET.getFloatSemantics(), "0"))};
      return DenseElementsAttr::get(cast<ShapedType>(type),
                                    ArrayRef<mlir::Complex<APFloat>>(values));
    } else {
      llvm::errs() << " unsupported eltype: " << T.getElementType()
                   << " of type " << type << "\n";
      llvm_unreachable("unsupported eltype");
    }
  } else if (auto T = cast<FloatType>(type)) {
    APFloat apvalue(T.getFloatSemantics(), value);
    return FloatAttr::get(T, apvalue);
    // NOTE `complex::ConstantOp` doesn't accept `TypedAttr`, only `ArrayAttr`
    // } else if (auto T = cast<ComplexType>(type)) {
    //   auto F = cast<FloatType>(T.getElementType());
    //   return mlir::ArrayAttr::get({
    //     FloatAttr::get(F, APFloat(F.getFloatSemantics(), value)),
    //     FloatAttr::get(F, APFloat(F.getFloatSemantics(), "0"));
    //   });
  } else {
    llvm::errs() << " unsupported type: " << type << "\n";
    llvm_unreachable("unsupported eltype");
  }
}

void mlir::enzyme::detail::branchingForwardHandler(Operation *inst,
                                                   OpBuilder &builder,
                                                   MGradientUtils *gutils) {
  auto newInst = gutils->getNewFromOriginal(inst);

  auto binst = cast<BranchOpInterface>(inst);

  // TODO generalize to cloneWithNewBlockArgs interface
  SmallVector<Value> newVals;

  SmallVector<int32_t> segSizes;
  // Keep non-differentiated, non-forwarded operands. These are the ones ahead
  // of the first operand any successor forwards -- a condition, a switch value.
  // When no successor takes an operand there is no such boundary to find, and
  // every operand the op has is one of these.
  size_t non_forwarded = binst->getNumOperands();
  for (size_t i = 0; i < newInst->getNumSuccessors(); i++) {
    auto ops = binst.getSuccessorOperands(i).getForwardedOperands();
    if (ops.empty())
      continue;
    non_forwarded = ops.getBeginOperandIndex();
    break;
  }

  for (size_t i = 0; i < non_forwarded; i++)
    newVals.push_back(gutils->getNewFromOriginal(binst->getOperand(i)));

  segSizes.push_back(newVals.size());
  for (size_t i = 0; i < newInst->getNumSuccessors(); i++) {
    size_t cur = newVals.size();
    auto ops = binst.getSuccessorOperands(i).getForwardedOperands();
    for (auto &&[idx, op] : llvm::enumerate(ops)) {
      auto arg =
          *binst.getSuccessorBlockArgument(ops.getBeginOperandIndex() + idx);
      newVals.push_back(gutils->getNewFromOriginal(op));
      if (!gutils->isConstantValue(arg)) {
        if (!gutils->isConstantValue(op)) {
          newVals.push_back(gutils->invertPointerM(op, builder));
        } else {
          Type retTy = cast<AutoDiffTypeInterface>(arg.getType())
                           .getShadowType(gutils->width);
          auto toret = cast<AutoDiffTypeInterface>(retTy).createNullValue(
              builder, op.getLoc());
          newVals.push_back(toret);
        }
      }
    }
    cur = newVals.size() - cur;
    segSizes.push_back(cur);
  }

  SmallVector<NamedAttribute> attrs(newInst->getAttrs());
  bool has_cases = false;
  for (auto &attr : attrs) {
    if (attr.getName() == "case_operand_segments") {
      has_cases = true;
    }
  }
  for (auto &attr : attrs) {
    if (attr.getName() == "operandSegmentSizes") {
      if (!has_cases) {
        attr.setValue(builder.getDenseI32ArrayAttr(segSizes));
      } else {
        SmallVector<int32_t> segSlices2(segSizes.begin(), segSizes.begin() + 2);
        segSlices2.push_back(0);
        for (size_t i = 2; i < segSizes.size(); i++)
          segSlices2[2] += segSizes[i];
        attr.setValue(builder.getDenseI32ArrayAttr(segSlices2));
      }
    }
    if (attr.getName() == "case_operand_segments") {
      SmallVector<int32_t> segSlices2(segSizes.begin() + 2, segSizes.end());
      attr.setValue(builder.getDenseI32ArrayAttr(segSlices2));
    }
  }

  gutils->getNewFromOriginal(inst->getBlock())
      ->push_back(newInst->create(newInst->getLoc(), newInst->getName(),
                                  TypeRange(), newVals, attrs,
                                  mlir::PropertyRef(), newInst->getSuccessors(),
                                  newInst->getNumRegions()));
  gutils->erase(newInst);
  return;
}

static bool contains(ArrayRef<int> ar, int v) {
  for (auto a : ar) {
    if (a == v) {
      return true;
    }
  }
  return false;
}

LogicalResult mlir::enzyme::detail::memoryIdentityForwardHandler(
    Operation *orig, OpBuilder &builder, MGradientUtils *gutils,
    ArrayRef<int> storedVals) {
  auto iface = cast<ActivityOpInterface>(orig);

  SmallVector<Value> newOperands;
  newOperands.reserve(orig->getNumOperands());
  SmallVector<bool> inverted(orig->getNumOperands(), false);
  for (OpOperand &operand : orig->getOpOperands()) {
    if (iface.isArgInactive(operand.getOperandNumber())) {
      newOperands.push_back(gutils->getNewFromOriginal(operand.get()));
    } else {
      if (gutils->isConstantValue(operand.get())) {

        if (contains(storedVals, operand.getOperandNumber()) ||
            contains(storedVals, -1)) {
          if (isa<AutoDiffTypeInterface>(operand.get().getType())) {
            // Zero for an immutable value; for a mutable value -- an
            // inactive pointer -- the primal is its own shadow.
            newOperands.push_back(oputils::inactiveStoredValueShadow(
                orig, *gutils, operand.get(), builder));
            continue;
          }
        }
        orig->emitError()
            << "Unsupported constant arg to memory identity forward "
               "handler(opidx="
            << operand.getOperandNumber() << ", op=" << operand.get() << ")\n";
        return failure();
      }
      inverted[newOperands.size()] = true;
      newOperands.push_back(gutils->invertPointerM(operand.get(), builder));
    }
  }

  // Assuming shadows following the originals are fine.
  // TODO: consider extending to have a ShadowableTerminatorOpInterface
  Operation *primal = gutils->getNewFromOriginal(orig);
  SmallVector<Operation *, 1> shadows;
  if (gutils->width == 1) {
    Operation *shadow = builder.clone(*primal);
    shadow->setOperands(newOperands);
    shadows.push_back(shadow);
  } else {
    for (size_t w = 0; w < gutils->width; w++) {
      SmallVector<Value> newOperands2(newOperands);
      for (size_t i = 0; i < newOperands.size(); i++) {
        if (!inverted[i])
          continue;
        newOperands2[i] = enzyme::getExtractValue(
            builder, orig->getLoc(), orig->getOperands()[i].getType(),
            newOperands2[i], w);
      }
      Operation *shadow = builder.clone(*primal);
      shadow->setOperands(newOperands2);
      shadows.push_back(shadow);
    }
  }
  for (auto &&[i, oval] : llvm::enumerate(orig->getResults())) {
    if (gutils->isConstantValue(oval))
      continue;
    Value sval;
    if (gutils->width == 1) {
      sval = shadows[0]->getResult(i);
    } else {
      SmallVector<Value> shadowRes;
      for (auto s : shadows) {
        shadowRes.push_back(s->getResult(i));
      }
      sval = enzyme::getConcatValue(builder, orig->getLoc(), shadowRes);
    }
    gutils->setDiffe(oval, sval, builder);
  }

  // A store into memory whose primal contents the caller declared unneeded
  // (enzyme_dupnoneed) need not happen: the shadow store above is the whole
  // of the derivative.
  if (auto store = dyn_cast<enzyme::StoreLikeInterface>(orig))
    if (gutils->primalStoreElidable(store.getStoredPointer()))
      gutils->erase(primal);

  return success();
}

LogicalResult mlir::enzyme::detail::allocationForwardHandler(
    Operation *orig, OpBuilder &builder, MGradientUtils *gutils, bool zero) {

  Operation *primal = gutils->getNewFromOriginal(orig);
  Operation *shadow = builder.clone(*primal);

  Value shadowRes = shadow->getResult(0);

  gutils->setInvertedPointer(orig->getResult(0), shadowRes);
  gutils->eraseIfUnused(orig);

  if (zero) {
    // Fill with zeros
    if (auto iface = dyn_cast<AutoDiffTypeInterface>(shadowRes.getType())) {
      return iface.zeroInPlace(builder, orig->getLoc(), shadowRes);
    } else {
      orig->emitError() << "Type " << shadowRes.getType()
                        << " does not implement "
                           "AutoDiffTypeInterface";
      return failure();
    }
  }
  return success();
}

void mlir::enzyme::detail::returnReverseHandler(Operation *op,
                                                OpBuilder &builder,
                                                MGradientUtilsReverse *gutils) {
  size_t num_out = 0;
  for (auto act : gutils->RetDiffeTypes) {
    if (act == DIFFE_TYPE::OUT_DIFF)
      num_out++;
  }

  size_t idx = 0;
  auto args = gutils->newFunc->getRegions().begin()->begin()->getArguments();

  for (auto &&[op, act] : llvm::zip(op->getOperands(), gutils->RetDiffeTypes)) {
    if (act == DIFFE_TYPE::OUT_DIFF) {
      if (!gutils->isConstantValue(op)) {
        auto d_out = args[args.size() - num_out + idx];
        gutils->addToDiffe(op, d_out, builder);
      }
      idx++;
    }
  }
}

void mlir::enzyme::detail::regionTerminatorForwardHandler(
    Operation *origTerminator, OpBuilder &builder, MGradientUtils *gutils) {
  auto parentOp = origTerminator->getParentOp();

  llvm::SmallDenseSet<unsigned> operandsToShadow;
  auto termIface = dyn_cast<RegionBranchTerminatorOpInterface>(origTerminator);
  auto regionBranchOp =
      dyn_cast<RegionBranchOpInterface>(origTerminator->getParentOp());
  if (termIface && regionBranchOp) {

    SmallVector<RegionSuccessor> successors;
    termIface.getSuccessorRegions(
        SmallVector<Attribute>(origTerminator->getNumOperands(), Attribute()),
        successors);

    for (auto &successor : successors) {
      OperandRange operandRange = termIface.getSuccessorOperands(successor);
      ValueRange targetValues =
          successor.isOperation()
              ? parentOp->getResults()
              : regionBranchOp.getSuccessorInputs(successor);
      // The parent may carry more results than the terminator forwards to it
      // (a gpu wrapper's token-like result has no yield operand); only the
      // paired prefix can be shadowed.
      size_t numPaired = std::min(operandRange.size(), targetValues.size());
      for (auto &&[i, target] :
           llvm::enumerate(targetValues.take_front(numPaired))) {
        if (!gutils->isConstantValue(target))
          operandsToShadow.insert(operandRange.getBeginOperandIndex() + i);
      }
    }
  } else {
    // The parent may carry more results than its terminator forwards (a gpu
    // wrapper's token-like result has no yield operand); only the paired
    // results can be shadowed.
    size_t numPaired = std::min<size_t>(parentOp->getNumResults(),
                                        origTerminator->getNumOperands());
    for (auto res : parentOp->getResults().take_front(numPaired)) {
      if (!gutils->isConstantValue(res))
        operandsToShadow.insert(res.getResultNumber());
    }
  }

  SmallVector<Value> newOperands;
  newOperands.reserve(origTerminator->getNumOperands() +
                      operandsToShadow.size());
  for (OpOperand &operand : origTerminator->getOpOperands()) {
    newOperands.push_back(gutils->getNewFromOriginal(operand.get()));
    if (operandsToShadow.contains(operand.getOperandNumber()))
      newOperands.push_back(gutils->invertPointerM(operand.get(), builder));
  }

  // Assuming shadows following the originals are fine.
  // TODO: consider extending to have a ShadowableTerminatorOpInterface
  Operation *replTerminator = gutils->getNewFromOriginal(origTerminator);
  replTerminator->setOperands(newOperands);
}

LogicalResult mlir::enzyme::detail::controlFlowForwardHandler(
    Operation *op, OpBuilder &builder, MGradientUtils *gutils) {

  // For all operands that are forwarded to the body, if they are active, also
  // add the shadow as operand.
  auto regionBranchOp = dyn_cast<RegionBranchOpInterface>(op);
  if (!regionBranchOp) {
    op->emitError() << " RegionBranchOpInterface not implemented for " << *op
                    << "\n";
    return failure();
  }
  auto iface = dyn_cast<ControlFlowAutoDiffOpInterface>(op);
  if (!iface) {
    op->emitError() << " ControlFlowAutoDiffOpInterface not implemented for "
                    << *op << "\n";
    return failure();
  }

  // TODO: we may need to record, for every successor, which of its inputs
  // need a shadow to recreate the body correctly.
  llvm::SmallDenseSet<unsigned> operandPositionsToShadow;
  llvm::SmallDenseSet<unsigned> resultPositionsToShadow;

  SmallVector<RegionSuccessor> entrySuccessors;
  regionBranchOp.getEntrySuccessorRegions(
      SmallVector<Attribute>(op->getNumOperands(), Attribute()),
      entrySuccessors);

  for (const RegionSuccessor &successor : entrySuccessors) {

    OperandRange operandRange =
        iface.getSuccessorOperands(regionBranchOp, successor);

    ValueRange targetValues =
        successor.isOperation() ? op->getResults()
                                : regionBranchOp.getSuccessorInputs(successor);

    // Need to know which of the arguments are being forwarded to from
    // operands.
    for (auto &&[i, regionValue, operand] :
         llvm::enumerate(targetValues, operandRange)) {
      if (gutils->isConstantValue(regionValue))
        continue;
      operandPositionsToShadow.insert(operandRange.getBeginOperandIndex() + i);
      if (successor.isOperation())
        resultPositionsToShadow.insert(i);
    }
  }

  for (auto res : op->getResults())
    if (!gutils->isConstantValue(res))
      resultPositionsToShadow.insert(res.getResultNumber());

  return controlFlowForwardHandler(
      op, builder, gutils, operandPositionsToShadow, resultPositionsToShadow);
}

LogicalResult mlir::enzyme::detail::controlFlowForwardHandler(
    Operation *op, OpBuilder &builder, MGradientUtils *gutils,
    const llvm::SmallDenseSet<unsigned> &operandPositionsToShadow,
    const llvm::SmallDenseSet<unsigned> &resultPositionsToShadow) {
  // For all active results, add shadow types.
  // For now, assuming all results are relevant.
  Operation *newOp = gutils->getNewFromOriginal(op);
  SmallVector<Type> newOpResultTypes;
  newOpResultTypes.reserve(op->getNumResults() * 2);
  for (auto result : op->getResults()) {
    // TODO only if used (can we DCE the primal after having done the
    // derivative).
    newOpResultTypes.push_back(result.getType());
    if (!gutils->isConstantValue(result)) {
      assert(resultPositionsToShadow.count(result.getResultNumber()));
    }
    if (!resultPositionsToShadow.count(result.getResultNumber()))
      continue;
    auto typeIface = dyn_cast<AutoDiffTypeInterface>(result.getType());
    if (!typeIface) {
      op->emitError() << " AutoDiffTypeInterface not implemented for "
                      << result.getType() << "\n";
      return failure();
    }
    newOpResultTypes.push_back(typeIface.getShadowType(gutils->width));
  }

  SmallVector<Value> newOperands;
  newOperands.reserve(op->getNumOperands() + operandPositionsToShadow.size());
  for (OpOperand &operand : op->getOpOperands()) {
    newOperands.push_back(gutils->getNewFromOriginal(operand.get()));
    if (operandPositionsToShadow.contains(operand.getOperandNumber()))
      newOperands.push_back(gutils->invertPointerM(operand.get(), builder));
  }
  // We are assuming the op can forward additional operands, listed
  // immediately after the original operands, to the same regions.
  // ^^
  // Our interface guarantees this.
  // We also assume that the region-holding op returns all of the values
  // yielded by terminators, and only those values.

  auto iface = dyn_cast<ControlFlowAutoDiffOpInterface>(op);
  if (!iface) {
    op->emitError() << " ControlFlowAutoDiffOpInterface not implemented for "
                    << *op << "\n";
    return failure();
  }
  Operation *replacement = iface.createWithShadows(
      builder, gutils, op, newOperands, newOpResultTypes);
  assert(replacement->getNumResults() == newOpResultTypes.size());
  for (auto &&[region, replacementRegion] :
       llvm::zip(newOp->getRegions(), replacement->getRegions())) {
    replacementRegion.takeBody(region);
  }

  // Inject the mapping for the new results into GradientUtil's shadow
  // table.
  SmallVector<Value> reps;
  size_t idx = 0;
  for (OpResult r : op->getResults()) {
    // TODO only if used
    reps.push_back(replacement->getResult(idx));
    idx++;
    if (!gutils->isConstantValue(r)) {
      assert(resultPositionsToShadow.count(r.getResultNumber()));
      auto inverted = gutils->invertedPointers.lookupOrNull(r);
      assert(inverted);
      gutils->invertedPointers.map(r, replacement->getResult(idx));
      inverted.replaceAllUsesWith(replacement->getResult(idx));
      gutils->erase(inverted.getDefiningOp());
      idx++;
    } else if (resultPositionsToShadow.count(r.getResultNumber())) {
      idx++;
    }
  }

  // Differentiate body.
  for (auto &origRegion : op->getRegions()) {
    for (auto &origBlock : origRegion) {
      for (Operation &o : origBlock) {
        if (failed(gutils->visitChild(&o))) {
          return failure();
        }
      }
    }
  }

  // Replace all uses of original results
  gutils->replaceOrigOpWith(op, reps);
  gutils->erase(newOp);
  gutils->originalToNewFnOps[op] = replacement;

  return success();
}

namespace edetail = mlir::enzyme::detail;

// The callee of `op`, or null where it is not a direct call to something this
// can see the body of.
static FunctionOpInterface getDirectCallee(Operation *op) {
  auto callOp = dyn_cast<CallOpInterface>(op);
  if (!callOp)
    return nullptr;
  auto sym = dyn_cast<SymbolRefAttr>(callOp.getCallableForCallee());
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<FunctionOpInterface>(
      SymbolTable::lookupNearestSymbolFrom(op, sym));
}

LogicalResult edetail::callForwardHandler(Operation *orig, OpBuilder &builder,
                                          MGradientUtils *gutils) {
  DerivativeMode mode = DerivativeMode::ForwardMode;

  auto fn = getDirectCallee(orig);
  if (!fn) {
    return orig->emitError()
           << "could not find the callee of: " << *orig << "\n";
  }
  if (fn.getFunctionBody().empty()) {
    return orig->emitError()
           << "cannot differentiate a call to a function without a body and "
              "without a registered derivative: "
           << fn.getNameAttr() << "\n";
  }

  auto narg = orig->getNumOperands();
  auto nret = orig->getNumResults();

  std::vector<DIFFE_TYPE> RetActivity;
  RetActivity.reserve(nret);
  for (auto res : orig->getResults()) {
    RetActivity.push_back(gutils->isConstantValue(res) ? DIFFE_TYPE::CONSTANT
                                                       : DIFFE_TYPE::DUP_ARG);
  }

  std::vector<DIFFE_TYPE> ArgActivity;
  ArgActivity.reserve(narg);
  for (auto arg : orig->getOperands()) {
    if (gutils->isConstantValue(arg)) {
      ArgActivity.push_back(DIFFE_TYPE::CONSTANT);
      continue;
    }
    // A pointer whose base the caller declared enzyme_dupnoneed keeps that
    // declaration through the call: the callee is where the stores live,
    // and it can only skip their primal halves if it is told.
    ArgActivity.push_back(gutils->getDiffeTypeOfBase(arg) ==
                                  DIFFE_TYPE::DUP_NONEED
                              ? DIFFE_TYPE::DUP_NONEED
                              : DIFFE_TYPE::DUP_ARG);
  }

  std::vector<bool> returnPrimal(nret, true);
  std::vector<bool> returnShadow(nret, false);

  auto type_args = gutils->TA.getAnalyzedTypeInfo(fn);

  bool freeMemory = true;
  size_t width = gutils->width;

  std::vector<bool> overwritten_args(narg, false);

  auto forwardFn = gutils->Logic.CreateForwardDiff(
      fn, RetActivity, ArgActivity, gutils->TA, returnPrimal, mode, freeMemory,
      width,
      /* addedType */ nullptr, type_args, overwritten_args,
      /* augmented */ nullptr, gutils->omp, gutils->postpasses,
      gutils->verifyPostPasses, gutils->strongZero);

  if (!forwardFn)
    return orig->emitError()
           << "failed to create forward-mode derivative for callee "
           << fn.getNameAttr() << "\n";

  SmallVector<Value> fwdArguments;

  for (auto &&[arg, act] : llvm::zip_equal(orig->getOperands(), ArgActivity)) {
    fwdArguments.push_back(gutils->getNewFromOriginal(arg));
    if (act == DIFFE_TYPE::DUP_ARG || act == DIFFE_TYPE::DUP_NONEED)
      fwdArguments.push_back(gutils->invertPointerM(arg, builder));
  }

  auto *fwdCallOp = cast<AutoDiffFunctionInterface>(forwardFn.getOperation())
                        .createCall(builder, orig->getLoc(), fwdArguments);

  SmallVector<Value> primals;
  primals.reserve(nret);

  int fwdIndex = 0;
  for (auto &&[ret, act] : llvm::zip_equal(orig->getResults(), RetActivity)) {
    auto fwdRet = fwdCallOp->getResult(fwdIndex);
    primals.push_back(fwdRet);

    fwdIndex++;

    if (act == DIFFE_TYPE::DUP_ARG) {
      gutils->setDiffe(ret, fwdCallOp->getResult(fwdIndex), builder);
      fwdIndex++;
    }
  }

  auto newOp = gutils->getNewFromOriginal(orig);
  gutils->replaceOrigOpWith(orig, primals);
  gutils->erase(newOp);

  return success();
}

// The activities of a call's operands and results, as the caller's activity
// analysis sees them. A custom reverse rule spells its argument activities as
// enzyme_dup / enzyme_active / enzyme_const, so dupnoneed is not distinguished
// here -- a dupnoneed pointer is passed as plain dup, which is conservative.
static void getCallActivity(Operation *orig, MGradientUtilsReverse *gutils,
                            std::vector<DIFFE_TYPE> &ArgActivity,
                            std::vector<DIFFE_TYPE> &RetActivity) {
  auto activityOf = [&](Value v) {
    return gutils->isConstantValue(v) ? DIFFE_TYPE::CONSTANT
           : cast<AutoDiffTypeInterface>(v.getType()).isMutable()
               ? DIFFE_TYPE::DUP_ARG
               : DIFFE_TYPE::OUT_DIFF;
  };
  for (auto arg : orig->getOperands())
    ArgActivity.push_back(activityOf(arg));
  for (auto res : orig->getResults())
    RetActivity.push_back(activityOf(res));
}

static Activity diffeTypeToActivity(DIFFE_TYPE act) {
  switch (act) {
  case DIFFE_TYPE::CONSTANT:
    return Activity::enzyme_const;
  case DIFFE_TYPE::OUT_DIFF:
    return Activity::enzyme_active;
  case DIFFE_TYPE::DUP_ARG:
    return Activity::enzyme_dup;
  case DIFFE_TYPE::DUP_NONEED:
    return Activity::enzyme_dupnoneed;
  default:
    llvm_unreachable("cannot handle act");
  }
}

static bool isActiveActivity(Attribute attr) {
  auto act = cast<ActivityAttr>(attr).getValue();
  return act == Activity::enzyme_active || act == Activity::enzyme_activenoneed;
}

// A callee may carry hand-written reverse rules in `enzyme.custom_rule`: one
// rule, or a rule set of one rule per activity pattern. The rule used for a
// call is the one that serves its activities (see
// CustomReverseRuleOp::activityMatch) while differentiating the fewest
// arguments and results, so no unused cotangent is computed when a more
// specific rule exists; ties go to the rule listed first. Null when the callee
// names no rule that serves the call.
static CustomReverseRuleOp selectCustomRule(FunctionOpInterface fn,
                                            ArrayRef<DIFFE_TYPE> ArgActivity,
                                            ArrayRef<DIFFE_TYPE> RetActivity) {
  auto rules = lookupCustomReverseRules(fn, "enzyme.custom_rule");
  if (failed(rules))
    return nullptr;

  SmallVector<Activity> ArgAct =
      llvm::map_to_vector(ArgActivity, diffeTypeToActivity);
  SmallVector<Activity> RetAct =
      llvm::map_to_vector(RetActivity, diffeTypeToActivity);

  CustomReverseRuleOp best = nullptr;
  for (auto rule : *rules) {
    if (failed(rule.activityMatch(ArgAct, RetAct)))
      continue;
    if (!best || rule.numDifferentiated() < best.numDifferentiated())
      best = rule;
  }
  return best;
}

// Split mode builds the callee's augmented primal and reverse out of a custom
// reverse rule, whose signatures are spelled as builtin FunctionTypes (see
// CreateSplitModeDiff). An llvm.func spells its signature as an
// LLVMFunctionType, so a call to one still takes the combined-mode path
// below.
static bool splitModeSupported(FunctionOpInterface fn) {
  return isa<FunctionType>(fn.getFunctionType());
}

// A call to a func-like callee is differentiated in split mode: the primal
// becomes a call to the callee's augmented primal, which also returns the
// values the rule caches, and the adjoint becomes a call to the callee's
// reverse, which takes those values back. The caller caches them like any
// other value, so a call inside a loop or a branch needs nothing special.
// Nothing the callee overwrote has to be reconstructed by the caller --
// whatever the reverse needs was cached while the primal ran.
static LogicalResult callReverseHandlerSplit(Operation *orig,
                                             OpBuilder &builder,
                                             MGradientUtilsReverse *gutils,
                                             SmallVector<Value> caches,
                                             FunctionOpInterface fn) {
  if (gutils->width != 1)
    return orig->emitError() << "custom reverse rules only support width 1";

  std::vector<DIFFE_TYPE> ArgActivity, RetActivity;
  getCallActivity(orig, gutils, ArgActivity, RetActivity);

  if (llvm::any_of(RetActivity,
                   [&](auto act) { return act == DIFFE_TYPE::DUP_ARG; })) {
    return orig->emitError()
           << "could not emit adjoint with mutable return types in: " << *orig
           << "\n";
  }

  // The augmented primal was created in cacheValues, naming the rule chosen
  // or derived there, or a placeholder if there was none.
  auto primalCall =
      cast<CallAugmentedPrimalOp>(gutils->getNewFromOriginal(orig));
  auto cr = SymbolTable::lookupNearestSymbolFrom<CustomReverseRuleOp>(
      orig, primalCall.getFnAttr());

  if (!cr) {
    auto authored = lookupCustomReverseRules(fn, "enzyme.custom_rule");
    if (failed(authored))
      return orig->emitError() << "enzyme.custom_rule of " << fn.getNameAttr()
                               << " must name enzyme.custom_reverse_rule ops";

    if (!authored->empty()) {
      auto diag = orig->emitError()
                  << "could not find a rule with the right activity";
      for (auto rule : *authored)
        diag << " (rule activity=" << rule.getActivity()
             << ", ret_activity=" << rule.getRetActivity() << ")";
      return diag;
    }

    return orig->emitError()
           << "failed to create reverse-mode adjoint for callee "
           << fn.getNameAttr() << "\n";
  }

  auto crArgActivity = cr.getActivity();
  auto crRetActivity = cr.getRetActivity();

  if (crArgActivity.size() != ArgActivity.size())
    return orig->emitError()
           << "cannot apply custom rule for func " << fn.getNameAttr()
           << " (wrong arg activity size)";

  if (crRetActivity.size() != RetActivity.size())
    return orig->emitError()
           << "cannot apply custom rule to func " << fn.getNameAttr()
           << " (wrong ret activity size)";

  SmallVector<Value> cacheValues;
  for (Value cache : caches)
    cacheValues.push_back(gutils->popCache(cache, builder));

  SmallVector<Value> operands;
  SmallVector<Type> resultTypes;

  // One cotangent per result the rule differentiates. A result the caller
  // holds constant contributes zero.
  for (auto [act, res, crAct] :
       llvm::zip_equal(RetActivity, orig->getResults(), crRetActivity)) {
    if (!isActiveActivity(crAct))
      continue;
    if (act == DIFFE_TYPE::OUT_DIFF)
      operands.push_back(gutils->diffe(res, builder));
    else
      operands.push_back(cast<AutoDiffTypeInterface>(res.getType())
                             .createNullValue(builder, orig->getLoc()));
  }

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(primalCall);

    int operandIndex = 0;
    for (auto [act, operand, crAct] :
         llvm::zip_equal(ArgActivity, orig->getOperands(), crArgActivity)) {
      if (isActiveActivity(crAct)) {
        resultTypes.push_back(cast<AutoDiffTypeInterface>(operand.getType())
                                  .getShadowType(/*width=*/1));
      }
      if (act == DIFFE_TYPE::DUP_ARG) {
        operandIndex++;
        primalCall->insertOperands(operandIndex,
                                   gutils->invertPointerM(operand, builder));
      }
      operandIndex++;
    }
  }

  auto revCall =
      CallCustomReverseOp::create(builder, orig->getLoc(), resultTypes,
                                  cr.getSymName(), operands, cacheValues);

  // The rule returns one cotangent per argument it differentiates. One it
  // computes for an operand the caller holds constant is dropped.
  int didx = 0;
  for (auto [act, operand, crAct] :
       llvm::zip_equal(ArgActivity, orig->getOperands(), crArgActivity)) {
    if (!isActiveActivity(crAct))
      continue;
    if (act == DIFFE_TYPE::OUT_DIFF)
      gutils->addToDiffe(operand, revCall->getResult(didx), builder);
    didx++;
  }

  for (auto [act, res] : llvm::zip_equal(RetActivity, orig->getResults())) {
    if (act == DIFFE_TYPE::OUT_DIFF)
      gutils->zeroDiffe(res, builder);
  }

  return success();
}

// The rule a split-mode call uses, fixed before its primal runs: the caller
// carries the rule's caches as values from the augmented primal to the
// reverse, so their types must be known here. An authored rule that serves
// the call's activities is used; otherwise, if the callee names none, a rule
// is derived from its body (or one derived earlier for the same activity is
// reused). Null if there is no usable rule; the adjoint reports why.
static CustomReverseRuleOp
getSplitModeRule(Operation *orig, MGradientUtilsReverse *gutils,
                 FunctionOpInterface fn, std::vector<DIFFE_TYPE> &ArgActivity,
                 std::vector<DIFFE_TYPE> &RetActivity) {
  if (gutils->width != 1)
    return nullptr;

  CustomReverseRuleOp cr = selectCustomRule(fn, ArgActivity, RetActivity);

  auto authored = lookupCustomReverseRules(fn, "enzyme.custom_rule");
  bool derive = !cr && succeeded(authored) && authored->empty() &&
                llvm::none_of(RetActivity, [](DIFFE_TYPE act) {
                  return act == DIFFE_TYPE::DUP_ARG;
                });
  if (derive) {
    auto narg = orig->getNumOperands();
    auto nret = orig->getNumResults();
    std::vector<bool> overwritten_args(narg, true);
    std::vector<bool> returnShadow(nret, false);
    std::vector<bool> returnPrimal(nret, false);

    auto type_args = gutils->TA.getAnalyzedTypeInfo(fn);

    bool freeMemory = true;
    size_t width = gutils->width;

    auto myCr = gutils->Logic.CreateSplitModeDiff(
        fn, RetActivity, ArgActivity, gutils->TA, returnPrimal, returnShadow,
        DerivativeMode::ReverseModeGradient, freeMemory, width,
        /*addedType*/ nullptr, type_args, overwritten_args,
        /*augmented*/ nullptr, gutils->omp, gutils->postpasses,
        gutils->verifyPostPasses, gutils->strongZero);
    if (myCr)
      cr =
          SymbolTable::lookupNearestSymbolFrom<CustomReverseRuleOp>(orig, myCr);
  }

  if (cr && failed(finalizeCustomReverseRule(cr)))
    return nullptr;
  return cr;
}

static SmallVector<Value> callCacheValuesSplit(Operation *orig,
                                               MGradientUtilsReverse *gutils,
                                               FunctionOpInterface fn) {
  Operation *newOp = gutils->getNewFromOriginal(orig);
  OpBuilder cacheBuilder(newOp);

  std::vector<DIFFE_TYPE> ArgActivity, RetActivity;
  getCallActivity(orig, gutils, ArgActivity, RetActivity);

  auto cr = getSplitModeRule(orig, gutils, fn, ArgActivity, RetActivity);

  FlatSymbolRefAttr fnAttr =
      cr ? FlatSymbolRefAttr::get(cr.getSymNameAttr())
         : FlatSymbolRefAttr::get(orig->getContext(), "<placeholder>");
  SmallVector<Type> cacheTypes;
  if (cr)
    cacheTypes = getCustomReverseRuleCacheTypes(cr);

  SmallVector<Value> operands(newOp->getOperands());

  auto primal = CallAugmentedPrimalOp::create(cacheBuilder, orig->getLoc(),
                                              newOp->getResultTypes(),
                                              cacheTypes, fnAttr, operands);

  for (auto [oldRes, newRes] :
       llvm::zip_equal(newOp->getResults(), primal.getOutputs()))
    oldRes.replaceAllUsesWith(newRes);

  SmallVector<Value> cachedArguments;
  for (Value cache : primal.getCaches())
    cachedArguments.push_back(gutils->initAndPushCache(cache, cacheBuilder));

  gutils->erase(newOp);
  gutils->originalToNewFnOps[orig] = primal;

  return cachedArguments;
}

// Combined-mode reverse recomputes the callee without overwritten-argument
// analysis, so only callees without memory effects are supported.
static bool memoryEffectsNone(LLVM::MemoryEffectsAttr me) {
  return me && me.getArgMem() == LLVM::ModRefInfo::NoModRef &&
         me.getInaccessibleMem() == LLVM::ModRefInfo::NoModRef &&
         me.getOther() == LLVM::ModRefInfo::NoModRef &&
         me.getErrnoMem() == LLVM::ModRefInfo::NoModRef &&
         me.getTargetMem0() == LLVM::ModRefInfo::NoModRef &&
         me.getTargetMem1() == LLVM::ModRefInfo::NoModRef;
}

static bool splitReverseMemoryOkImpl(Operation *orig, FunctionOpInterface fn,
                                     SmallPtrSetImpl<Operation *> &visited) {
  if (auto call = dyn_cast<LLVM::CallOp>(orig))
    if (memoryEffectsNone(call.getMemoryEffectsAttr()))
      return true;
  if (auto llvmFn = dyn_cast<LLVM::LLVMFuncOp>(fn.getOperation())) {
    if (memoryEffectsNone(llvmFn.getMemoryEffectsAttr()))
      return true;
    if (auto pass = llvmFn->getAttrOfType<ArrayAttr>("passthrough"))
      for (Attribute a : pass)
        if (auto s = dyn_cast<StringAttr>(a))
          if (s.getValue() == "readnone")
            return true;
  }
  if (fn.getFunctionBody().empty())
    return false;
  // A cycle contributes no effects beyond those already under check.
  if (!visited.insert(fn.getOperation()).second)
    return true;
  WalkResult res = fn.getFunctionBody().walk([&](Operation *op) {
    if (isMemoryEffectFree(op))
      return WalkResult::advance();
    // A call op answers conservatively for itself -- its effects are its
    // callee's, so ask the callee.
    if (auto callee = getDirectCallee(op))
      if (splitReverseMemoryOkImpl(op, callee, visited))
        return WalkResult::advance();
    return WalkResult::interrupt();
  });
  return !res.wasInterrupted();
}

static bool splitReverseMemoryOk(Operation *orig, FunctionOpInterface fn) {
  SmallPtrSet<Operation *, 8> visited;
  return splitReverseMemoryOkImpl(orig, fn, visited);
}

static LogicalResult checkSplitReverseMemory(Operation *orig,
                                             FunctionOpInterface fn) {
  if (splitReverseMemoryOk(orig, fn))
    return success();
  return orig->emitError()
         << "cannot differentiate a call in reverse mode whose callee "
            "touches memory; caching of overwritten arguments is not yet "
            "implemented here: "
         << fn.getNameAttr() << "\n";
}

static LogicalResult callReverseHandlerCombined(Operation *orig,
                                                OpBuilder &builder,
                                                MGradientUtilsReverse *gutils,
                                                SmallVector<Value> caches,
                                                FunctionOpInterface fn) {
  DerivativeMode mode = DerivativeMode::ReverseModeGradient;

  if (failed(checkSplitReverseMemory(orig, fn)))
    return failure();

  auto narg = orig->getNumOperands();
  auto nret = orig->getNumResults();

  std::vector<DIFFE_TYPE> RetActivity;
  for (auto res : orig->getResults()) {
    RetActivity.push_back(
        gutils->isConstantValue(res) ? DIFFE_TYPE::CONSTANT
        : cast<AutoDiffTypeInterface>(res.getType()).isMutable()
            ? DIFFE_TYPE::DUP_ARG
            : DIFFE_TYPE::OUT_DIFF);
  }

  std::vector<DIFFE_TYPE> ArgActivity;
  for (auto arg : orig->getOperands()) {
    if (gutils->isConstantValue(arg)) {
      ArgActivity.push_back(DIFFE_TYPE::CONSTANT);
      continue;
    }
    if (cast<AutoDiffTypeInterface>(arg.getType()).isMutable()) {
      // A pointer whose base the caller declared enzyme_dupnoneed keeps
      // that declaration through the call: the callee is where the stores
      // live, and it can only skip their primal halves if it is told.
      ArgActivity.push_back(gutils->getDiffeTypeOfBase(arg) ==
                                    DIFFE_TYPE::DUP_NONEED
                                ? DIFFE_TYPE::DUP_NONEED
                                : DIFFE_TYPE::DUP_ARG);
      continue;
    }
    ArgActivity.push_back(DIFFE_TYPE::OUT_DIFF);
  }

  if (llvm::any_of(RetActivity,
                   [&](auto act) { return act == DIFFE_TYPE::DUP_ARG; })) {
    return orig->emitError()
           << "could not emit adjoint with mutable return types in: " << *orig
           << "\n";
  }

  std::vector<bool> overwritten_args(narg, true);
  std::vector<bool> returnShadow(nret, false);
  std::vector<bool> returnPrimal(nret, false);

  auto type_args = gutils->TA.getAnalyzedTypeInfo(fn);

  bool freeMemory = true;
  size_t width = gutils->width;

  auto revFn = gutils->Logic.CreateReverseDiff(
      fn, RetActivity, ArgActivity, gutils->TA, returnPrimal, returnShadow,
      mode, freeMemory, gutils->AtomicAdd, width, /*addedType*/ nullptr,
      type_args, overwritten_args, /*augmented*/ nullptr, gutils->omp,
      gutils->postpasses, gutils->verifyPostPasses, gutils->strongZero,
      /*markReadonly=*/false);

  if (!revFn)
    return orig->emitError()
           << "failed to create reverse-mode adjoint for callee "
           << fn.getNameAttr() << "\n";

  SmallVector<Value> revArguments;

  size_t cacheIdx = 0;
  for (auto [arg, act] : llvm::zip_equal(orig->getOperands(), ArgActivity)) {
    revArguments.push_back(gutils->popCache(caches[cacheIdx++], builder));
    if (act == DIFFE_TYPE::DUP_ARG || act == DIFFE_TYPE::DUP_NONEED)
      revArguments.push_back(gutils->popCache(caches[cacheIdx++], builder));
  }
  assert(cacheIdx == caches.size());

  for (auto result : orig->getResults()) {
    if (gutils->isConstantValue(result))
      continue;
    revArguments.push_back(gutils->diffe(result, builder));
  }

  auto *revCallOp = cast<AutoDiffFunctionInterface>(revFn.getOperation())
                        .createCall(builder, orig->getLoc(), revArguments);

  int revIndex = 0, fwdIndex = 0;
  for (auto [arg, act] : llvm::zip_equal(orig->getOperands(), ArgActivity)) {
    fwdIndex++;

    if (gutils->isConstantValue(arg))
      continue;

    if (act == DIFFE_TYPE::DUP_ARG || act == DIFFE_TYPE::DUP_NONEED) {
      cast<ClonableTypeInterface>(arg.getType())
          .freeClonedValue(builder, revArguments[fwdIndex - 1]);
      fwdIndex++;
    } else {
      auto diffe = revCallOp->getResult(revIndex);
      gutils->addToDiffe(arg, diffe, builder);
      revIndex++;
    }
  }

  return success();
}

static SmallVector<Value> callCacheValuesCombined(Operation *orig,
                                                  MGradientUtilsReverse *gutils,
                                                  FunctionOpInterface fn) {
  SmallVector<Value> cachedArguments;

  // A callee the adjoint will refuse gets nothing cached: the caching of
  // pointer arguments would already need the sizes and copies that the
  // refusal is about.
  if (!splitReverseMemoryOk(orig, fn))
    return cachedArguments;

  Operation *newOp = gutils->getNewFromOriginal(orig);
  OpBuilder cacheBuilder(newOp);

  for (auto arg : orig->getOperands()) {
    Value toCache = gutils->getNewFromOriginal(arg);
    if (auto iface = dyn_cast<ClonableTypeInterface>(arg.getType())) {
      toCache = iface.cloneValue(cacheBuilder, toCache);
    }
    Value cache = gutils->initAndPushCache(toCache, cacheBuilder);
    cachedArguments.push_back(cache);
    // Cache the shadow pointer too; the reverse may not be able to rebuild it.
    if (!gutils->isConstantValue(arg) &&
        cast<AutoDiffTypeInterface>(arg.getType()).isMutable()) {
      Value shadow = gutils->invertPointerM(arg, cacheBuilder);
      cachedArguments.push_back(gutils->initAndPushCache(shadow, cacheBuilder));
    }
  }

  return cachedArguments;
}

LogicalResult edetail::callReverseHandler(Operation *orig, OpBuilder &builder,
                                          MGradientUtilsReverse *gutils,
                                          SmallVector<Value> caches) {
  auto fn = getDirectCallee(orig);
  if (!fn) {
    return orig->emitError()
           << "could not find the callee of: " << *orig << "\n";
  }
  if (fn.getFunctionBody().empty()) {
    return orig->emitError()
           << "cannot differentiate a call to a function without a body and "
              "without a registered derivative: "
           << fn.getNameAttr() << "\n";
  }

  // cacheValues chose the mode: an augmented primal in place of the call
  // means split mode.
  if (isa<CallAugmentedPrimalOp>(gutils->getNewFromOriginal(orig)))
    return callReverseHandlerSplit(orig, builder, gutils, caches, fn);
  return callReverseHandlerCombined(orig, builder, gutils, caches, fn);
}

// Whether a rule for exactly this call's activities is still being derived:
// the call is reached again from inside that derivation (recursion). The
// caller of a split-mode call carries the rule's caches as typed values, and
// those cannot contain themselves, so such a call takes the combined-mode
// path, which recomputes the callee in its reverse instead.
static bool splitRuleInProgress(Operation *orig, MGradientUtilsReverse *gutils,
                                FunctionOpInterface fn) {
  auto rules = lookupCustomReverseRules(fn, "enzyme.derived_rules");
  if (failed(rules))
    return false;

  std::vector<DIFFE_TYPE> ArgActivity, RetActivity;
  getCallActivity(orig, gutils, ArgActivity, RetActivity);
  SmallVector<Activity> ArgAct =
      llvm::map_to_vector(ArgActivity, diffeTypeToActivity);
  SmallVector<Activity> RetAct =
      llvm::map_to_vector(RetActivity, diffeTypeToActivity);

  return llvm::any_of(*rules, [&](CustomReverseRuleOp rule) {
    return rule->hasAttr("enzyme.in_progress") &&
           rule->hasAttr("enzyme.strong_zero") == gutils->strongZero &&
           rule.activityEquals(ArgAct, RetAct);
  });
}

SmallVector<Value> edetail::callCacheValues(Operation *orig,
                                            MGradientUtilsReverse *gutils) {
  // A callee the adjoint will refuse gets nothing cached, and the primal is
  // left as the plain call it already is.
  auto fn = getDirectCallee(orig);
  if (!fn || fn.getFunctionBody().empty())
    return SmallVector<Value>();

  if (splitModeSupported(fn) && !splitRuleInProgress(orig, gutils, fn))
    return callCacheValuesSplit(orig, gutils, fn);
  return callCacheValuesCombined(orig, gutils, fn);
}
