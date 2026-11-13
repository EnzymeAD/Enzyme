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
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Interfaces/Utils.h"
#include "Passes/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Casting.h"
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
    } else if (auto IET = dyn_cast<IntegerType>(T.getElementType())) {
      APInt values[] = {APInt(IET.getWidth(), value, /*radix=*/10)};
      return DenseElementsAttr::get(cast<ShapedType>(type),
                                    ArrayRef<APInt>(values));
    } else {
      llvm::errs() << " unsupported eltype: " << T.getElementType()
                   << " of type " << type << "\n";
      llvm_unreachable("unsupported eltype");
    }
  } else if (auto T = dyn_cast<FloatType>(type)) {
    APFloat apvalue(T.getFloatSemantics(), value);
    return FloatAttr::get(T, apvalue);
    // NOTE `complex::ConstantOp` doesn't accept `TypedAttr`, only `ArrayAttr`
    // } else if (auto T = cast<ComplexType>(type)) {
    //   auto F = cast<FloatType>(T.getElementType());
    //   return mlir::ArrayAttr::get({
    //     FloatAttr::get(F, APFloat(F.getFloatSemantics(), value)),
    //     FloatAttr::get(F, APFloat(F.getFloatSemantics(), "0"));
    //   });
  } else if (auto T = dyn_cast<IntegerType>(type)) {
    APInt apvalue(T.getWidth(), value, /*radix=*/10);
    return IntegerAttr::get(T, apvalue);
  } else {
    llvm::errs() << " unsupported type: " << type << "\n";
    llvm_unreachable("unsupported eltype");
  }
}

mlir::Attribute mlir::enzyme::getComplexConstantAttr(mlir::Type type,
                                                     llvm::StringRef re,
                                                     llvm::StringRef im) {
  using namespace mlir;

  if (auto T = dyn_cast<ComplexType>(type)) {
    auto F = cast<FloatType>(T.getElementType());
    APFloat revalue(F.getFloatSemantics(), re);
    APFloat imvalue(F.getFloatSemantics(), im);
    return mlir::ArrayAttr::get(
        type.getContext(),
        {FloatAttr::get(F, revalue), FloatAttr::get(F, imvalue)});
  } else if (auto T = dyn_cast<TensorType>(type)) {
    if (auto CET = dyn_cast<ComplexType>(T.getElementType())) {
      auto ET = cast<FloatType>(CET.getElementType());
      APFloat revalue(ET.getFloatSemantics(), re);
      APFloat imvalue(ET.getFloatSemantics(), im);
      mlir::Complex<APFloat> values[] = {
          mlir::Complex<APFloat>(revalue, imvalue)};
      return DenseElementsAttr::get(cast<ShapedType>(type),
                                    ArrayRef<mlir::Complex<APFloat>>(values));
    }
  }
  llvm::errs() << " unsupported type: " << type << "\n";
  llvm_unreachable("unsupported eltype");
}

bool mlir::enzyme::allResultsActive(Operation *op,
                                    const MGradientUtils *gutils) {
  for (Value result : op->getResults())
    if (gutils->isConstantValue(result))
      return false;
  return true;
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

  bool allStored = contains(storedVals, -1);
  bool needsShadow = false;
  for (Value v : orig->getResults()) {
    if (!gutils->isConstantValue(v)) {
      needsShadow = true;
      break;
    }
  }
  if (!needsShadow && !allStored) {
    for (OpOperand &operand : orig->getOpOperands()) {
      unsigned idx = operand.getOperandNumber();
      if (!iface.isArgInactive(idx) && !contains(storedVals, idx) &&
          !gutils->isConstantValue(operand.get())) {
        needsShadow = true;
        break;
      }
    }
  }
  if (!needsShadow)
    return success();

  SmallVector<Value> newOperands;
  newOperands.reserve(orig->getNumOperands());
  SmallVector<bool> inverted(orig->getNumOperands(), false);
  for (OpOperand &operand : orig->getOpOperands()) {
    if (iface.isArgInactive(operand.getOperandNumber())) {
      newOperands.push_back(gutils->getNewFromOriginal(operand.get()));
    } else {
      if (gutils->isConstantValue(operand.get())) {

        if (allStored || contains(storedVals, operand.getOperandNumber())) {
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
  // An inactive allocation has no shadow.
  if (gutils->isConstantValue(orig->getResult(0)))
    return success();

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
  gutils->originalToNewFn.map(op, replacement);

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

// The reverse call runs long after the forward one, against whatever memory
// looks like by then -- even argument memory may have been overwritten in
// between, and deciding which of it was is the overwritten-args analysis
// Enzyme's LLVM side has and this side does not yet. Until it does, only a
// callee that touches no memory at all is differentiable here: readnone, or
// every op in its body free of memory effects.
// Whether a memory-effects attribute -- later LLVM's spelling of
// readnone -- rules out every kind of access.
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

// Trace `v` within `fn` back to either entry block arguments of `fn` or local
// allocations inside `fn`. Returns false if `v` may originate from any other
// memory (e.g., globals, loaded pointers, or unknown call results).
static bool traceCalleeMemoryOrigins(Value v, FunctionOpInterface fn,
                                     SmallVectorImpl<unsigned> &argIndices,
                                     bool &hasLocalAlloc) {
  SmallVector<Value, 4> worklist = {v};
  SmallPtrSet<Value, 8> seen;
  Block *entryBlock = &fn.getFunctionBody().front();

  while (!worklist.empty()) {
    Value cur = oputils::getBaseObject(worklist.pop_back_val());
    if (!seen.insert(cur).second)
      continue;

    if (auto castOp = cur.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (castOp.getNumOperands() != 1)
        return false;
      worklist.push_back(castOp.getOperand(0));
      continue;
    }
    if (auto selOp = cur.getDefiningOp<arith::SelectOp>()) {
      worklist.push_back(selOp.getTrueValue());
      worklist.push_back(selOp.getFalseValue());
      continue;
    }
    if (auto selOp = cur.getDefiningOp<LLVM::SelectOp>()) {
      worklist.push_back(selOp.getTrueValue());
      worklist.push_back(selOp.getFalseValue());
      continue;
    }
    if (auto ifOp = cur.getDefiningOp<scf::IfOp>()) {
      unsigned resIdx = cast<OpResult>(cur).getResultNumber();
      worklist.push_back(ifOp.thenYield().getOperand(resIdx));
      if (!ifOp.getElseRegion().empty())
        worklist.push_back(ifOp.elseYield().getOperand(resIdx));
      continue;
    }

    if (auto ba = dyn_cast<BlockArgument>(cur)) {
      if (ba.getOwner() == entryBlock) {
        argIndices.push_back(ba.getArgNumber());
        continue;
      }
      if (ba.getOwner()->getParent() == &fn.getFunctionBody()) {
        for (Block *pred : ba.getOwner()->getPredecessors()) {
          auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
          if (!branch)
            return false;
          for (unsigned succIdx = 0; succIdx < pred->getNumSuccessors();
               ++succIdx) {
            if (pred->getSuccessor(succIdx) != ba.getOwner())
              continue;
            SuccessorOperands succOps = branch.getSuccessorOperands(succIdx);
            if (ba.getArgNumber() >= succOps.getProducedOperandCount()) {
              unsigned opIdx =
                  ba.getArgNumber() - succOps.getProducedOperandCount();
              if (opIdx < succOps.size())
                worklist.push_back(succOps[opIdx]);
              else
                return false;
            } else {
              return false;
            }
          }
        }
        continue;
      }
      return false;
    }

    if (Operation *def = cur.getDefiningOp()) {
      if (def->getParentOfType<FunctionOpInterface>() == fn &&
          (isa<enzyme::AllocaLikeInterface>(def) ||
           isa<enzyme::MallocLikeInterface>(def))) {
        hasLocalAlloc = true;
        continue;
      }
    }

    return false;
  }

  return true;
}

static bool writeMayEscapePointer(Operation *op, Value dest) {
  if (auto store = dyn_cast<memref::StoreOp>(op)) {
    auto ad =
        dyn_cast<AutoDiffTypeInterface>(store.getValueToStore().getType());
    return ad && ad.isMutable();
  }
  if (auto store = dyn_cast<LLVM::StoreOp>(op)) {
    auto ad = dyn_cast<AutoDiffTypeInterface>(store.getValue().getType());
    return ad && ad.isMutable();
  }
  if (auto copy = dyn_cast<memref::CopyOp>(op)) {
    auto ad = dyn_cast<AutoDiffTypeInterface>(
        copy.getSource().getType().getElementType());
    return ad && ad.isMutable();
  }
  if (isa<LLVM::MemcpyOp, LLVM::MemmoveOp, LLVM::MemsetOp>(op))
    return false;

  for (Value operand : op->getOperands()) {
    if (operand == dest)
      continue;
    if (auto ad = dyn_cast<AutoDiffTypeInterface>(operand.getType()))
      if (ad.isMutable())
        return true;
  }
  return false;
}

// Summarizes which entry arguments of `fn` are read or written when called at
// `orig`. Returns false if `fn` may access non-argument memory (globals,
// loaded pointers, unknown side effects), free caller memory, or escape a
// pointer through a written argument.
static bool analyzeCalleeArgEffectsImpl(Operation *orig, FunctionOpInterface fn,
                                        SmallVectorImpl<bool> &argRead,
                                        SmallVectorImpl<bool> &argWrite,
                                        SmallPtrSetImpl<Operation *> &visited) {
  unsigned numArgs = fn.getNumArguments();
  argRead.assign(numArgs, false);
  argWrite.assign(numArgs, false);

  if (splitReverseMemoryOk(orig, fn))
    return true;
  if (fn.getFunctionBody().empty())
    return false;
  if (!visited.insert(fn.getOperation()).second)
    return false;
  auto eraseOnExit =
      llvm::scope_exit([&]() { visited.erase(fn.getOperation()); });

  auto walkRegion = [&](auto &&self, Region &region) -> bool {
    for (Block &blk : region) {
      for (Operation &op : blk) {
        if (isMemoryEffectFree(&op))
          continue;

        if (op.getNumRegions() > 0 &&
            op.hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
          for (Region &subR : op.getRegions())
            if (!self(self, subR))
              return false;
          continue;
        }

        if (auto callee = getDirectCallee(&op)) {
          SmallVector<bool> subRead, subWrite;
          if (!analyzeCalleeArgEffectsImpl(&op, callee, subRead, subWrite,
                                           visited))
            return false;
          for (unsigned j = 0, e = op.getNumOperands(); j < e; ++j) {
            if (!subRead[j] && !subWrite[j])
              continue;
            SmallVector<unsigned, 4> roots;
            bool hasLocal = false;
            if (!traceCalleeMemoryOrigins(op.getOperand(j), fn, roots,
                                          hasLocal))
              return false;
            for (unsigned idx : roots) {
              if (subRead[j])
                argRead[idx] = true;
              if (subWrite[j])
                argWrite[idx] = true;
            }
          }
          continue;
        }

        auto memIface = dyn_cast<MemoryEffectOpInterface>(&op);
        if (!memIface)
          return false;

        SmallVector<MemoryEffects::EffectInstance, 4> effects;
        memIface.getEffects(effects);
        for (const MemoryEffects::EffectInstance &eff : effects) {
          if (isa<MemoryEffects::Allocate>(eff.getEffect()))
            continue;
          Value val = eff.getValue();
          if (!val)
            return false;

          SmallVector<unsigned, 4> roots;
          bool hasLocal = false;
          if (!traceCalleeMemoryOrigins(val, fn, roots, hasLocal))
            return false;

          if (isa<MemoryEffects::Free>(eff.getEffect())) {
            if (!roots.empty())
              return false;
            continue;
          }
          if (isa<MemoryEffects::Read>(eff.getEffect())) {
            for (unsigned idx : roots)
              argRead[idx] = true;
          }
          if (isa<MemoryEffects::Write>(eff.getEffect())) {
            if (!roots.empty() && writeMayEscapePointer(&op, val))
              return false;
            for (unsigned idx : roots)
              argWrite[idx] = true;
          }
        }
      }
    }
    return true;
  };

  if (!walkRegion(walkRegion, fn.getFunctionBody()))
    return false;

  // Reject callees that return a local allocation (mutable return escaping fn).
  for (Block &blk : fn.getFunctionBody()) {
    if (blk.getNumSuccessors() != 0)
      continue;
    for (Value retVal : blk.getTerminator()->getOperands()) {
      if (auto ad = dyn_cast<AutoDiffTypeInterface>(retVal.getType())) {
        if (ad.isMutable()) {
          SmallVector<unsigned, 4> roots;
          bool hasLocal = false;
          if (!traceCalleeMemoryOrigins(retVal, fn, roots, hasLocal) ||
              hasLocal)
            return false;
        }
      }
    }
  }

  return true;
}

static bool opMayWriteAlias(Operation *op, Value target) {
  if (isMemoryEffectFree(op))
    return false;

  if (op->getNumRegions() > 0 &&
      op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (Region &subR : op->getRegions())
      for (Block &blk : subR)
        for (Operation &nested : blk)
          if (opMayWriteAlias(&nested, target))
            return true;
    return false;
  }

  if (auto callee = getDirectCallee(op)) {
    SmallVector<bool> subRead, subWrite;
    SmallPtrSet<Operation *, 8> visited;
    if (!analyzeCalleeArgEffectsImpl(op, callee, subRead, subWrite, visited))
      return true;
    for (unsigned j = 0, e = op->getNumOperands(); j < e; ++j)
      if (subWrite[j] && oputils::mayAlias(op->getOperand(j), target))
        return true;
    return false;
  }

  if (auto memIface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    memIface.getEffects(effects);
    for (const MemoryEffects::EffectInstance &eff : effects) {
      if (isa<MemoryEffects::Write, MemoryEffects::Free>(eff.getEffect())) {
        if (!eff.getValue() || oputils::mayAlias(eff.getValue(), target))
          return true;
      }
    }
    return false;
  }

  return true;
}

static bool isOverwrittenInCallerAfterCall(Operation *orig, Value target) {
  Operation *cur = orig;
  while (cur && !isa<FunctionOpInterface>(cur)) {
    Block *blk = cur->getBlock();
    for (auto it = std::next(cur->getIterator()), end = blk->end(); it != end;
         ++it) {
      if (opMayWriteAlias(&*it, target))
        return true;
    }

    SmallPtrSet<Block *, 8> visitedBlocks;
    SmallVector<Block *, 8> worklist;
    for (Block *succ : blk->getSuccessors())
      worklist.push_back(succ);
    while (!worklist.empty()) {
      Block *b = worklist.pop_back_val();
      if (!visitedBlocks.insert(b).second)
        continue;
      for (Operation &op : *b) {
        if (&op != orig && opMayWriteAlias(&op, target))
          return true;
      }
      for (Block *succ : b->getSuccessors())
        worklist.push_back(succ);
    }

    Operation *parent = blk->getParentOp();
    if (!parent || isa<FunctionOpInterface>(parent))
      break;

    // An enclosing non-if region op (e.g. scf.for / affine.for / scf.while)
    // may re-execute its body in later iterations before the reverse pass runs.
    if (!isa<scf::IfOp>(parent)) {
      bool foundWrite = false;
      parent->walk([&](Operation *nested) {
        if (nested == orig || nested == parent)
          return WalkResult::advance();
        if (nested->getNumRegions() > 0 &&
            nested->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
          return WalkResult::advance();
        if (opMayWriteAlias(nested, target)) {
          foundWrite = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (foundWrite)
        return true;
    }

    cur = parent;
  }
  return false;
}

struct ReverseCallArgPlan {
  SmallVector<bool> argRead;
  SmallVector<bool> argWrite;
  SmallVector<bool> needsClone;
  std::string failureReason;

  bool ok() const { return failureReason.empty(); }
};

static ReverseCallArgPlan analyzeReverseCall(Operation *orig,
                                             FunctionOpInterface fn,
                                             MGradientUtilsReverse *gutils) {
  ReverseCallArgPlan plan;
  unsigned narg = orig->getNumOperands();
  plan.needsClone.assign(narg, false);

  SmallPtrSet<Operation *, 8> visited;
  if (!analyzeCalleeArgEffectsImpl(orig, fn, plan.argRead, plan.argWrite,
                                   visited)) {
    plan.failureReason = "cannot differentiate a call in reverse mode whose "
                         "callee accesses non-argument memory: ";
    return plan;
  }

  for (unsigned i = 0; i < narg; ++i) {
    if (plan.argWrite[i] || (plan.argRead[i] && isOverwrittenInCallerAfterCall(
                                                    orig, orig->getOperand(i))))
      plan.needsClone[i] = true;
  }

  // If the same SSA value appears in multiple operand slots and any slot needs
  // a snapshot clone, share that clone across all its slots to preserve
  // must-alias semantics during reverse replay.
  SmallPtrSet<Value, 4> clonedValues;
  for (unsigned i = 0; i < narg; ++i)
    if (plan.needsClone[i])
      clonedValues.insert(orig->getOperand(i));
  for (unsigned i = 0; i < narg; ++i)
    if (clonedValues.contains(orig->getOperand(i)))
      plan.needsClone[i] = true;

  // Reject distinct operands that may alias while the callee writes one of
  // them, since cloning them separately would break aliasing on replay.
  for (unsigned i = 0; i < narg; ++i) {
    if (!plan.argRead[i] && !plan.argWrite[i])
      continue;
    for (unsigned j = i + 1; j < narg; ++j) {
      if (!plan.argRead[j] && !plan.argWrite[j])
        continue;
      if (!plan.argWrite[i] && !plan.argWrite[j])
        continue;
      if (orig->getOperand(i) == orig->getOperand(j))
        continue;
      if (oputils::mayAlias(orig->getOperand(i), orig->getOperand(j))) {
        plan.failureReason = "cannot differentiate a call in reverse mode "
                             "whose operands may alias a written argument: ";
        return plan;
      }
    }
  }

  for (unsigned i = 0; i < narg; ++i) {
    if (!plan.needsClone[i])
      continue;
    Value origArg = orig->getOperand(i);
    Value newArg = gutils->getNewFromOriginal(origArg);
    auto iface = dyn_cast<ClonableTypeInterface>(origArg.getType());
    if (!iface || !iface.canCloneValue(newArg)) {
      plan.failureReason = "cannot differentiate a call in reverse mode: "
                           "cannot clone overwritten argument of callee ";
      return plan;
    }
  }

  return plan;
}

LogicalResult edetail::callReverseHandler(Operation *orig, OpBuilder &builder,
                                          MGradientUtilsReverse *gutils,
                                          SmallVector<Value> caches) {
  DerivativeMode mode = DerivativeMode::ReverseModeGradient;

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

  ReverseCallArgPlan plan = analyzeReverseCall(orig, fn, gutils);
  if (!plan.ok())
    return orig->emitError() << plan.failureReason << fn.getNameAttr() << "\n";

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

  std::vector<bool> overwritten_args(plan.needsClone.begin(),
                                     plan.needsClone.end());
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
  llvm::SmallDenseMap<Value, Value, 4> poppedClones;
  SmallVector<std::pair<Type, Value>, 4> clonesToFree;

  size_t cacheIdx = 0;
  for (unsigned i = 0, e = orig->getNumOperands(); i < e; ++i) {
    Value arg = orig->getOperand(i);
    DIFFE_TYPE act = ArgActivity[i];
    if (plan.needsClone[i]) {
      auto it = poppedClones.find(arg);
      Value clone;
      if (it == poppedClones.end()) {
        clone = gutils->popCache(caches[cacheIdx++], builder);
        poppedClones[arg] = clone;
        clonesToFree.emplace_back(arg.getType(), clone);
      } else {
        clone = it->second;
      }
      Value callArg = clone;
      if (callArg.getType() != arg.getType() &&
          isa<LLVM::LLVMPointerType>(callArg.getType()) &&
          isa<LLVM::LLVMPointerType>(arg.getType())) {
        callArg = LLVM::AddrSpaceCastOp::create(builder, orig->getLoc(),
                                                arg.getType(), callArg);
      }
      revArguments.push_back(callArg);
    } else {
      revArguments.push_back(gutils->popCache(caches[cacheIdx++], builder));
    }
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

  for (auto [ty, cloneVal] : clonesToFree)
    cast<ClonableTypeInterface>(ty).freeClonedValue(builder, cloneVal);

  int revIndex = 0;
  for (auto [arg, act] : llvm::zip_equal(orig->getOperands(), ArgActivity)) {
    if (act == DIFFE_TYPE::OUT_DIFF) {
      auto diffe = revCallOp->getResult(revIndex);
      gutils->addToDiffe(arg, diffe, builder);
      revIndex++;
    }
  }

  return success();
}

SmallVector<Value> edetail::callCacheValues(Operation *orig,
                                            MGradientUtilsReverse *gutils) {
  SmallVector<Value> cachedArguments;

  auto fn = getDirectCallee(orig);
  if (!fn || fn.getFunctionBody().empty())
    return cachedArguments;

  ReverseCallArgPlan plan = analyzeReverseCall(orig, fn, gutils);
  if (!plan.ok())
    return cachedArguments;

  Operation *newOp = gutils->getNewFromOriginal(orig);
  OpBuilder cacheBuilder(newOp);
  SmallPtrSet<Value, 4> alreadyCloned;

  for (unsigned i = 0, e = orig->getNumOperands(); i < e; ++i) {
    Value arg = orig->getOperand(i);
    Value toCache = gutils->getNewFromOriginal(arg);
    if (plan.needsClone[i]) {
      if (alreadyCloned.insert(arg).second) {
        Value clone = cast<ClonableTypeInterface>(arg.getType())
                          .cloneValue(cacheBuilder, toCache);
        cachedArguments.push_back(
            gutils->initAndPushCache(clone, cacheBuilder));
      }
    } else {
      cachedArguments.push_back(
          gutils->initAndPushCache(toCache, cacheBuilder));
    }
    if (!gutils->isConstantValue(arg) &&
        cast<AutoDiffTypeInterface>(arg.getType()).isMutable()) {
      Value shadow = gutils->invertPointerM(arg, cacheBuilder);
      cachedArguments.push_back(gutils->initAndPushCache(shadow, cacheBuilder));
    }
  }

  return cachedArguments;
}
