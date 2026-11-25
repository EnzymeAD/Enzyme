//===- ImpulseUtils.cpp - Utilities for Impulse dialect passes ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interfaces/ImpulseUtils.h"
#include "CloneFunction.h"
#include "Dialect/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/APFloat.h"

using namespace mlir;
using namespace mlir::impulse;

ImpulseUtils *ImpulseUtils::CreateFromClone(FunctionOpInterface toeval,
                                            ImpulseMode mode,
                                            int64_t positionSize,
                                            int64_t constraintSize) {
  if (toeval.getFunctionBody().empty()) {
    llvm::errs() << toeval << "\n";
    llvm_unreachable("Creating ImpulseUtils from empty function");
  }

  OpBuilder builder(toeval.getContext());

  std::string suffix;
  auto originalInputs =
      cast<mlir::FunctionType>(toeval.getFunctionType()).getInputs();
  auto originalResults =
      cast<mlir::FunctionType>(toeval.getFunctionType()).getResults();
  // Arguments of the clone that precede the original inputs.
  SmallVector<Type> leadingTypes;
  SmallVector<mlir::Type, 4> ResultTypes;

  switch (mode) {
  case ImpulseMode::Call:
    suffix = "call";
    ResultTypes.append(originalResults.begin(), originalResults.end());
    break;
  case ImpulseMode::Generate:
    suffix = "generate";
    if (positionSize <= 0 || constraintSize < 0) {
      toeval.emitError("Impulse: Unexpected size parameters");
      return nullptr;
    }
    leadingTypes.push_back(
        RankedTensorType::get({1, constraintSize}, builder.getF64Type()));
    ResultTypes.push_back(
        RankedTensorType::get({1, positionSize}, builder.getF64Type()));
    ResultTypes.push_back(RankedTensorType::get({}, builder.getF64Type()));
    ResultTypes.append(originalResults.begin(), originalResults.end());
    break;
  case ImpulseMode::Regenerate:
    suffix = "regenerate";
    if (positionSize < 0) {
      toeval.emitError("Impulse: Unexpected size parameters");
      return nullptr;
    }
    leadingTypes.push_back(
        RankedTensorType::get({1, positionSize}, builder.getF64Type()));
    if (constraintSize >= 0)
      leadingTypes.push_back(
          RankedTensorType::get({1, constraintSize}, builder.getF64Type()));
    ResultTypes.push_back(
        RankedTensorType::get({1, positionSize}, builder.getF64Type()));
    ResultTypes.push_back(RankedTensorType::get({}, builder.getF64Type()));
    ResultTypes.append(originalResults.begin(), originalResults.end());
    break;
  case ImpulseMode::Simulate:
    suffix = "simulate";
    if (positionSize < 0) {
      toeval.emitError("Impulse: Unexpected size parameters");
      return nullptr;
    }
    ResultTypes.push_back(
        RankedTensorType::get({1, positionSize}, builder.getF64Type()));
    ResultTypes.push_back(RankedTensorType::get({}, builder.getF64Type()));
    ResultTypes.append(originalResults.begin(), originalResults.end());
    break;
  default:
    llvm_unreachable("Invalid ImpulseMode\n");
  }

  SmallVector<Type> operandTypes(leadingTypes);
  operandTypes.append(originalInputs.begin(), originalInputs.end());
  mlir::FunctionType functionType =
      builder.getFunctionType(operandTypes, ResultTypes);
  auto NewF = cast<FunctionOpInterface>(toeval->cloneWithoutRegions());
  SymbolTable::setSymbolName(NewF, toeval.getName().str() + "." + suffix);
  NewF.setType(functionType);

  Operation *parent = toeval->getParentWithTrait<OpTrait::SymbolTable>();
  SymbolTable table(parent);
  table.insert(NewF);

  IRMapping originalToNew;
  cloneInto(&toeval.getFunctionBody(), &NewF.getFunctionBody(), originalToNew);

  Block &entry = NewF.getFunctionBody().front();
  for (auto [index, type] : llvm::enumerate(leadingTypes))
    entry.insertArgument(index, type, toeval.getLoc());

  return new ImpulseUtils(NewF, toeval, originalToNew, mode);
}

Value impulse::emitSplatConstant(OpBuilder &builder, Location loc,
                                 RankedTensorType type, double value) {
  return arith::ConstantOp::create(
      builder, loc, type,
      DenseElementsAttr::get(
          type, builder.getFloatAttr(type.getElementType(), value)));
}

Value impulse::emitI64ScalarConstant(OpBuilder &builder, Location loc,
                                     int64_t value) {
  auto type = RankedTensorType::get({}, builder.getI64Type());
  return arith::ConstantOp::create(
      builder, loc, type,
      DenseElementsAttr::get(type, builder.getI64IntegerAttr(value)));
}

Value impulse::emitBroadcastScalar(OpBuilder &builder, Location loc,
                                   Value scalar, RankedTensorType type) {
  return enzyme::BroadcastOp::create(
      builder, loc, type, scalar,
      builder.getDenseI64ArrayAttr(type.getShape()));
}

Value impulse::emitLogAddExp(OpBuilder &builder, Location loc, Value lhs,
                             Value rhs) {
  auto amax = arith::MaximumFOp::create(builder, loc, lhs, rhs);
  auto delta = arith::SubFOp::create(builder, loc, lhs, rhs);
  auto isNaN = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::UNE,
                                     delta, delta);
  auto nanResult = arith::AddFOp::create(builder, loc, lhs, rhs);
  auto absDelta = math::AbsFOp::create(builder, loc, delta);
  auto negAbsDelta = arith::NegFOp::create(builder, loc, absDelta);
  auto expNegAbsDelta = math::ExpOp::create(builder, loc, negAbsDelta);
  auto log1pResult = math::Log1pOp::create(builder, loc, expNegAbsDelta);
  auto normalResult = arith::AddFOp::create(builder, loc, amax, log1pResult);
  return arith::SelectOp::create(builder, loc, isNaN, nanResult, normalResult);
}

Value impulse::emitIndicatorMatrix(OpBuilder &builder, Location loc,
                                   RankedTensorType type,
                                   function_ref<bool(int64_t, int64_t)> isOne) {
  const llvm::fltSemantics &semantics =
      cast<FloatType>(type.getElementType()).getFloatSemantics();
  APFloat zero(semantics, 0);
  APFloat one(semantics, 1);
  SmallVector<APFloat> values;
  values.reserve(type.getNumElements());
  for (int64_t i = 0, rows = type.getDimSize(0); i < rows; ++i)
    for (int64_t j = 0, cols = type.getDimSize(1); j < cols; ++j)
      values.push_back(isOne(i, j) ? one : zero);
  return arith::ConstantOp::create(builder, loc, type,
                                   DenseElementsAttr::get(type, values));
}

SmallVector<Value> impulse::cloneBlockBody(OpBuilder &builder, Block &block,
                                           IRMapping &mapping) {
  for (Operation &op : block.without_terminator())
    builder.clone(op, mapping);
  return llvm::map_to_vector(
      block.getTerminator()->getOperands(),
      [&](Value operand) { return mapping.lookupOrDefault(operand); });
}

SmallVector<NamedAttribute>
impulse::buildGradientRegionAttrs(Builder &builder, unsigned numActiveArgs,
                                  DictionaryAttr extra) {
  MLIRContext *context = builder.getContext();
  auto active =
      enzyme::ActivityAttr::get(context, enzyme::Activity::enzyme_active);
  auto constant =
      enzyme::ActivityAttr::get(context, enzyme::Activity::enzyme_const);
  SmallVector<NamedAttribute> attrs{
      builder.getNamedAttr(
          "activity",
          builder.getArrayAttr(SmallVector<Attribute>(numActiveArgs, active))),
      builder.getNamedAttr("ret_activity",
                           builder.getArrayAttr({active, constant})),
  };
  if (extra)
    llvm::append_range(attrs, extra);
  return attrs;
}
