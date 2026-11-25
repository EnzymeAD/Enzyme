//===- ImpulseUtils.cpp - Utilities for Impulse dialect passes ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interfaces/ImpulseUtils.h"
#include "CloneFunction.h"
#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/APFloat.h"

using namespace mlir;
using namespace mlir::impulse;

FunctionOpInterface impulse::cloneForMode(FunctionOpInterface fn,
                                          ImpulseMode mode,
                                          int64_t positionSize,
                                          int64_t constraintSize) {
  assert(!fn.getFunctionBody().empty() && "cloning an empty function");
  Builder builder(fn.getContext());
  Type f64 = builder.getF64Type();
  StringRef suffix;
  SmallVector<Type> argTypes;
  switch (mode) {
  case ImpulseMode::Call:
    suffix = "call";
    break;
  case ImpulseMode::Simulate:
    suffix = "simulate";
    break;
  case ImpulseMode::Generate:
    suffix = "generate";
    argTypes.push_back(RankedTensorType::get({1, constraintSize}, f64));
    break;
  case ImpulseMode::Regenerate:
    suffix = "regenerate";
    argTypes.push_back(RankedTensorType::get({1, positionSize}, f64));
    if (constraintSize >= 0)
      argTypes.push_back(RankedTensorType::get({1, constraintSize}, f64));
    break;
  }
  int64_t numLeadingArgs = argTypes.size();
  SmallVector<Type> resultTypes;
  if (mode != ImpulseMode::Call) {
    resultTypes.push_back(RankedTensorType::get({1, positionSize}, f64));
    resultTypes.push_back(RankedTensorType::get({}, f64));
  }
  auto fnType = cast<FunctionType>(fn.getFunctionType());
  llvm::append_range(argTypes, fnType.getInputs());
  llvm::append_range(resultTypes, fnType.getResults());

  auto newFn = cast<FunctionOpInterface>(fn->cloneWithoutRegions());
  SymbolTable::setSymbolName(newFn, (fn.getName() + "." + suffix).str());
  newFn.setType(builder.getFunctionType(argTypes, resultTypes));
  SymbolTable(fn->getParentWithTrait<OpTrait::SymbolTable>()).insert(newFn);

  IRMapping mapping;
  cloneInto(&fn.getFunctionBody(), &newFn.getFunctionBody(), mapping);
  Block &entry = newFn.getFunctionBody().front();
  for (int64_t index = 0; index < numLeadingArgs; ++index)
    entry.insertArgument(index, argTypes[index], fn.getLoc());
  return newFn;
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

Value impulse::emitI1ScalarConstant(OpBuilder &builder, Location loc,
                                    bool value) {
  auto type = RankedTensorType::get({}, builder.getI1Type());
  return arith::ConstantOp::create(
      builder, loc, type,
      DenseElementsAttr::get(type, builder.getBoolAttr(value)));
}

Value impulse::emitBroadcastScalar(OpBuilder &builder, Location loc,
                                   Value scalar, RankedTensorType type) {
  return enzyme::BroadcastOp::create(
      builder, loc, type, scalar,
      builder.getDenseI64ArrayAttr(type.getShape()));
}

Value impulse::emitBroadcastConstant(OpBuilder &builder, Location loc,
                                     RankedTensorType type, double value) {
  Value scalar = emitSplatConstant(
      builder, loc, RankedTensorType::get({}, type.getElementType()), value);
  return emitBroadcastScalar(builder, loc, scalar, type);
}

Value impulse::emitMatrixConstant(
    OpBuilder &builder, Location loc, RankedTensorType type,
    function_ref<double(int64_t row, int64_t col)> entry) {
  const llvm::fltSemantics &semantics =
      cast<FloatType>(type.getElementType()).getFloatSemantics();
  SmallVector<APFloat> values;
  values.reserve(type.getNumElements());
  for (int64_t row = 0, rows = type.getDimSize(0); row < rows; ++row) {
    for (int64_t col = 0, cols = type.getDimSize(1); col < cols; ++col) {
      APFloat value(entry(row, col));
      bool losesInfo;
      value.convert(semantics, APFloat::rmNearestTiesToEven, &losesInfo);
      values.push_back(value);
    }
  }
  return arith::ConstantOp::create(builder, loc, type,
                                   DenseElementsAttr::get(type, values));
}

Value impulse::emitSumAll(OpBuilder &builder, Location loc, Value tensor) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto scalarType = RankedTensorType::get({}, tensorType.getElementType());
  Value ones = emitSplatConstant(builder, loc, tensorType, 1.0);
  DenseI64ArrayAttr allDims = builder.getDenseI64ArrayAttr(
      llvm::to_vector(llvm::seq<int64_t>(0, tensorType.getRank())));
  return impulse::DotOp::create(builder, loc, scalarType, tensor, ones, allDims,
                                allDims);
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

Value impulse::emitSoftplus(OpBuilder &builder, Location loc, Value x) {
  Value expX = math::ExpOp::create(builder, loc, x);
  return math::Log1pOp::create(builder, loc, expX);
}

Value impulse::emitSoftplusInv(OpBuilder &builder, Location loc, Value x) {
  Value expX = math::ExpOp::create(builder, loc, x);
  Value one =
      emitSplatConstant(builder, loc, cast<RankedTensorType>(x.getType()), 1.0);
  Value expXMinus1 = arith::SubFOp::create(builder, loc, expX, one);
  return math::LogOp::create(builder, loc, expXMinus1);
}

Value impulse::emitRandomDraw(OpBuilder &builder, Location loc, Value &rng,
                              RankedTensorType sampleType,
                              RngDistribution distribution) {
  auto scalarType = RankedTensorType::get({}, sampleType.getElementType());
  Value zero = emitSplatConstant(builder, loc, scalarType, 0.0);
  Value one = emitSplatConstant(builder, loc, scalarType, 1.0);
  auto op = impulse::RandomOp::create(
      builder, loc, TypeRange{rng.getType(), sampleType}, rng, zero, one,
      impulse::RngDistributionAttr::get(builder.getContext(), distribution));
  rng = op.getOutputRngState();
  return op.getResult();
}

Value impulse::emitUniformDraw(OpBuilder &builder, Location loc, Value &rng,
                               RankedTensorType sampleType) {
  return emitRandomDraw(builder, loc, rng, sampleType,
                        RngDistribution::UNIFORM);
}

Value impulse::emitRowSlice(OpBuilder &builder, Location loc, Value row,
                            int64_t column, RankedTensorType type) {
  int64_t numElements = type.getNumElements();
  auto sliceType =
      RankedTensorType::get({1, numElements}, type.getElementType());
  Value slice = impulse::SliceOp::create(
      builder, loc, sliceType, row, builder.getDenseI64ArrayAttr({0, column}),
      builder.getDenseI64ArrayAttr({1, column + numElements}),
      builder.getDenseI64ArrayAttr({1, 1}));
  return impulse::ReshapeOp::create(builder, loc, type, slice);
}

SmallVector<Value> impulse::cloneBlockBody(OpBuilder &builder, Block &block,
                                           IRMapping &mapping) {
  for (Operation &op : block.without_terminator())
    builder.clone(op, mapping);
  return llvm::map_to_vector(
      block.getTerminator()->getOperands(),
      [&](Value operand) { return mapping.lookupOrDefault(operand); });
}

Value impulse::conditionalDump(OpBuilder &builder, Location loc, Value value,
                               StringRef label, bool debugDump) {
  if (debugDump) {
    return enzyme::DumpOp::create(builder, loc, value.getType(), value,
                                  builder.getStringAttr(label))
        .getOutput();
  }
  return value;
}

/// Attributes of an `enzyme.autodiff_region` with `numActiveArgs` active
/// arguments that differentiates its first result and passes its second
/// through as a constant, followed by the entries of `extra` when it is set.
static SmallVector<NamedAttribute>
buildGradientRegionAttrs(Builder &builder, unsigned numActiveArgs,
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

ValueAndGradient impulse::emitGradientRegion(
    OpBuilder &builder, Location loc, ValueRange args, Value seed, Value &rng,
    DictionaryAttr extraAttrs,
    function_ref<Value(ValueRange regionArgs, Value &rng)> body) {
  SmallVector<Value> inputs(args);
  inputs.push_back(seed);
  SmallVector<Type> resultTypes{seed.getType(), rng.getType()};
  llvm::append_range(resultTypes, args.getTypes());
  auto regionOp = enzyme::AutoDiffRegionOp::create(
      builder, loc, resultTypes, inputs,
      buildGradientRegionAttrs(builder, args.size(), extraAttrs));

  Block *block = builder.createBlock(&regionOp.getBody());
  for (Value arg : args)
    block->addArgument(arg.getType(), loc);
  builder.setInsertionPointToStart(block);
  Value bodyRng = rng;
  Value value = body(block->getArguments(), bodyRng);
  enzyme::YieldOp::create(builder, loc, ValueRange{value, bodyRng});
  builder.setInsertionPointAfter(regionOp);

  rng = regionOp.getResult(1);
  return {regionOp.getResult(0),
          llvm::to_vector_of<Value>(regionOp.getResults().drop_front(2))};
}
