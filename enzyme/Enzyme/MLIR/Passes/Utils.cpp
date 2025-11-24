//===- Utils.cpp - General Utilities -------* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Utils.h"

using namespace mlir;
using namespace mlir::enzyme;

linalg::GenericOp Utils::adjointToGeneric(enzyme::GenericAdjointOp &op,
                                          OpBuilder &builder, Location loc) {
  auto inputs = op.getInputs();
  auto outputs = op.getOutputs();
  auto resultTensors = op.getResultTensors();
  auto indexingMaps = op.getIndexingMapsAttr();
  auto iteratorTypes = op.getIteratorTypesAttr();

  auto genericOp = mlir::linalg::GenericOp::create(
      builder, loc, TypeRange(resultTensors), ValueRange(inputs),
      ValueRange(outputs), ArrayAttr(indexingMaps), ArrayAttr(iteratorTypes),
      StringAttr(), StringAttr());

  auto &body = genericOp.getRegion();
  body.takeBody(op.getRegion());

  op.erase();

  return genericOp;
}

bool mlir::enzyme::opCmp(Operation *a, Operation *b) {
  if (a == b)
    return false;

  // Ancestors are less than their descendants.
  if (a->isProperAncestor(b)) {
    return true;
  } else if (b->isProperAncestor(a->getParentOp())) {
    return false;
  }

  // Move a and b to be direct descendents of the same op
  while (!a->getParentOp()->isAncestor(b))
    a = a->getParentOp();

  while (!b->getParentOp()->isAncestor(a))
    b = b->getParentOp();

  assert(a->getParentOp() == b->getParentOp());

  if (a->getBlock() == b->getBlock()) {
    return a->isBeforeInBlock(b);
  } else {
    return blockCmp(a->getBlock(), b->getBlock());
  }
}

bool mlir::enzyme::regionCmp(Region *a, Region *b) {
  if (a == b)
    return false;

  // Ancestors are less than their descendants.
  if (a->getParentOp()->isProperAncestor(b->getParentOp())) {
    return true;
  } else if (b->getParentOp()->isProperAncestor(a->getParentOp())) {
    return false;
  }

  if (a->getParentOp() == b->getParentOp()) {
    return a->getRegionNumber() < b->getRegionNumber();
  }
  return opCmp(a->getParentOp(), b->getParentOp());
}

bool mlir::enzyme::blockCmp(Block *a, Block *b) {
  if (a == b)
    return false;

  // Ancestors are less than their descendants.
  if (a->getParent()->isProperAncestor(b->getParent())) {
    return true;
  } else if (b->getParent()->isProperAncestor(a->getParent())) {
    return false;
  }

  if (a->getParent() == b->getParent()) {
    // If the blocks are in the same region, then the first one in
    // the region is less than the second one.
    for (auto &bb : *b->getParent()) {
      if (&bb == a)
        return true;
    }
    return false;
  }

  return regionCmp(a->getParent(), b->getParent());
}

bool mlir::enzyme::valueCmp(mlir::Value a, mlir::Value b) {
  // Equal values are not less than each other.
  if (a == b)
    return false;

  auto ba = dyn_cast<BlockArgument>(a);
  auto bb = dyn_cast<BlockArgument>(b);
  // Define block arguments are less than non-block arguments.
  if (ba && !bb)
    return true;
  if (!ba && bb)
    return false;
  if (ba && bb) {
    if (ba.getOwner() == bb.getOwner()) {
      return ba.getArgNumber() < bb.getArgNumber();
    }
    return blockCmp(ba.getOwner(), bb.getOwner());
  }

  OpResult ra = cast<OpResult>(a);
  OpResult rb = cast<OpResult>(b);

  if (ra.getOwner() == rb.getOwner()) {
    return ra.getResultNumber() < rb.getResultNumber();
  } else {
    return opCmp(ra.getOwner(), rb.getOwner());
  }
}

Type mlir::enzyme::getConcatType(Value val, int64_t width) {
  auto valTy = val.getType();
  if (auto valTensorTy = dyn_cast<TensorType>(valTy)) {
    // val is a tensor, prepend batch width to shape
    SmallVector<int64_t> out_shape = {width};
    out_shape.append(valTensorTy.getShape().begin(),
                     valTensorTy.getShape().end());
    auto outTy = valTensorTy.clone(out_shape);
    return outTy;
  } else if (auto valMemrefTy = dyn_cast<MemRefType>(valTy)) {
    // val is a memref, prepend batch width
    SmallVector<int64_t> out_shape = {width};
    out_shape.append(valMemrefTy.getShape().begin(),
                     valMemrefTy.getShape().end());
    auto outTy = valMemrefTy.clone(out_shape);
    return outTy;
  } else {
    // val is a scalar
    return RankedTensorType::get(width, valTy);
  }
}

Value mlir::enzyme::getConcatValue(OpBuilder &builder, Location loc,
                                   ArrayRef<Value> argList) {
  int64_t width = argList.size();
  Type out_type = mlir::enzyme::getConcatType(argList.front(), width);
  mlir::Value out = enzyme::ConcatOp::create(builder, loc, out_type, argList);
  return out;
}

Value mlir::enzyme::getExtractValue(OpBuilder &builder, Location loc,
                                    Type argTy, Value val, int64_t index) {
  // Extract the original output from the tensorized output at the given index.
  IntegerAttr indexAttr = builder.getI64IntegerAttr(index);
  Value out = enzyme::ExtractOp::create(builder, loc, argTy, val, indexAttr);
  return out;
}
