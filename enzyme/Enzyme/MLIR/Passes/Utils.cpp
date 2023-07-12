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

  auto genericOp = builder.create<mlir::linalg::GenericOp>(
      loc, TypeRange(resultTensors), ValueRange(inputs), ValueRange(outputs),
      ArrayAttr(indexingMaps), ArrayAttr(iteratorTypes), StringAttr(),
      StringAttr());

  auto &body = genericOp.getRegion();
  body.takeBody(op.getRegion());

  op.erase();

  return genericOp;
}