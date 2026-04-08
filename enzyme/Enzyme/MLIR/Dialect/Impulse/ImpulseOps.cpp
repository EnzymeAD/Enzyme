//===- ImpulseOps.cpp - Impulse dialect ops ----------------------*- C++
//-*-===//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Impulse/Impulse.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::impulse;

//===----------------------------------------------------------------------===//
// SampleOp
//===----------------------------------------------------------------------===//

LogicalResult SampleOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  if (getLogpdfAttr()) {
    auto global = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
        *this, getLogpdfAttr());
    if (!global)
      return emitOpError("'")
             << getLogpdf().value() << "' does not reference a valid global "
             << "funcOp";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GenerateOp
//===----------------------------------------------------------------------===//

LogicalResult GenerateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

//===----------------------------------------------------------------------===//
// SimulateOp
//===----------------------------------------------------------------------===//

LogicalResult SimulateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

//===----------------------------------------------------------------------===//
// RegenerateOp
//===----------------------------------------------------------------------===//

LogicalResult
RegenerateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

//===----------------------------------------------------------------------===//
// MHOp
//===----------------------------------------------------------------------===//

LogicalResult MHOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

//===----------------------------------------------------------------------===//
// InferOp
//===----------------------------------------------------------------------===//

LogicalResult InferOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (auto fnAttr = getFnAttr()) {
    auto global =
        symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, fnAttr);
    if (!global)
      return emitOpError("'")
             << getFn().value() << "' does not reference a valid global funcOp";
  }

  if (auto logpdfAttr = getLogpdfFnAttr()) {
    auto global =
        symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, logpdfAttr);
    if (!global)
      return emitOpError("'") << logpdfAttr.getValue()
                              << "' does not reference a valid global funcOp";
  }

  return success();
}

LogicalResult InferOp::verify() {
  bool hasHMC = getHmcConfig().has_value();
  bool hasNUTS = getNutsConfig().has_value();

  if (hasHMC + hasNUTS != 1) {
    return emitOpError(
        "Exactly one of hmc_config or nuts_config must be specified");
  }

  if (!getFnAttr() && !getLogpdfFnAttr()) {
    return emitOpError("one of `fn` or `logpdf_fn` must be specified");
  }

  if (getFnAttr() && getLogpdfFnAttr()) {
    return emitOpError("specifying both `fn` and `logpdf_fn` is unsupported");
  }

  if (getLogpdfFnAttr() && !getInitialPosition()) {
    return emitOpError(
        "custom logpdf mode requires `initial_position` to be provided");
  }

  return success();
}
