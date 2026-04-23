//===- ImpulseUtils.cpp - Utilities for Impulse dialect passes ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interfaces/ImpulseUtils.h"
#include "Dialect/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

// TODO: this shouldn't depend on specific dialects except Enzyme.
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "CloneFunction.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/BreadthFirstIterator.h"

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
  SmallVector<mlir::Type, 4> OperandTypes;
  SmallVector<mlir::Type, 4> ResultTypes;

  switch (mode) {
  case ImpulseMode::Call:
    suffix = "call";
    OperandTypes.append(originalInputs.begin(), originalInputs.end());
    ResultTypes.append(originalResults.begin(), originalResults.end());
    break;
  case ImpulseMode::Generate:
    suffix = "generate";
    if (positionSize <= 0 || constraintSize < 0) {
      toeval.emitError("Impulse: Unexpected size parameters");
      return nullptr;
    }
    OperandTypes.push_back(
        RankedTensorType::get({1, constraintSize}, builder.getF64Type()));
    OperandTypes.append(originalInputs.begin(), originalInputs.end());
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
    OperandTypes.push_back(
        RankedTensorType::get({1, positionSize}, builder.getF64Type()));
    OperandTypes.append(originalInputs.begin(), originalInputs.end());
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
    OperandTypes.append(originalInputs.begin(), originalInputs.end());
    ResultTypes.push_back(
        RankedTensorType::get({1, positionSize}, builder.getF64Type()));
    ResultTypes.push_back(RankedTensorType::get({}, builder.getF64Type()));
    ResultTypes.append(originalResults.begin(), originalResults.end());
    break;
  default:
    llvm_unreachable("Invalid ImpulseMode\n");
  }

  auto FTy = builder.getFunctionType(OperandTypes, ResultTypes);
  auto NewF = cast<FunctionOpInterface>(toeval->cloneWithoutRegions());
  SymbolTable::setSymbolName(NewF, toeval.getName().str() + "." + suffix);
  NewF.setType(FTy);

  Operation *parent = toeval->getParentWithTrait<OpTrait::SymbolTable>();
  SymbolTable table(parent);
  table.insert(NewF);

  IRMapping originalToNew;
  std::map<Operation *, Operation *> originalToNewOps;
  cloneInto(&toeval.getFunctionBody(), &NewF.getFunctionBody(), originalToNew,
            originalToNewOps);

  if (mode == ImpulseMode::Generate) {
    Block &entry = NewF.getFunctionBody().front();
    entry.insertArgument(
        0u, RankedTensorType::get({1, constraintSize}, builder.getF64Type()),
        toeval.getLoc());
  }

  if (mode == ImpulseMode::Regenerate) {
    Block &entry = NewF.getFunctionBody().front();
    entry.insertArgument(
        0u, RankedTensorType::get({1, positionSize}, builder.getF64Type()),
        toeval.getLoc());
  }

  return new ImpulseUtils(NewF, toeval, originalToNew, originalToNewOps, mode);
}
