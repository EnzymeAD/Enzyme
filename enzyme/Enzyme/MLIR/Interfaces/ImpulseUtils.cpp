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
