//===- ProbProgUtils.cpp - Utilities for probprog interfaces
//--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interfaces/ProbProgUtils.h"
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
using namespace mlir::enzyme;

Value mlir::enzyme::MProbProgUtils::getTrace() {
  if (!trace) {
    OpBuilder builder(initializationBlock, initializationBlock->begin());
    auto initTraceOp = builder.create<enzyme::InitTraceOp>(
        (initializationBlock->rbegin())->getLoc(),
        enzyme::TraceType::get(initializationBlock->begin()->getContext()));
    trace = initTraceOp.getTrace();
  }
  return trace;
}

MProbProgUtils *MProbProgUtils::CreateFromClone(FunctionOpInterface toeval,
                                                MProbProgMode mode) {
  if (toeval.getFunctionBody().empty()) {
    llvm::errs() << toeval << "\n";
    llvm_unreachable("Creating MProbProgUtils from empty function");
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
  case MProbProgMode::Call:
    suffix = "call";
    OperandTypes.append(originalInputs.begin(), originalInputs.end());
    ResultTypes.append(originalResults.begin(), originalResults.end());
    break;
  case MProbProgMode::Generate:
    suffix = "generate";
    OperandTypes.push_back(enzyme::ConstraintType::get(toeval.getContext()));
    OperandTypes.append(originalInputs.begin(), originalInputs.end());
    ResultTypes.push_back(enzyme::TraceType::get(toeval.getContext()));
    ResultTypes.push_back(RankedTensorType::get({}, builder.getF64Type()));
    ResultTypes.append(originalResults.begin(), originalResults.end());
    break;
  case MProbProgMode::Simulate:
    suffix = "simulate";
    OperandTypes.append(originalInputs.begin(), originalInputs.end());
    ResultTypes.push_back(enzyme::TraceType::get(toeval.getContext()));
    ResultTypes.push_back(RankedTensorType::get({}, builder.getF64Type()));
    ResultTypes.append(originalResults.begin(), originalResults.end());
    break;
  default:
    llvm_unreachable("Invalid MProbProgMode\n");
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

  if (mode == MProbProgMode::Generate) {
    Block &entry = NewF.getFunctionBody().front();
    entry.insertArgument(0u, enzyme::ConstraintType::get(toeval.getContext()),
                         toeval.getLoc());
  }

  return new MProbProgUtils(NewF, toeval, originalToNew, originalToNewOps,
                            mode);
}
