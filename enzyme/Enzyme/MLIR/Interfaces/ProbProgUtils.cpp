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
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/BreadthFirstIterator.h"

using namespace mlir;
using namespace mlir::enzyme;

MProbProgUtils *MProbProgUtils::CreateFromClone(FunctionOpInterface toeval,
                                                MProbProgMode mode) {
  if (toeval.getFunctionBody().empty()) {
    llvm::errs() << toeval << "\n";
    llvm_unreachable("Creating MProbProgUtils from empty function");
  }

  std::string suffix;

  switch (mode) {
  case MProbProgMode::Call:
    suffix = "call";
    break;
  case MProbProgMode::Generate:
    suffix = "generate";
    break;
  case MProbProgMode::Simulate:
    suffix = "simulate";
    break;
  default:
    llvm_unreachable("Invalid MProbProgMode\n");
  }

  OpBuilder builder(toeval.getContext());

  auto NewF = cast<FunctionOpInterface>(toeval->cloneWithoutRegions());
  SymbolTable::setSymbolName(NewF, toeval.getName().str() + "." + suffix);
  NewF.setType(toeval.getFunctionType());

  Operation *parent = toeval->getParentWithTrait<OpTrait::SymbolTable>();
  SymbolTable table(parent);
  table.insert(NewF);

  IRMapping originalToNew;
  std::map<Operation *, Operation *> originalToNewOps;
  cloneInto(&toeval.getFunctionBody(), &NewF.getFunctionBody(), originalToNew,
            originalToNewOps);

  return new MProbProgUtils(NewF, toeval, originalToNew, originalToNewOps,
                            mode);
}
