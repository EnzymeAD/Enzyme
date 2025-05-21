//===- ProbProgUtils.h - Utilities for probprog interfaces -------* C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_PROBPROG_UTILS_H
#define ENZYME_MLIR_INTERFACES_PROBPROG_UTILS_H

#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "CloneFunction.h"
#include "EnzymeLogic.h"

#include <functional>

namespace mlir {
namespace enzyme {

class MProbProgUtils {
public:
  FunctionOpInterface newFunc;

  MProbProgMode mode;
  FunctionOpInterface oldFunc;
  IRMapping originalToNewFn;
  std::map<Operation *, Operation *> originalToNewFnOps;

private:
  Block *initializationBlock;

public:
  MProbProgUtils(FunctionOpInterface newFunc_, FunctionOpInterface oldFunc_,
                 IRMapping &originalToNewFn_,
                 std::map<Operation *, Operation *> &originalToNewFnOps_,
                 MProbProgMode mode_)
      : newFunc(newFunc_), mode(mode_), oldFunc(oldFunc_),
        originalToNewFn(originalToNewFn_),
        originalToNewFnOps(originalToNewFnOps_),
        initializationBlock(&*(newFunc.getFunctionBody().begin())) {}

  void processSampleOp(enzyme::SampleOp sampleOp, OpBuilder &b,
                       SymbolTableCollection &symbolTable);

  static MProbProgUtils *CreateFromClone(FunctionOpInterface toeval,
                                         MProbProgMode mode);
};

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_PROBPROG_UTILS_H
