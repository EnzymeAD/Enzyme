//===- ImpulseUtils.h - Utilities for Impulse dialect passes -----* C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_IMPULSE_UTILS_H
#define ENZYME_MLIR_INTERFACES_IMPULSE_UTILS_H

#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "CloneFunction.h"
#include "EnzymeLogic.h"

#include <functional>

namespace mlir {
namespace impulse {

enum class ImpulseMode {
  Call = 0,
  Simulate = 1,
  Generate = 2,
  Regenerate = 3,
};

class ImpulseUtils {
public:
  FunctionOpInterface newFunc;

  ImpulseMode mode;
  FunctionOpInterface oldFunc;
  IRMapping originalToNewFn;
  std::map<Operation *, Operation *> originalToNewFnOps;

  Block *initializationBlock;

  ImpulseUtils(FunctionOpInterface newFunc_, FunctionOpInterface oldFunc_,
               IRMapping &originalToNewFn_,
               std::map<Operation *, Operation *> &originalToNewFnOps_,
               ImpulseMode mode_)
      : newFunc(newFunc_), mode(mode_), oldFunc(oldFunc_),
        originalToNewFn(originalToNewFn_),
        originalToNewFnOps(originalToNewFnOps_),
        initializationBlock(&*(newFunc.getFunctionBody().begin())) {}

  static ImpulseUtils *CreateFromClone(FunctionOpInterface toeval,
                                       ImpulseMode mode,
                                       int64_t positionSize = -1,
                                       int64_t constraintSize = -1);
};

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_IMPULSE_UTILS_H
