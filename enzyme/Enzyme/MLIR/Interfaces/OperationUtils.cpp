//===- OperationUtils.cpp - Utilities for operation interfaces
//--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interfaces/OperationUtils.h"
#include "Dialect/Ops.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::enzyme;
namespace mlir {
namespace enzyme {
namespace oputils {

bool isReadOnly(Operation *op) {
  // If the op has memory effects, try to characterize them to see if the op
  // is trivially dead here.
  if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    // Check to see if this op either has no effects, or only allocates/read
    // memory.
    SmallVector<MemoryEffects::EffectInstance, 1> effects;
    effectInterface.getEffects(effects);
    if (!llvm::all_of(effects, [op](const MemoryEffects::EffectInstance &it) {
          return isa<MemoryEffects::Read>(it.getEffect());
        })) {
      return false;
    }
  }

  bool isRecursiveContainer =
      op->hasTrait<OpTrait::HasRecursiveMemoryEffects>() ||
      isa<FunctionOpInterface>(op);
  if (isRecursiveContainer) {
    for (Region &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &nestedOp : block)
          if (!isReadOnly(&nestedOp))
            return false;
      }
    }
  }

  return true;
}
} // namespace oputils
} // namespace enzyme
} // namespace mlir
