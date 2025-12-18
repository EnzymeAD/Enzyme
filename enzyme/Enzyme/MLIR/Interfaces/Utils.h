//===- Utils.h - Utilities for gradient interfaces -------* C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace enzyme {
namespace oputils {

const std::set<std::string> &getNonCapturingFunctions();

// Checks if the operation/function has any memory write effects. This enables
// batching specific AD optimiziations(which are triggered only if the primal
// function doesnt modify memory operands)
bool isReadOnly(Operation *op);

// Checks if 2 values v1 and v2 may alias with each other locally
bool mayAlias(Value v1, Value v2);

// check if 2 memory effects' underlying values alias with each other
bool mayAlias(MemoryEffects::EffectInstance &A,
              MemoryEffects::EffectInstance &B);

// check if a memory effect's underlying values alias with a value
bool mayAlias(mlir::MemoryEffects::EffectInstance a, mlir::Value v2);

/// Returns the side effects of an operation(similar to
/// `mlir::getEffectsRecursively`). If the operation has RecursiveMemoryEffects,
/// include all side effects of child operations.
///
/// Also accounts for LLVM and autodiff-specific memory effects which are not
/// captured by the default `mlir::getEffectsRecursively`
bool collectOpEffects(Operation *rootOp,
                      SmallVector<MemoryEffects::EffectInstance> &effects);

// Specialize memory effect collection for a FunctionOpInterface
SmallVector<MemoryEffects::EffectInstance>
collectFnEffects(FunctionOpInterface fnOp);

MemoryEffects::EffectInstance getEffectOfVal(Value val,
                                             MemoryEffects::Effect *effect,
                                             SideEffects::Resource *resource);
} // namespace oputils
} // namespace enzyme
} // namespace mlir
