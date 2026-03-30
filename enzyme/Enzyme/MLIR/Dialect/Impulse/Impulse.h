//===- Impulse.h - Impulse dialect --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_IMPULSE_H
#define ENZYME_IMPULSE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "Dialect/Impulse/ImpulseEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/Impulse/ImpulseAttributes.h.inc"

#include "Dialect/Impulse/ImpulseOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Dialect/Impulse/ImpulseOps.h.inc"

#endif // ENZYME_IMPULSE_H
