//===- EnzymeOps.h - Enzyme dialect ops -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEOPS_H
#define ENZYMEOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "Dialect/EnzymeAttributeInterfaces.h.inc"
#include "Dialect/EnzymeEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/EnzymeAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/EnzymeOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Dialect/EnzymeOps.h.inc"

// #include "Dialect/EnzymeTypes.h.inc"

#endif // ENZYMEOPS_H
