//===- LLVMExtAutoDiffOpInterfaceImpl.cpp - Interface external model
//--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the LLVMExt dialect.
//
//===----------------------------------------------------------------------===//

<<<<<<< HEAD
#include "Dialect/LLVMExt/Dialect.h"
=======
#include "Dialect/LLVMExt/LLVMExt.h"
>>>>>>> upstream/main
#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/LLVMExtDerivatives.inc"
} // namespace

void mlir::enzyme::registerLLVMExtDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, llvm_ext::LLVMExtDialect *) {
    registerInterfaces(context);
  });
}
