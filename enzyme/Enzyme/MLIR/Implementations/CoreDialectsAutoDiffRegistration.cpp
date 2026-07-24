//===- CoreDialectsAutoDiffRegistration.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Aggregate entry point registering autodiff external models for every core
// upstream dialect. Isolated in its own TU so that linking it pulls in all the
// per-dialect models (Linalg, NVVM, ...); consumers needing only a subset (e.g.
// the flang -fc1 plugin, which must not pull in Linalg) link the individual
// register*DialectAutoDiffInterface functions instead and skip this TU.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"

#include "mlir/IR/DialectRegistry.h"

void mlir::enzyme::registerCoreDialectAutodiffInterfaces(
    DialectRegistry &registry) {
  enzyme::registerAffineDialectAutoDiffInterface(registry);
  enzyme::registerArithDialectAutoDiffInterface(registry);
  enzyme::registerBuiltinDialectAutoDiffInterface(registry);
  enzyme::registerComplexDialectAutoDiffInterface(registry);
  enzyme::registerLLVMDialectAutoDiffInterface(registry);
  enzyme::registerLLVMExtDialectAutoDiffInterface(registry);
  enzyme::registerNVVMDialectAutoDiffInterface(registry);
  enzyme::registerMathDialectAutoDiffInterface(registry);
  enzyme::registerMemRefDialectAutoDiffInterface(registry);
  enzyme::registerComplexDialectAutoDiffInterface(registry);
  enzyme::registerSCFDialectAutoDiffInterface(registry);
  enzyme::registerCFDialectAutoDiffInterface(registry);
  enzyme::registerLinalgDialectAutoDiffInterface(registry);
  enzyme::registerFuncDialectAutoDiffInterface(registry);
  enzyme::registerTensorDialectAutoDiffInterface(registry);
  enzyme::registerGPUDialectAutoDiffInterface(registry);
  enzyme::registerEnzymeDialectAutoDiffInterface(registry);
}
