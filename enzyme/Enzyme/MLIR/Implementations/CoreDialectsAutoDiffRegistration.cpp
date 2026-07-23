//===- CoreDialectsAutoDiffRegistration.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The aggregate registration entry point that attaches Enzyme's autodiff
// external models for every core upstream dialect. It is deliberately isolated
// in its own translation unit: because it references every per-dialect
// registration (Linalg and NVVM included), linking it forces those models -- and
// their dialect symbols -- into the consumer. Tools that want them all
// (enzymemlir-opt, fir-enzyme-opt, ...) call this; a consumer that only needs a
// subset (the lean `flang -fc1` plugin, which must not pull in Linalg symbols
// flang does not export) instead links the individual
// register*DialectAutoDiffInterface functions it wants and never references this
// TU, so the Linalg/NVVM models are not linked at all.
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
