//===- EnzymeFortranAutoDiffRegistration.cpp - shared Fortran registration ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared by both plugins (enzyme-fir-plugin.cpp and
// HLFIRFlangPluginRegistration.cpp): the Enzyme dialect plus the autodiff
// models for FIR/HLFIR and the upstream dialects flang lowers to.
//
// This is deliberately a subset of registerCoreDialectAutodiffInterfaces. The
// Linalg, NVVM and Affine models are omitted because neither `flang -fc1` nor
// `fir-opt` registers those dialects, so a plugin cannot resolve their symbols
// against the host. MemRef is omitted too: Fortran lowers to !fir.ref, and
// MemRef's zeroInPlace pulls in linalg.fill.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Implementations/HLFIRAutoDiffOpInterfaceImpl.h"

#include "Dialect/Dialect.h"

#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

void mlir::enzyme::registerEnzymeFortranInterfaces(DialectRegistry &registry) {
  registry.insert<mlir::enzyme::EnzymeDialect>();
  registerArithDialectAutoDiffInterface(registry);
  registerBuiltinDialectAutoDiffInterface(registry);
  registerComplexDialectAutoDiffInterface(registry);
  registerLLVMDialectAutoDiffInterface(registry);
  registerMathDialectAutoDiffInterface(registry);
  registerSCFDialectAutoDiffInterface(registry);
  registerCFDialectAutoDiffInterface(registry);
  registerFuncDialectAutoDiffInterface(registry);
  registerTensorDialectAutoDiffInterface(registry);
  registerEnzymeDialectAutoDiffInterface(registry);
  // The by-reference model: without it nothing can differentiate a Fortran
  // function, whose arguments and locals are all !fir.ref<T>.
  registerFIRDialectAutoDiffInterface(registry);
}
