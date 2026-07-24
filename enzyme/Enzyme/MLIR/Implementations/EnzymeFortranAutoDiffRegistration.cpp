//===- EnzymeFortranAutoDiffRegistration.cpp - shared Fortran registration ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared registration of the Enzyme dialect + the autodiff interface external
// models the `enzyme` differentiation pass needs to differentiate Fortran, i.e.
// HLFIR/FIR plus the upstream dialects flang lowers to (arith, math, complex,
// cf, scf, tensor, func, LLVM). Both delivery vehicles reuse this one entry
// point so they register exactly the same set:
//
//   * the fir-opt MLIR plugin  -- enzyme-fir-plugin.cpp (mlirGetDialectPluginInfo)
//   * the flang -fc1 -load one -- HLFIRFlangPluginRegistration.cpp
//
// It is deliberately a *subset* of registerCoreDialectAutodiffInterfaces: the
// Linalg, NVVM and Affine models are omitted because Fortran HLFIR never
// produces those ops at this stage and -- crucially for the lean plugins --
// neither `flang -fc1` nor `fir-opt` registers those dialects, so their symbols
// are not exported for the plugin to resolve against. The MemRef model is
// likewise omitted: Fortran lowers to !fir.ref, not memref, and MemRef's
// zeroInPlace pulls in linalg.fill (a Linalg symbol the host does not export).
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
}
