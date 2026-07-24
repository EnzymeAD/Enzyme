//===- fir-enzyme-opt.cpp - fir-opt with Enzyme-MLIR built in -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is `fir-opt` with Enzyme-MLIR statically linked in: a single MlirOptMain
// binary that registers both the FIR/HLFIR dialects+passes (exactly as Flang's
// fir-opt does) and the Enzyme dialect, passes, and autodiff interfaces.
//
//   flang-new -fc1 -emit-hlfir foo.f90 -o foo.hlfir
//   fir-enzyme-opt --enzyme foo.hlfir -o foo.diff.hlfir
//
// This is the bring-up / FileCheck path for differentiating Fortran array
// intrinsics *while they are still first-class `hlfir.*` ops* (see
// PLAN_flang_enzyme_mlir.md). It is the sibling of enzyme-fir-plugin.cpp: the
// plugin is the right vehicle when the host fir-opt shares a single libMLIR/
// libLLVM with the plugin (a shared-library LLVM build); this combined tool is
// the right vehicle for a fully-static LLVM build, where loading a second
// static copy of LLVM into fir-opt would double-register cl::opt options. Both
// share the same Enzyme registration entry points.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "flang/Optimizer/Passes/Pipelines.h"
#include "flang/Optimizer/Support/InitFIR.h"

#include "Dialect/Dialect.h"
#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Implementations/HLFIRAutoDiffOpInterfaceImpl.h"
#include "Passes/Passes.h"

using namespace mlir;

int main(int argc, char **argv) {
  // FIR/HLFIR side, mirroring flang/tools/fir-opt/fir-opt.cpp.
  fir::support::registerMLIRPassesForFortranTools();
  fir::registerFlangPipelinePasses();

  // Enzyme passes (`enzyme`, `enzyme-wrap`, ...) plus the HLFIR Fortran-hook
  // lowering pass (`enzyme-lower-fortran-calls`).
  mlir::enzyme::registerenzymePasses();
  mlir::enzyme::registerHLFIRLowerEnzymeCallsPass();

  DialectRegistry registry;
  fir::support::registerDialects(registry);
  registry.insert<mlir::memref::MemRefDialect>();
  fir::support::addFIRExtensions(registry);

  // Enzyme dialect + autodiff interface external models for the upstream
  // dialects Enzyme differentiates through. The FIR/HLFIR autodiff models are
  // attached separately (see registerFIRDialectAutoDiffInterface /
  // registerHLFIRDialectAutoDiffInterface) once those flang-dependent
  // implementations are linked in.
  registry.insert<mlir::enzyme::EnzymeDialect>();
  enzyme::registerCoreDialectAutodiffInterfaces(registry);

  return failed(MlirOptMain(
      argc, argv, "FIR + Enzyme modular optimizer driver\n", registry));
}
