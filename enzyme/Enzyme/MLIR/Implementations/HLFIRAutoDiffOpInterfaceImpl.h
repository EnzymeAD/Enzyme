//===- HLFIRAutoDiffOpInterfaceImpl.h - HLFIR AD registration ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registration for the Enzyme autodiff interface external models on Flang's
// HLFIR dialect (hlfir.matmul, hlfir.sum, ...). This lives in a separate,
// Flang-dependent library so the core Enzyme-MLIR build does not gain a Flang
// dependency; it is linked only into tools that already carry HLFIR
// (fir-enzyme-opt). See PLAN_flang_enzyme_mlir.md, Tier 1.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_IMPLEMENTATIONS_HLFIRAUTODIFFOPINTERFACEIMPL_H
#define ENZYME_MLIR_IMPLEMENTATIONS_HLFIRAUTODIFFOPINTERFACEIMPL_H

#include <memory>

namespace mlir {
class DialectRegistry;
class Pass;
namespace enzyme {
// Attaches the Enzyme AutoDiffTypeInterface to hlfir.expr and the forward/
// reverse autodiff op interfaces to the differentiable hlfir.* intrinsics.
void registerHLFIRDialectAutoDiffInterface(DialectRegistry &registry);

// Attaches the Enzyme AutoDiffTypeInterface to !fir.ref (active memory) and the
// active-memory-identity autodiff models to the FIR/HLFIR memory ops
// (fir.load/store/alloca, hlfir.declare/assign), so whole by-reference Fortran
// functions can be differentiated.
void registerFIRDialectAutoDiffInterface(DialectRegistry &registry);

// A pass that rewrites Fortran differentiation-hook calls
// (fir.call @...f__enzyme_fwddiff / f__enzyme_autodiff) into enzyme.fwddiff /
// enzyme.autodiff ops, parsing enzyme_{const,dup,dupnoneed,out} activity
// markers. Mirrors HandleAutoDiff/getMetadataName in Enzyme.cpp (the LLVM path).
std::unique_ptr<Pass> createHLFIRLowerEnzymeCallsPass();
void registerHLFIRLowerEnzymeCallsPass();
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_IMPLEMENTATIONS_HLFIRAUTODIFFOPINTERFACEIMPL_H
