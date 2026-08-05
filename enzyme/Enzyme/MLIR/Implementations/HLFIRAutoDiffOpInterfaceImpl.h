//===- HLFIRAutoDiffOpInterfaceImpl.h - Flang registration -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registration entry points for the Flang-dependent Enzyme plumbing. These live
// in their own library so the core Enzyme-MLIR build does not gain a Flang
// dependency; it is bundled only into the FIREnzyme and FlangEnzymeMLIR
// plugins, which resolve FIR/HLFIR from the host.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_IMPLEMENTATIONS_HLFIRAUTODIFFOPINTERFACEIMPL_H
#define ENZYME_MLIR_IMPLEMENTATIONS_HLFIRAUTODIFFOPINTERFACEIMPL_H

#include <memory>

namespace mlir {
class DialectRegistry;
class Pass;
namespace enzyme {
// Registers the Enzyme dialect and the autodiff models needed to differentiate
// Fortran. Both plugins call this, so they register the same set.
void registerEnzymeFortranInterfaces(DialectRegistry &registry);

// Attaches the AutoDiffTypeInterface to hlfir.expr and the forward and reverse
// models to the differentiable hlfir.* intrinsics.
void registerHLFIRDialectAutoDiffInterface(DialectRegistry &registry);

// Attaches the AutoDiffTypeInterface to !fir.ref and the active-memory models
// to the FIR/HLFIR memory ops (fir.load/store/alloca, hlfir.declare/assign).
void registerFIRDialectAutoDiffInterface(DialectRegistry &registry);

// Rewrites Fortran differentiation-hook calls (f__enzyme_fwddiff /
// f__enzyme_autodiff) into enzyme.fwddiff / enzyme.autodiff ops.
std::unique_ptr<Pass> createHLFIRLowerEnzymeCallsPass();
void registerHLFIRLowerEnzymeCallsPass();
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_IMPLEMENTATIONS_HLFIRAUTODIFFOPINTERFACEIMPL_H
