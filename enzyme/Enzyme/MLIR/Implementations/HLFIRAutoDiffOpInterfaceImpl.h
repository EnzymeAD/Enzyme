//===- HLFIRAutoDiffOpInterfaceImpl.h - Flang Enzyme registration -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registration entry points for the Flang-dependent Enzyme plumbing. These live
// in a separate, Flang-dependent library so the core Enzyme-MLIR build does not
// gain a Flang dependency; it is bundled only into the Enzyme fir-opt/flang
// plugins (FIREnzyme, FlangEnzymeMLIR), which resolve HLFIR from the host. This
// foundational layer declares the shared Fortran registration and the call-
// lowering pass; the FIR/HLFIR autodiff registration entry points are declared
// alongside their implementations in the later autodiff layers. See
// PLAN_flang_enzyme_mlir.md.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_IMPLEMENTATIONS_HLFIRAUTODIFFOPINTERFACEIMPL_H
#define ENZYME_MLIR_IMPLEMENTATIONS_HLFIRAUTODIFFOPINTERFACEIMPL_H

#include <memory>

namespace mlir {
class DialectRegistry;
class Pass;
namespace enzyme {
// Registers the Enzyme dialect and the subset of autodiff interface external
// models needed to differentiate Fortran (HLFIR/FIR) code, into `registry`.
// Shared by both delivery vehicles -- the fir-opt MLIR plugin
// (enzyme-fir-plugin.cpp) and the flang -fc1 -load plugin
// (HLFIRFlangPluginRegistration.cpp) -- so they register exactly the same set.
// A deliberate subset of registerCoreDialectAutodiffInterfaces (see
// EnzymeFortranAutoDiffRegistration.cpp for why Linalg/NVVM/Affine are omitted).
void registerEnzymeFortranInterfaces(DialectRegistry &registry);
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
