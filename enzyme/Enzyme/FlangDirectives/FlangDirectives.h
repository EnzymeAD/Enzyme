//===- FlangDirectives.h - !DIR$ ENZYME for LLVM Enzyme ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The passes of the FlangEnzymeDirectives plugin, which `flang -fc1 -load`s
// into flang's own pipeline. They prepare a Fortran program for LLVM Enzyme
// (which runs later, e.g. in the LTO link), and need nothing of Enzyme-MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_FLANG_DIRECTIVES_H
#define ENZYME_FLANG_DIRECTIVES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace enzyme {

// Turn the !DIR$ ENZYME directives that flang lowered to `fir.directives` on
// their subject into the registrations Enzyme reads (see
// FortranDirectives.cpp).
std::unique_ptr<Pass> createFortranDirectivesPass();
void registerFortranDirectivesPass();

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_FLANG_DIRECTIVES_H
