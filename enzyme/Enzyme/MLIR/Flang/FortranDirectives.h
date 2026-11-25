//===- FortranDirectives.h - !DIR$ ENZYME directives ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_IMPLEMENTATIONS_FORTRANDIRECTIVES_H
#define ENZYME_MLIR_IMPLEMENTATIONS_FORTRANDIRECTIVES_H

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

namespace mlir {
class Pass;
namespace enzyme {
// Turn the !DIR$ ENZYME directives, which flang lowers onto their subject as
// `enzyme.<keyword>` attributes, into the registrations Enzyme reads (see
// FortranDirectives.cpp).
std::unique_ptr<Pass> createFortranDirectivesPass();
void registerFortranDirectivesPass();

// The attribute that carries the directive `keyword` on its subject.
std::string directiveAttrName(llvm::StringRef keyword);

// Claim the !DIR$ ENZYME (and !$enzyme) directives with flang's plugin
// directives and add the enzyme-fortran-directives pass to flang's pipeline
// (see FlangEnzymeDirectives.cpp).
void registerFlangEnzymeDirectives();
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_IMPLEMENTATIONS_FORTRANDIRECTIVES_H
