//===- FIRTypeAnnotations.h - Fortran types for LLVM Enzyme -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_IMPLEMENTATIONS_FIRTYPEANNOTATIONS_H
#define ENZYME_MLIR_IMPLEMENTATIONS_FIRTYPEANNOTATIONS_H

#include <memory>

namespace mlir {
class Pass;
namespace enzyme {
// Attach the Fortran types LLVM IR erases as `enzyme.type`, which become
// !enzyme_type metadata and "enzyme_type" attributes for LLVM Enzyme's type
// analysis (see FIRTypeAnnotations.cpp).
std::unique_ptr<Pass> createFIRTypeAnnotationsPass();
void registerFIRTypeAnnotationsPass();
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_IMPLEMENTATIONS_FIRTYPEANNOTATIONS_H
