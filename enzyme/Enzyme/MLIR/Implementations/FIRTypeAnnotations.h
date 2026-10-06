//===- FIRTypeAnnotations.h - Fortran types for LLVM Enzyme ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The enzyme-fir-type-annotations pass (FIRTypeAnnotations.cpp), which the
// FlangEnzymeMLIR plugin adds at flang's FIROptLast extension point.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_IMPLEMENTATIONS_FIRTYPEANNOTATIONS_H
#define ENZYME_MLIR_IMPLEMENTATIONS_FIRTYPEANNOTATIONS_H

#include <memory>

namespace mlir {
class Pass;
namespace enzyme {
std::unique_ptr<Pass> createFIRTypeAnnotationsPass();
void registerFIRTypeAnnotationsPass();
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_IMPLEMENTATIONS_FIRTYPEANNOTATIONS_H
