//===- EnzymeToLLVMIRTranslation.h - Enzyme dialect to LLVM IR --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_TARGET_ENZYMETOLLVMIRTRANSLATION_H
#define ENZYME_MLIR_TARGET_ENZYMETOLLVMIRTRANSLATION_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
namespace enzyme {
/// Register the translation of the enzyme dialect's type tree attributes
/// (EnzymeDialect::get*TypeTree*AttrName) to the `!enzyme_type` metadata and
/// `"enzyme_type"` attributes that LLVM Enzyme's type analysis reads. Does
/// nothing if the dialect already has a translation interface.
void registerEnzymeDialectTranslation(DialectRegistry &registry);
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_TARGET_ENZYMETOLLVMIRTRANSLATION_H
