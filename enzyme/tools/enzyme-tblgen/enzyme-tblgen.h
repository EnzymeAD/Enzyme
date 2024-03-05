//===- enzyme-tblgen.h - Top-Level TableGen headers for Enzyme ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the main function for Enzyme's TableGen.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_TBLGEN_H
#define ENZYME_TBLGEN_H 1

#include <llvm/TableGen/Record.h>

enum ActionType {
  MLIRDerivatives,
  CallDerivatives,
  InstDerivatives,
  BinopDerivatives,
  IntrDerivatives,
  GenBlasDerivatives,
  UpdateBlasDecl,
  UpdateBlasTA,
  GenBlasDiffUse,
  GenHeaderVariables,
};

void emitDiffUse(const llvm::RecordKeeper &recordKeeper, llvm::raw_ostream &os,
                 ActionType intrinsic);

#endif
