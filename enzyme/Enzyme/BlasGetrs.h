//===- BlasGetrs.h - Derivatives of LAPACK LU solves ------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef ENZYME_BLAS_GETRS_H
#define ENZYME_BLAS_GETRS_H

#include "Utils.h"

namespace getrs {
enum TapeField {
  Trans,
  N,
  NRHS,
  Factors,
  LDA,
  Pivots,
  Solution,
  LDX,
  Valid,
  ShadowLDB,
  ShadowLDA,
  NumFields
};
} // namespace getrs

struct GetrsCache {
  bool Factors;
  bool Pivots;
  bool Solution;
};

// The snapshot contains scalar values, packed factors, pivots and (when needed)
// the primal solution. It never contains shadows of the discrete pivots.
llvm::StructType *getGetrsTapeType(llvm::LLVMContext &, const BlasInfo &);
llvm::Function *getOrInsertGetrsSnapshot(llvm::Module &, llvm::Function *,
                                         const BlasInfo &, bool activeFactors,
                                         GetrsCache);
llvm::Function *getOrInsertGetrsDerivative(llvm::Module &, llvm::Function *,
                                           const BlasInfo &, bool reverse,
                                           bool activeFactors, bool freeTape,
                                           GetrsCache);

#endif
