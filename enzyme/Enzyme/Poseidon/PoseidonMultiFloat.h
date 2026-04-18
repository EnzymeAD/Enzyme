//=- PoseidonMultiFloat.h - Double-single arithmetic for Poseidon ---------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_POSEIDON_MULTIFLOAT_H
#define ENZYME_POSEIDON_MULTIFLOAT_H

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"

struct DSValue {
  llvm::Value *hi;
  llvm::Value *lo;
};

DSValue emitTwoSum(llvm::IRBuilder<> &B, llvm::Value *a, llvm::Value *b);
DSValue emitFastTwoSum(llvm::IRBuilder<> &B, llvm::Value *a, llvm::Value *b);
DSValue emitTwoProdFMA(llvm::IRBuilder<> &B, llvm::Value *a, llvm::Value *b);

DSValue emitDSAdd(llvm::IRBuilder<> &B, DSValue x, DSValue y);
DSValue emitDSSub(llvm::IRBuilder<> &B, DSValue x, DSValue y);
DSValue emitDSMul(llvm::IRBuilder<> &B, DSValue x, DSValue y);
DSValue emitDSDiv(llvm::IRBuilder<> &B, DSValue x, DSValue y);
DSValue emitDSSqrt(llvm::IRBuilder<> &B, DSValue x);
DSValue emitDSNeg(llvm::IRBuilder<> &B, DSValue x);

DSValue emitF64ToDS(llvm::IRBuilder<> &B, llvm::Value *f64val);
llvm::Value *emitDSToF64(llvm::IRBuilder<> &B, DSValue ds);

void applyMultiFloat(
    llvm::ArrayRef<llvm::Instruction *> instsToChange,
    const llvm::SmallPtrSetImpl<llvm::Instruction *> &allChanged,
    llvm::DenseMap<llvm::Value *, llvm::Value *> *restoredValues = nullptr);

#endif
