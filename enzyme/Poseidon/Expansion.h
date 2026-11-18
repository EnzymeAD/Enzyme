//=- Expansion.h - Double-single arithmetic for Poseidon ------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_EXPANSION_H
#define POSEIDON_EXPANSION_H

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"

namespace poseidon {

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

// Boundary converters: components are always F32, the source/target type may
// be F32 or F64.
DSValue emitToDS(llvm::IRBuilder<> &B, llvm::Value *fpval);
llvm::Value *emitDSToFP(llvm::IRBuilder<> &B, DSValue ds, llvm::Type *targetTy);

void applyExpansion(
    llvm::ArrayRef<llvm::Instruction *> instsToChange,
    const llvm::SmallPtrSetImpl<llvm::Instruction *> &allChanged,
    llvm::DenseMap<llvm::Value *, llvm::Value *> *restoredValues = nullptr);

// Wider FP32 expansions (n = 3, 4), parallel to the two-component DSValue path
// and sharing only the EFT primitives. The component sequences reproduce the
// QxW triple/quad-word routines used by mX_real's Sloppy tier, followed by
// mX_real's Normalize<Regular>, so a materialized expansion matches mX_real's
// hand-written code operation for operation.

// An n-component FP32 expansion. x[0] is the most significant limb; the value
// is the unevaluated sum of the limbs. Every emitter below returns a value
// whose limbs are normalized (|x[i+1]| <= ulp(x[i])/2).
struct ExpansionValue {
  llvm::SmallVector<llvm::Value *, 4> x;
  unsigned n() const { return x.size(); }
};

ExpansionValue emitExpansionAdd(llvm::IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b);
ExpansionValue emitExpansionSub(llvm::IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b);
ExpansionValue emitExpansionMul(llvm::IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b);
ExpansionValue emitExpansionDiv(llvm::IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b);
ExpansionValue emitExpansionSqrt(llvm::IRBuilder<> &B, const ExpansionValue &a);
ExpansionValue emitExpansionNeg(llvm::IRBuilder<> &B, const ExpansionValue &a);

// Boundary converters. Components are always F32; the surrounding program's
// type may be F32 or F64.
ExpansionValue emitToExpansion(llvm::IRBuilder<> &B, llvm::Value *fpval,
                               unsigned n);
llvm::Value *emitExpansionToFP(llvm::IRBuilder<> &B, const ExpansionValue &v,
                               llvm::Type *targetTy);

// Materialize as an n-component expansion (n >= 3); mirrors applyExpansion
// but never touches the two-component df64 staging machinery.
void applyExpansion(
    unsigned n, llvm::ArrayRef<llvm::Instruction *> instsToChange,
    const llvm::SmallPtrSetImpl<llvm::Instruction *> &allChanged,
    llvm::DenseMap<llvm::Value *, llvm::Value *> *restoredValues = nullptr);

// Cancel the join/split roundtrip a df64 conversion leaves on an expansion
// restore; `foldedLimbs` collects the limbs the fold exposed.
bool foldDSPairRoundtrip(llvm::Function &F,
                         llvm::SmallVectorImpl<llvm::Value *> *foldedLimbs);

// Drain the limbs the last applyExpansion produced, so getCompCost can re-root
// its cost walk on them.
void takeLastExpansionLimbs(llvm::SmallVectorImpl<llvm::Value *> &out);

// Drain the limbs the last applyExpansion produced, so getCompCost can re-root
// its cost walk on them; separate from takeLastExpansionLimbs because one
// candidate can apply both kinds of change.
void takeLastExpLimbs(llvm::SmallVectorImpl<llvm::Value *> &out);
// Start a fresh limb record; one candidate may apply several expansion changes.
void resetExpLimbs();

} // namespace poseidon
#endif // POSEIDON_EXPANSION_H
