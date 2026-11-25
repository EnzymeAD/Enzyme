//===- Expansion.h - floating-point expansions over FP32 ------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_EXPANSION_H
#define POSEIDON_EXPANSION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"

namespace poseidon {

struct Options;

// An n-component FP32 expansion, n = 2, 3, 4. x[0] is the most significant
// limb; the value is the unevaluated sum of the limbs. Every emitter below
// returns a value whose limbs are normalized (|x[i+1]| <= ulp(x[i])/2), except
// the square root of an operand with a zero leading limb, which is the operand.
// The emitters emit the routines of Core/ExpansionRoutines.h, operation for
// operation.
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
// a * a: the square routine for n >= 3, the product for n = 2.
ExpansionValue emitExpansionSqr(llvm::IRBuilder<> &B, const ExpansionValue &a);
ExpansionValue emitExpansionDiv(llvm::IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b);
ExpansionValue emitExpansionSqrt(llvm::IRBuilder<> &B, const ExpansionValue &a);
ExpansionValue emitExpansionNeg(llvm::IRBuilder<> &B, const ExpansionValue &a);
// a * b + c, for llvm.fma and llvm.fmuladd: the fused routine for n = 2, the
// product and then the sum for n >= 3.
ExpansionValue emitExpansionFMA(llvm::IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b,
                                const ExpansionValue &c);
// a * a + c, for a fused multiply-add whose two factors are one value.
ExpansionValue emitExpansionFMASqr(llvm::IRBuilder<> &B,
                                   const ExpansionValue &a,
                                   const ExpansionValue &c);

// Boundary converters. Components are always F32; the surrounding program's
// type may be F32 or F64.
ExpansionValue emitToExpansion(llvm::IRBuilder<> &B, llvm::Value *fpval,
                               unsigned n);
llvm::Value *emitExpansionToFP(llvm::IRBuilder<> &B, const ExpansionValue &v,
                               llvm::Type *targetTy);

// The libm function a call to `name` computes, without the f or l suffix of
// its float and long double variants: "fmax" for fmaxf and for __nv_fmax.
llvm::StringRef libmBaseName(llvm::StringRef name);

// Materialize `instsToChange` as an n-component expansion and erase them, with
// the FP64 PHIs the expansion carries. `restoredValues`, if given, receives
// each erased value whose users now read another, with that value; `limbs`, if
// given, receives the limbs of each converted instruction, in order.
void applyExpansion(
    unsigned n, llvm::ArrayRef<llvm::Instruction *> instsToChange,
    const llvm::SmallPtrSetImpl<llvm::Instruction *> &allChanged,
    llvm::DenseMap<llvm::Value *, llvm::Value *> *restoredValues = nullptr,
    llvm::SmallVectorImpl<llvm::Value *> *limbs = nullptr);

// The Dekker split of an FP64 value into FP32 limbs, as emitToExpansion emits
// it for a pair:
//   hi = fptrunc V ; hib = fpext hi ; los = fsub V, hib ; lo = fptrunc los
// where hib and los are read by the split alone.
struct DekkerSplit {
  llvm::FPTruncInst *hi, *lo;
  llvm::FPExtInst *hib;
  llvm::BinaryOperator *los;
  // Makes the readers of hi and lo read hiV and loV, and erases the split.
  void replaceWith(llvm::Value *hiV, llvm::Value *loV);
};

// The Dekker splits that read V, one per FP32 rounding of V, into `splits`;
// false if there is none or V has another reader, other than an fcmp when
// `allowFCmp`.
bool matchDekkerSplits(llvm::Value *V, bool allowFCmp,
                       llvm::SmallVectorImpl<DekkerSplit> &splits);

// Cancel the join/split roundtrip a df64 conversion leaves on an expansion
// restore; `foldedLimbs` collects the limbs the fold exposed.
bool foldDSPairRoundtrip(llvm::Function &F,
                         llvm::SmallVectorImpl<llvm::Value *> *foldedLimbs,
                         bool print);

// Whether expansion candidates may be proposed for F: false when this LLVM
// does not round F's floating-point operations as written.
bool expansionsRoundedAsWritten(const Options &O, llvm::Function &F,
                                bool hasSubgraphs);

} // namespace poseidon
#endif // POSEIDON_EXPANSION_H
