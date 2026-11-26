//===- Expansion.h - floating-point expansions over FP32 ------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TECHNIQUES_PRECISION_EXPANSION_H
#define POSEIDON_TECHNIQUES_PRECISION_EXPANSION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"

namespace poseidon {

struct Options;

/// An n-component FP32 expansion, n = 2, 3, 4. x[0] is the most significant
/// component; the value is the unevaluated sum of the components. Every emitter
/// below returns a value whose components are normalized (|x[i+1]| <=
/// ulp(x[i])/2), except the square root of an operand with a zero leading
/// component, which is the operand. The emitters emit the routines of
/// Model/ExpansionRoutines.h, operation for operation.
struct ExpansionValue {
  /// The components, each an F32 value, most significant first.
  llvm::SmallVector<llvm::Value *, 4> x;
  /// The component count.
  unsigned n() const { return x.size(); }
};

/// a + b, at B. The operands have one component count, as has the result.
ExpansionValue emitExpansionAdd(llvm::IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b);
/// a - b, at B.
ExpansionValue emitExpansionSub(llvm::IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b);
/// a * b, at B.
ExpansionValue emitExpansionMul(llvm::IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b);
/// a * a: the square routine for n >= 3, the product for n = 2.
ExpansionValue emitExpansionSqr(llvm::IRBuilder<> &B, const ExpansionValue &a);
/// a / b, at B.
ExpansionValue emitExpansionDiv(llvm::IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b);
/// The square root of a, at B.
ExpansionValue emitExpansionSqrt(llvm::IRBuilder<> &B, const ExpansionValue &a);
/// -a, at B: each component negated.
ExpansionValue emitExpansionNeg(llvm::IRBuilder<> &B, const ExpansionValue &a);
/// a * b + c, for llvm.fma and llvm.fmuladd: the fused routine for n = 2, the
/// product and then the sum for n >= 3.
ExpansionValue emitExpansionFMA(llvm::IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b,
                                const ExpansionValue &c);
/// a * a + c, for a fused multiply-add whose two factors are one value.
ExpansionValue emitExpansionFMASqr(llvm::IRBuilder<> &B,
                                   const ExpansionValue &a,
                                   const ExpansionValue &c);

/// `fpval`, an F32 or F64 value, as an `n`-component expansion, at B.
ExpansionValue emitToExpansion(llvm::IRBuilder<> &B, llvm::Value *fpval,
                               unsigned n);
/// The sum of v's components rounded to `targetTy`, F32 or F64, at B.
llvm::Value *emitExpansionToFP(llvm::IRBuilder<> &B, const ExpansionValue &v,
                               llvm::Type *targetTy);

/// Gives each reader of `I` that `inside` rejects the value `restore` emits
/// right after `I` (after the PHIs, for a PHI); that value, or null when every
/// reader is inside.
llvm::Value *
restoreOutside(llvm::Instruction *I,
               llvm::function_ref<bool(llvm::Instruction *)> inside,
               llvm::function_ref<llvm::Value *(llvm::IRBuilder<> &)> restore);

/// Materialize `instsToChange`, in an order where an operation follows its
/// operands, as an n-component expansion and erase them, with the FP64 PHIs the
/// expansion carries. `restoredValues`, if given, receives each erased value
/// whose users now read another, with that value; `components`, if given,
/// receives the components of each converted instruction, in order. An
/// instruction without a routine is left as written, and named under `print`.
void applyExpansion(
    unsigned n, llvm::ArrayRef<llvm::Instruction *> instsToChange,
    llvm::DenseMap<llvm::Value *, llvm::Value *> *restoredValues,
    llvm::SmallVectorImpl<llvm::Value *> *components, bool print);

/// The Dekker split of an FP64 value into FP32 components, as emitToExpansion
/// emits it for a pair:
///   hi = fptrunc V ; hib = fpext hi ; los = fsub V, hib ; lo = fptrunc los
/// where hib and los are read by the split alone.
struct DekkerSplit {
  /// The leading and the trailing component.
  llvm::FPTruncInst *hi, *lo;
  /// hi widened back to FP64.
  llvm::FPExtInst *hib;
  /// V less hib, in FP64.
  llvm::BinaryOperator *los;
  /// Makes the readers of hi and lo read hiV and loV, and erases the split.
  void replaceWith(llvm::Value *hiV, llvm::Value *loV);
};

/// The Dekker splits that read V, one per FP32 rounding of V, into `splits`;
/// false if there is none or V has another reader, other than an fcmp when
/// `allowFCmp`.
bool matchDekkerSplits(llvm::Value *V, bool allowFCmp,
                       llvm::SmallVectorImpl<DekkerSplit> &splits);

/// Cancels the join/split roundtrip a df64 conversion leaves on an expansion
/// restore of F; whether one was cancelled. `foldedComponents`, if given,
/// collects the components the fold exposed.
bool foldExpansionRoundtrip(
    llvm::Function &F, llvm::SmallVectorImpl<llvm::Value *> *foldedComponents,
    bool print);

/// Whether expansion candidates may be proposed for F: false, with a warning,
/// when this LLVM does not round F's floating-point operations as written, and
/// then an error under -poseidon-strict-mode. Always true for a site with no
/// subgraph.
llvm::Expected<bool> expansionsRoundedAsWritten(const Options &O,
                                                llvm::Function &F,
                                                bool hasSubgraphs);

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_PRECISION_EXPANSION_H
