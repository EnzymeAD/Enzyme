//=- MatmulInternal.h - declarations shared by the matmul sources ---------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MATMUL_INTERNAL_H
#define POSEIDON_MATMUL_INTERNAL_H

#include "HostDispatch.h"
#include "Matmul.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>

namespace poseidon {

struct CostModelState;
struct Options;
class Session;

// An instruction of the product, where an error about it is reported.
inline const llvm::Instruction &anchorOf(const AbstractMatmul &m) {
  if (m.scalarLoop.fma)
    return *m.scalarLoop.fma;
  if (m.hostGemm && m.hostGemm->cStore)
    return *m.hostGemm->cStore;
  return *m.footprint.front();
}

// A multiply-add: llvm.fma / llvm.fmuladd (`fmul` null), or a contractable
// fadd of a contractable fmul; `acc` is the addend.
struct FMAMatch {
  llvm::Instruction *fma;
  llvm::Instruction *fmul;
  llvm::Value *a, *b, *acc;
};
std::optional<FMAMatch> matchFMA(llvm::Value *V);

// Whether `fma` and `phi` are each other's only users inside `L`.
bool isClosedFmaPhiCycle(const llvm::Loop *L, llvm::Instruction *fma,
                         llvm::PHINode *phi);

std::string mmaShapeSuffix(unsigned M, unsigned N, unsigned K);
std::string inKernelCostClass(const SchemeVariant &v);
std::optional<double> priceInKernelRaiseFromMeasuredRow(
    Session &S, const AbstractMatmul &m, const std::string &cls, unsigned tileM,
    unsigned tileN, unsigned tileK, double baselinePerMac,
    const std::string &what);

struct WmmaTarget {
  unsigned M, N, K;
  FPKind inputPrec;
  FPKind accPrec;
};
const llvm::SmallVector<WmmaTarget, 16> &getAvailableWmmaTargets(Session &S);

// What the matrix-product pipeline records over a compile.
struct MatmulState {
  // The GEMM note of each optimized body clone, keyed by the clone.
  NoteMap<GemmBodyNote> GemmBodies;
  // In the order the sites were split, so that of two sites of one wrapper the
  // same one is recorded last in every compile.
  llvm::MapVector<const llvm::Function *, PendingGemm> PendingGemms;

  // The tensor-core shapes the cost model prices, and the warnings already
  // given.
  std::optional<llvm::SmallVector<WmmaTarget, 16>> AvailableWmmaTargets;
  std::set<const void *> WarnedRectangularRows;
  std::set<const void *> WarnedSquareNativeRows;
  std::set<std::string> ReportedDirectRows;
  std::set<std::string> ReportedInKernelRows;
  bool LegacyMemoEntriesNoted = false;
};
bool inKernelRaiseFits(const Options &O, llvm::Function *F,
                       const AbstractMatmul &m, const MatmulUnit::Option &opt,
                       uint64_t existingBytes, llvm::StringRef className);

// The arithmetic of one candidate, as the matrix accuracy model simulates it.
struct MatmulAccuracyQuery {
  enum class Model {
    // Operands rounded to inputPrec, or to inputMantBits significand bits when
    // that is nonzero, every multiply-accumulate rounded to accPrec, k-tiles
    // of orderTileK folded into the accumulator when that is nonzero; the
    // operands' smallest magnitude has to lie in the range of exponentPrec
    // (Invalid = inputPrec).
    Rounded,
    // __poseidon_tcec_dgemm in compute `mode`, on cuMpSGEMM.
    TcecLibrary,
    // __poseidon_ozaki_dgemm with `moduli` moduli.
    OzakiII,
  };
  Model model = Model::Rounded;
  FPKind inputPrec = FPKind::Invalid;
  FPKind accPrec = FPKind::Invalid;
  FPKind exponentPrec = FPKind::Invalid;
  unsigned orderTileK = 0;
  unsigned inputMantBits = 0;
  unsigned mode = 0;
  unsigned moduli = 0;
};

struct MatmulAccuracy {
  double accuracyCost = 0.0;
  double domainError = 0.0;
};

// The accuracy of each query on the product `m`, all on the same sampled
// operands. `confidence` is the fraction of the sampled inputs the returned
// domain error bounds (the percentile it is read off).
void evaluateMatmulAccuracy(Session &S, const AbstractMatmul &m,
                            const MatmulProfile &prof, double confidence,
                            unsigned sampleLogBits,
                            llvm::ArrayRef<MatmulAccuracyQuery> queries,
                            llvm::MutableArrayRef<MatmulAccuracy> results);

} // namespace poseidon
#endif // POSEIDON_MATMUL_INTERNAL_H
