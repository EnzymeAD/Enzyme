//===- MatmulInternal.h - declarations shared by the matmul sources -------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MATMUL_INTERNAL_H
#define POSEIDON_MATMUL_INTERNAL_H

#include "Matmul.h"
#include "Model/FPFormat.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"

#include <cstdint>
#include <optional>

namespace poseidon {

struct Options;
class Session;

// The fraction of the sampled inputs each candidate's domain error bounds: the
// percentile of the sampled relative errors a matrix product is scored at.
constexpr double kDefaultConfidence = 0.95;

// An instruction of the product, where an error about it is reported.
inline const llvm::Instruction &anchorOf(const AbstractMatmul &m) {
  if (m.scalarLoop.fma)
    return *m.scalarLoop.fma;
  if (m.hostGemm && m.hostGemm->cStore)
    return *m.hostGemm->cStore;
  return *m.footprint.front();
}

// The multiply-add and the loads of a reduction phi.
struct ReductionCycle {
  llvm::Instruction *fma;
  llvm::Instruction *fmul;
  llvm::LoadInst *aLoad, *bLoad;
};
// A header phi of `L` whose `latch` value is a multiply-add of two loads of the
// phi's type into the phi, the two each other's only users inside `L`. On a
// mismatch `why` receives the reason.
std::optional<ReductionCycle>
matchReductionCycle(const llvm::Loop *L, llvm::PHINode &phi,
                    llvm::BasicBlock *latch, llvm::StringRef *why = nullptr);

// Calls `fn` on each innermost loop of `LI`.
void forEachInnermostLoop(llvm::LoopInfo &LI,
                          llvm::function_ref<void(llvm::Loop *)> fn);

// Compute the GemmBodyNote for matmul m inside optimized body F; false
// (note.valid = false) if anything is unsupported.
bool computeGemmBodyNote(const Options &O, llvm::Function &F,
                         const AbstractMatmul &m, GemmBodyNote &note);

// Records the launch descriptor of the library call `v` that the host compile
// rewrites the launch from, fissioning a fused constant-trip body.
void applyLibraryCall(Session &S, const AbstractMatmul &m,
                      const SchemeVariant &v);

// Emit the tensor-core code of the in-kernel direct or split scheme in place
// of a scalar-loop matmul; aborts on any precondition violation.
void materializeInKernelRaise(const AbstractMatmul &m,
                              const MatmulUnit::Option &opt);

// The arithmetic of one candidate, as the matrix accuracy model simulates it.
struct MatmulAccuracyQuery {
  enum class Model {
    // Operands rounded to inputPrec, or to inputMantBits significand bits when
    // that is nonzero, every multiply-accumulate rounded to accPrec, k-tiles
    // of orderTileK folded into the accumulator when that is nonzero; the
    // operands' smallest magnitude has to lie in the range of exponentPrec
    // (Invalid = inputPrec).
    Rounded,
    // __poseidon_tcec_dgemm in compute mode 1 (FP16TCEC), on cuMpSGEMM.
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
  unsigned moduli = 0;
};

struct MatmulAccuracy {
  double accuracyCost = 0.0;
  double domainError = 0.0;
};

// The profiled range of the elements of one operand.
struct CellStat {
  double min = 0.0;
  double max = 0.0;
  uint64_t count = 0;
  double minMag = 0.0; // smallest nonzero |value| observed (0 = unknown)
};

// What the profile says about a product: every element of an operand is drawn
// from that operand's range, and every output is weighted by `gradSum`.
struct MatmulProfile {
  CellStat a, b;
  double gradSum = 0.0;
  uint64_t totalCalls = 0;
};

// None when the profiling run never executed the product.
std::optional<MatmulProfile> loadProfile(const AbstractMatmul &m,
                                         const ProfileMap &scalarProfile);

// The accuracy of each query on the product `m`, all on the same sampled
// operands, with the domain error read off at kDefaultConfidence.
void evaluateMatmulAccuracy(Session &S, const AbstractMatmul &m,
                            const MatmulProfile &prof, unsigned sampleLogBits,
                            llvm::ArrayRef<MatmulAccuracyQuery> queries,
                            llvm::MutableArrayRef<MatmulAccuracy> results);

} // namespace poseidon
#endif // POSEIDON_MATMUL_INTERNAL_H
