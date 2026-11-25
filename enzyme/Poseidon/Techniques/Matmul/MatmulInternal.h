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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"

#include <cstdint>
#include <optional>
#include <utility>

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

// Emit the Direct scheme's tensor-core rewrite of a scalar-loop matmul;
// aborts on any precondition violation.
void materializeScalarLoopRaise(const AbstractMatmul &m,
                                const MatmulUnit::Option &opt);

// Emit the split scheme's N-slice rewrite of a scalar-loop matmul (a Veltkamp
// N-slice split, an N(N+1)/2-mma chain per K-block and a scaled readback
// combine) with an F16/BF16/TF32 input slice and an F32 accumulator; aborts
// on any precondition violation.
void materializeOzakiIRaise(const AbstractMatmul &m,
                            const MatmulUnit::Option &opt);

llvm::Type *llvmTypeForFPKind(llvm::LLVMContext &ctx, FPKind k);

// fptrunc/fpext between FP types (no-op when they match); a bitcast when the
// bit widths agree but the LLVM types differ (F32 vs TF32-as-F32).
llvm::Value *emitFPCast(llvm::IRBuilder<> &B, llvm::Value *v, llvm::Type *toTy);

llvm::GlobalVariable *getOrCreateSharedScratch(llvm::Module *M,
                                               const llvm::Twine &name,
                                               llvm::Type *eltTy,
                                               uint64_t numElts);

// Emit a (possibly fused) matrix row/column index: tid[fast] when slow is
// Unknown, otherwise tid[fast] + mult * tid[slow], where `mult` is the extent
// of the fast axis.
llvm::Value *emitTidIndex(llvm::IRBuilder<> &B, llvm::Module *M, TidAxis fast,
                          TidAxis slow, unsigned mult);

// (linearizedThreadId, blockSize) from tid.{x,y,z} and ntid.{x,y,z}.
// `is2DBlock` skips the tid.z / ntid.z reads; only set it when the block's z
// dim is statically 1 (e.g. profileHeader.maxBlockDim[2] == 1).
std::pair<llvm::Value *, llvm::Value *>
emitThreadLinAndBlockSize(llvm::IRBuilder<> &B, llvm::Module *M,
                          bool is2DBlock = false);

// Block-cooperative fill of a (numRows x numCols) tile in `scratchBase` from
// `srcBlockBase` (per-block base, no threadIdx contribution).
// `assumeBlockCoversAll` emits a single conditional instead of the loop and is
// wrong if blockSize < numElts at runtime, so set it only when the profile
// header proves the block covers the tile. Splits the current block; the
// builder ends in the "after" block.
void emitCooperativeScratchFill(
    llvm::IRBuilder<> &B, llvm::Value *scratchBase, llvm::Type *dstTy,
    FPKind dstKind, llvm::Value *srcBlockBase, llvm::Type *srcTy,
    uint64_t numRows, uint64_t numCols, int64_t srcRowStrideByte,
    int64_t srcColStrideByte, int64_t dstRowStrideByte,
    int64_t dstColStrideByte, llvm::Value *threadLin, llvm::Value *blockSize,
    bool assumeBlockCoversAll = false);

// Padded variant: fill a (fillRows x fillCols) tile whose valid source extent
// is (validRows x validCols) at runtime; every element outside it is written an
// exact zero. Zero-fill rather than masking the MMA: wmma fragments are
// warp-collective, so a per-thread guard around the MMA corrupts the whole
// fragment, while zeros in the pad contribute exactly 0 (an uninitialized pad
// would feed 0 * Inf = NaN into every output element).
void emitCooperativeScratchFillPadded(
    llvm::IRBuilder<> &B, llvm::Value *scratchBase, llvm::Type *dstTy,
    FPKind dstKind, llvm::Value *srcBlockBase, llvm::Type *srcTy,
    uint64_t fillRows, uint64_t fillCols, llvm::Value *validRows,
    llvm::Value *validCols, int64_t srcRowStrideByte, int64_t srcColStrideByte,
    int64_t dstRowStrideByte, int64_t dstColStrideByte, llvm::Value *threadLin,
    llvm::Value *blockSize, bool assumeBlockCoversAll = false);

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
