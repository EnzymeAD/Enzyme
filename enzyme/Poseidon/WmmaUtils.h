//=- WmmaUtils.h - WMMA materialization helpers --------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Low-level WMMA materialization helpers used by the scalar-loop raise
// (RaiseWMMA).
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_WMMA_UTILS_H
#define POSEIDON_WMMA_UTILS_H

#include "Precision.h"
#include "matmul/Matmul.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"

#include <cstdint>
#include <utility>

namespace llvm {
class GlobalVariable;
class LLVMContext;
class Module;
class Type;
class Value;
} // namespace llvm

namespace poseidon {

llvm::Type *llvmTypeForFPKind(llvm::LLVMContext &ctx, FPKind k);

llvm::Intrinsic::ID resolveWmmaIntrinsic(const llvm::Twine &name);

// fptrunc/fpext between FP types (no-op when they match); a bitcast when the
// bit widths agree but the LLVM types differ (F32 vs TF32-as-F32).
llvm::Value *emitFPCast(llvm::IRBuilder<> &B, llvm::Value *v, llvm::Type *toTy);

llvm::GlobalVariable *getOrCreateSharedScratch(llvm::Module *M,
                                               const llvm::Twine &name,
                                               llvm::Type *eltTy,
                                               uint64_t numElts);

// Emit the tid intrinsic for the given axis; aborts on Unknown / Other.
llvm::Value *emitTidIntrinsic(llvm::IRBuilder<> &B, llvm::Module *M,
                              TidAxis axis);

// Emit a (possibly fused) matrix row/column index: tid[fast] when slow is
// Unknown, otherwise tid[fast] + mult * tid[slow], where `mult` is the extent
// of the fast axis (checked against the profile header by the recognizer).
llvm::Value *emitTidIndex(llvm::IRBuilder<> &B, llvm::Module *M, TidAxis fast,
                          TidAxis slow, unsigned mult);

// (linearizedThreadId, blockSize) from tid.{x,y,z} and ntid.{x,y,z}.
// `is2DBlock` skips the tid.z / ntid.z reads; only set it when the block's z
// dim is statically 1 (e.g. profileHeader.maxBlockDim[2] == 1).
std::pair<llvm::Value *, llvm::Value *>
emitThreadLinAndBlockSize(llvm::IRBuilder<> &B, llvm::Module *M,
                          bool is2DBlock = false);

// Block-cooperative zero-init of gv; splits the current block and leaves the
// builder in the "after" block.
void emitParallelZeroInit(llvm::IRBuilder<> &B, llvm::GlobalVariable *gv,
                          llvm::Type *eltTy, uint64_t numElts,
                          llvm::Value *threadLin, llvm::Value *blockSize);

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

} // namespace poseidon
#endif // POSEIDON_WMMA_UTILS_H
