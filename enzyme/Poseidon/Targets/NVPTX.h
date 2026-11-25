//===- NVPTX.h - the NVPTX instructions of an in-kernel raise -------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The only target with an in-kernel matrix raise (Target::
// hasInKernelMatrixRaise): the WMMA fragment operations, the block barrier,
// the TF32 rounding and the thread geometry the raised code is built from.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TARGETS_NVPTX_H
#define POSEIDON_TARGETS_NVPTX_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class Function;
class IRBuilderBase;
class Module;
class Value;
} // namespace llvm

namespace poseidon {
namespace nvptx {

/// The threads that execute one WMMA fragment operation together.
constexpr unsigned kWarpSize = 32;
/// log2 of kWarpSize.
constexpr unsigned kLog2WarpSize = 5;
static_assert(1u << kLog2WarpSize == kWarpSize);

/// threadIdx on `axis` (0 to 2), read at `B`'s insertion point.
llvm::Value *emitThreadIndex(llvm::IRBuilderBase &B, llvm::Module *M,
                             unsigned axis);
/// blockIdx on `axis` (0 to 2), read at `B`'s insertion point.
llvm::Value *emitBlockIndex(llvm::IRBuilderBase &B, llvm::Module *M,
                            unsigned axis);
/// blockDim on `axis` (0 to 2), read at `B`'s insertion point.
llvm::Value *emitBlockDim(llvm::IRBuilderBase &B, llvm::Module *M,
                          unsigned axis);

/// The barrier of a thread block (__syncthreads) and its operands.
struct Barrier {
  /// The barrier intrinsic, declared in the module.
  llvm::Function *Fn;
  /// The operands a call of `Fn` takes.
  llvm::SmallVector<llvm::Value *, 1> Args;
};
/// The block barrier, declared in `M`.
Barrier declareBarrier(llvm::Module *M);

/// `x`, an f32, rounded to nearest to TF32 and kept in f32 storage, which the
/// TF32 fragment loads read as it is.
llvm::Value *emitRoundToTF32(llvm::IRBuilderBase &B, llvm::Module *M,
                             llvm::Value *x);

/// The fragment operations of one m x n x k shape, through generic pointers: A
/// and D row-major, B row-major or column-major. `operand` and `accumulator`
/// name the element types as the instructions spell them (f16, bf16, tf32,
/// f64; f16, f32, f64).
struct Wmma {
  /// The fragment loads of A and B, the multiply-accumulate, and the store of
  /// D, declared in the module.
  llvm::Function *LoadA, *LoadB, *Mma, *StoreD;
};
/// The fragment operations of the `m` x `n` x `k` shape, declared in `M`. A
/// shape and type combination NVPTX has no intrinsic for is a fatal error.
Wmma declareWmma(llvm::Module *M, unsigned m, unsigned n, unsigned k,
                 llvm::StringRef operand, llvm::StringRef accumulator,
                 bool bRowMajor);

} // namespace nvptx
} // namespace poseidon

#endif // POSEIDON_TARGETS_NVPTX_H
