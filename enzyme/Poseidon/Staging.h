//=- Staging.h - shared-memory staging narrowing --------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A kernel that stages FP64 through addrspace(3) scratch pays for the wide
// slot even after the solve narrows the arithmetic that reads it, so the
// narrowings below run after materialization. The pricing clone must run the
// same ones in the same order or the DP charges a candidate for traffic the
// emitted kernel never performs, which is why both sites go through
// applyStagingNarrowing rather than calling the arms themselves.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_STAGING_H
#define POSEIDON_STAGING_H

#include "llvm/IR/Function.h"

namespace poseidon {

// Narrow addrspace(3) staging buffers whose FP64 values are only ever consumed
// at FP32: stride K -> K/2, `load float` + fpext, fptrunc at the stores. Fires
// only when every consumer of a staged load rounds to FP32 (fptrunc or fcmp),
// which keeps the transform value-preserving.
bool narrowSharedStaging(llvm::Function &F);

// df64 analog: when every staged load is consumed by an emitF64ToDS Dekker
// split, store the {hi,lo} pair once at the staging store (hi@+0, lo@+4 in the
// same 8-byte slot) and turn each split into a pair load.
bool narrowSharedStagingDS(llvm::Function &F);

// The arms in the order every site must apply them: the FP32 narrowing, then
// the df64 one. `announce` prints the per-arm line the solve log carries. True
// if any arm changed F.
bool applyStagingNarrowing(llvm::Function &F, bool announce);

} // namespace poseidon
#endif // POSEIDON_STAGING_H
