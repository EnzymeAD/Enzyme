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

#include "llvm/ADT/STLFunctionalExtras.h"
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

// Same two transforms for a buffer that reaches the rewritten function as a
// POINTER PARAMETER. The arms above root at the global's addrspacecast and so
// only see GEPs hanging off that cast inside F; a kernel that declares its
// __shared__ scratch in the kernel and hands it to an annotated `noinline` body
// as a `double *` has every element GEP hanging off an Argument instead.
// `Proxy` is the function the site-argument mapping was recorded against (F
// itself at materialization, the original site function during pricing).
// `speculative` relaxes the value-preserving gate for PRICING ONLY: see
// -poseidon-narrow-staging-speculative.
bool narrowSharedStagingParam(llvm::Function &F,
                              llvm::Function *Proxy = nullptr,
                              bool speculative = false);
bool narrowSharedStagingParamDS(llvm::Function &F,
                                llvm::Function *Proxy = nullptr,
                                bool speculative = false);

// Which parameter arm applyStagingNarrowing runs. Both arms self-gate on the
// reader set, so the materializer runs Both; a pricing clone must pick one,
// because narrowing an expansion candidate's operand buffer to a single float
// would make its `lo` limb identically zero and price the expansion as free.
enum class ParamArm { None, Both, FP32, DS };

// The arms in the order every site must apply them: the global-rooted FP32
// narrowing, the parameter arms `param` selects, the caller's own step (df64
// parameter-array staging, recorded at materialization and priced on the
// clone), then the global-rooted df64 narrowing. `announce` prints the per-arm
// line the solve log carries. True if any arm changed F; `*narrowedParam`
// reports whether a parameter arm fired, which is what the speculative-pricing
// soundness check reads.
bool applyStagingNarrowing(
    llvm::Function &F, bool announce,
    llvm::function_ref<void(llvm::Function &)> between = {},
    ParamArm param = ParamArm::None, llvm::Function *paramProxy = nullptr,
    bool paramSpeculative = false, bool *narrowedParam = nullptr);

} // namespace poseidon
#endif // POSEIDON_STAGING_H
