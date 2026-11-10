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

struct Options;
class Session;

// Narrow addrspace(3) staging buffers whose FP64 values are only ever consumed
// at FP32: stride K -> K/2, `load float` + fpext, fptrunc at the stores. Fires
// only when every consumer of a staged load rounds to FP32 (fptrunc or fcmp),
// which keeps the transform value-preserving.
bool narrowSharedStaging(const Options &O, llvm::Function &F);

// df64 analog: when every staged load is consumed by an emitF64ToDS Dekker
// split, store the {hi,lo} pair once at the staging store (hi@+0, lo@+4 in the
// same 8-byte slot) and turn each split into a pair load.
bool narrowSharedStagingDS(const Options &O, llvm::Function &F);

// The same two transforms for a buffer that reaches F as a POINTER PARAMETER:
// the arms above root at the global's addrspacecast and see no GEP of a
// `double *` the kernel hands to an annotated `noinline` body. `Proxy` is the
// function the site arguments were recorded against (F at materialization, the
// site function when pricing a clone).
bool narrowSharedStagingParam(Session &S, llvm::Function &F,
                              llvm::Function *Proxy = nullptr);
bool narrowSharedStagingParamDS(Session &S, llvm::Function &F,
                                llvm::Function *Proxy = nullptr);

// Which parameter arm applyStagingNarrowing runs. Both self-gate on the reader
// set, so the materializer runs Both; a pricing clone runs one, because
// narrowing an expansion's operand buffer to one float zeroes its `lo` limb
// and prices the expansion as free.
enum class ParamArm { None, Both, FP32, DS };

// The arms in the order every site must apply them: the FP32 narrowing, the
// parameter arms `param` selects, then the df64 narrowing. `announce` prints
// the per-arm line the solve log carries. True if any arm changed F.
bool applyStagingNarrowing(Session &S, llvm::Function &F, bool announce,
                           ParamArm param = ParamArm::None,
                           llvm::Function *paramProxy = nullptr);

} // namespace poseidon
#endif // POSEIDON_STAGING_H
