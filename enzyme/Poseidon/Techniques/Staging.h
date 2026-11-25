//===- Staging.h - shared-memory staging narrowing ------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Narrowing of FP64 values staged through addrspace(3) memory once the
// arithmetic that reads them is narrowed. Pricing and materialization apply
// the same narrowings in the same order, through applyStagingNarrowing, so
// that a candidate is charged for the traffic the emitted kernel performs.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_STAGING_H
#define POSEIDON_STAGING_H

#include "llvm/IR/Function.h"

namespace poseidon {

struct Options;
class Session;

// The df64 narrowing of a staging buffer that reaches F as a pointer
// parameter, which the narrowing of addrspace(3) globals does not see: each
// Dekker split of a staged load becomes a pair load and each store stores the
// pair. `Proxy` is the function the site arguments were recorded against: F at
// materialization, the site function when pricing a clone. `announce` prints
// each narrowed buffer under -poseidon-print.
bool narrowSharedStagingParamDS(Session &S, llvm::Function &F,
                                llvm::Function &Proxy, bool announce);

// Which parameter transforms applyStagingNarrowing runs under
// -poseidon-narrow-param-staging. Each declines a buffer whose readers it does
// not fit, so the materializer runs Both; a pricing clone runs one, because
// narrowing an expansion's operand buffer to one float zeroes its `lo` limb
// and prices the expansion as free.
enum class ParamArm { Both, FP32, DS };

// The arms in the order every site must apply them: the FP32 narrowing, the
// parameter arms `param` selects, then the df64 narrowing. `announce` prints
// the per-arm line the solve log carries. True if any arm changed F.
bool applyStagingNarrowing(Session &S, llvm::Function &F, bool announce,
                           ParamArm param, llvm::Function &paramProxy);

} // namespace poseidon
#endif // POSEIDON_STAGING_H
