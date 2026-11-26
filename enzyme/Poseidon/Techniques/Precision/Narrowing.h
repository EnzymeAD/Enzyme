//===- Narrowing.h - shared staging after a precision change --------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Narrowing of FP64 values staged through shared (addrspace(3)) memory once a
// precision change narrows the arithmetic that reads them. Pricing and
// materialization apply the same narrowings in the same order, through
// narrowSharedStaging, so that a candidate is charged for the traffic the
// emitted kernel performs.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TECHNIQUES_PRECISION_NARROWING_H
#define POSEIDON_TECHNIQUES_PRECISION_NARROWING_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"

namespace poseidon {

class Session;
struct Options;
struct Subgraph;

/// Which parameter transforms narrowSharedStaging runs. Each declines a buffer
/// whose readers it does not fit, so the materializer runs Both; a pricing
/// clone runs one, because narrowing an expansion's operand buffer to one float
/// zeroes its `lo` component and prices the expansion as free.
enum class ParamNarrowing { Both, FP32, Pairs };

/// Narrows the shared staging of F after a precision change, in the order every
/// site must apply the narrowings: the buffers F reaches through the global at
/// one stride to FP32, those it reaches as parameters or through views to the
/// formats `param` selects, then the buffers it reaches through the global to
/// df64 pairs. `paramProxy` is the function the site arguments were recorded
/// against: F at materialization, the site function when pricing a clone.
/// `announce` prints what each narrowing did under -poseidon-print. True if any
/// changed F.
bool narrowSharedStaging(Session &S, llvm::Function &F, bool announce,
                         ParamNarrowing param, llvm::Function &paramProxy);

/// The df64 narrowing of the buffers that reach F as parameters or through
/// views, alone: each Dekker split of a staged load becomes a pair load and
/// each store stores the pair. It preserves values only once every load of the
/// buffer is split, so it runs after each two-component change. `Proxy` is
/// as narrowSharedStaging's `paramProxy`; true if it changed F.
bool narrowToPairs(Session &S, llvm::Function &F, llvm::Function &Proxy,
                   bool announce);

/// Under -poseidon-subgraph-merge-staging, merges every group of the
/// subgraphs of F that touch a common staging buffer (a pointer argument or an
/// addrspace(3) global reached by a load feeding the subgraph or a store
/// consuming it) into one, so that the tier of its precision unit is the
/// buffer's.
void mergeSharedStagingSubgraphs(const Options &O, llvm::Function &F,
                                 llvm::SmallVectorImpl<Subgraph> &subgraphs);

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_PRECISION_NARROWING_H
