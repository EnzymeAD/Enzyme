//===- Subgraphs.h - the FP subgraphs of a site ---------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The optimization units of a site: the connected components of its
// optimizable floating-point instructions, with the profile attached to their
// nodes, pruned, split at bottlenecks and merged where they share a staging
// buffer.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SUBGRAPHS_H
#define POSEIDON_SUBGRAPHS_H

#include "Model/FPNode.h"
#include "Profile/ProfileRead.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace llvm {
class Function;
} // namespace llvm

namespace poseidon {

struct Options;
struct AbstractMatmul;

// The FP subgraphs of F, flood-filled over its optimizable instructions, and
// a node for every value they read or compute. Under loose coverage the
// instructions `profileMap` records no execution of are in no subgraph.
void buildSubgraphs(
    const Options &O, llvm::Function &F, const ProfileMap &profileMap,
    const llvm::SmallVectorImpl<AbstractMatmul> &abstractMatmuls,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    llvm::SmallVectorImpl<Subgraph> &subgraphs);

// Sets the profiled sensitivity, gradient, execution count and value ranges
// of the subgraphs' nodes.
void attachProfile(
    const Options &O, const ProfileMap &profileMap,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    llvm::ArrayRef<Subgraph> subgraphs);

// The site's optimization units: the subgraphs after aggressive DCE, splitting
// and the shared-staging merge.
void shapeSubgraphs(
    const Options &O, llvm::Function &F, const ProfileMap &profileMap,
    const std::unordered_set<size_t> &zeroGradIdx,
    const llvm::SmallVectorImpl<AbstractMatmul> &abstractMatmuls,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    llvm::SmallVectorImpl<Subgraph> &subgraphs);

} // namespace poseidon
#endif // POSEIDON_SUBGRAPHS_H
