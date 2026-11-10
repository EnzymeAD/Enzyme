//===- Subgraphs.h - the FP subgraphs of a site --------------------------===//
//
// The optimization units of a site: the connected components of its
// optimizable floating-point instructions, with the profile attached to their
// nodes, pruned, split at bottlenecks and merged where they share a staging
// buffer.
//===---------------------------------------------------------------------===//
#ifndef POSEIDON_SUBGRAPHS_H
#define POSEIDON_SUBGRAPHS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "Profile/ProfileRead.h"
#include "Types.h"

namespace llvm {
class Function;
} // namespace llvm

namespace poseidon {

struct Options;
struct AbstractMatmul;

// The FP subgraphs of F, flood-filled over its optimizable instructions, and
// a node for every value they read or compute.
void buildSubgraphs(
    const Options &O, llvm::Function &F,
    const llvm::SmallVectorImpl<AbstractMatmul> &abstractMatmuls,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    llvm::SmallVectorImpl<Subgraph> &subgraphs);

// Sets the profiled sensitivity, gradient, execution count and value ranges
// of the subgraphs' nodes.
void attachProfile(
    const Options &O, const std::unordered_map<size_t, ProfileInfo> &profileMap,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    llvm::ArrayRef<Subgraph> subgraphs);

// The site's optimization units: the subgraphs after aggressive DCE, splitting
// and the shared-staging merge.
void shapeSubgraphs(
    const Options &O, llvm::Function &F,
    const std::unordered_map<size_t, ProfileInfo> &profileMap,
    const std::unordered_set<size_t> &zeroGradIdx,
    const llvm::SmallVectorImpl<AbstractMatmul> &abstractMatmuls,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    llvm::SmallVectorImpl<Subgraph> &subgraphs);

void splitSubgraphs(const Options &O,
                    llvm::SmallVectorImpl<Subgraph> &subgraphs);

// Merge every group of subgraphs that touch a common staging buffer (a pointer
// argument or an addrspace(3) global reached by a load feeding the subgraph or
// a store consuming it) into one unit, so the per-unit tier is the buffer's.
void mergeSharedStagingSubgraphs(const Options &O,
                                 llvm::SmallVectorImpl<Subgraph> &subgraphs,
                                 llvm::Function &F);

} // namespace poseidon
#endif // POSEIDON_SUBGRAPHS_H
