//=- Herbie.h - Herbie integration utilities ------------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for integrating with the Herbie tool for
// floating-point expression optimization.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_HERBIE_H
#define POSEIDON_HERBIE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Value.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "Core/CacheKey.h"
#include "Model/FPCore.h"
#include "Types.h"

namespace poseidon {

struct Options;
class Session;

// The parts of a cache key that say which Herbie searched: its binary and the
// platform it ranks candidates by.
void addHerbieKeyParts(Session &S, CacheKey &key);

// Collects Herbie's rewrites of `inputExprs` into `COs`, unpriced.
bool improveViaHerbie(Session &S, const std::vector<std::string> &inputExprs,
                      std::vector<RewriteUnit> &COs, int subgraphIdx,
                      llvm::StringRef funcTag);

std::string getPrecondition(
    const Options &O, const llvm::SmallSet<std::string, 8> &args,
    const std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
        &valueToNodeMap,
    const std::unordered_map<std::string, llvm::Value *> &symbolToValueMap);

void setUnifiedAccuracyCost(
    const Options &O, RewriteUnit &CO,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap);

double getCompCost(
    CostModelState &CM, const std::string &expr, llvm::Module *M,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
    const llvm::FastMathFlags &FMF,
    const llvm::SetVector<llvm::Value *> *subgraphInputs = nullptr);

// Herbie's rewrites of the outputs of `subgraph`, the site's subgraph number
// `subgraphCounter`, appended to `COs` unpriced. Its inputs are named from
// `symbolCounter`, which the site's subgraphs share.
void proposeHerbieCandidates(
    Session &S, llvm::Function &F, Subgraph &subgraph, int subgraphCounter,
    int &symbolCounter,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
    llvm::SmallVectorImpl<RewriteUnit> &COs);

} // namespace poseidon
#endif // POSEIDON_HERBIE_H
