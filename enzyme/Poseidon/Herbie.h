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

#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "Types.h"

namespace poseidon {

std::shared_ptr<FPNode> parseHerbieExpr(
    const std::string &expr,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap);

bool improveViaHerbie(
    const std::vector<std::string> &inputExprs,
    std::vector<CandidateOutput> &COs, llvm::Module *M,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
    int subgraphIdx, llvm::StringRef funcTag, llvm::StringRef cacheKey,
    llvm::StringRef keyTag = "");

std::string getHerbieOperator(const llvm::Instruction &I);

std::string getPrecondition(
    const llvm::SmallSet<std::string, 8> &args,
    const std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
        &valueToNodeMap,
    const std::unordered_map<std::string, llvm::Value *> &symbolToValueMap);

void setUnifiedAccuracyCost(
    CandidateOutput &CO,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap);

double getCompCost(
    const std::string &expr, llvm::Module *M,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
    const llvm::FastMathFlags &FMF);

} // namespace poseidon
#endif // POSEIDON_HERBIE_H
