//=- FPCore.h - FP expressions as FPCore and as C ---------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The textual forms of the expression graph: FPCore, the language Herbie reads
// and answers in, with its operator names for LLVM instructions, and the C an
// explain record shows a rewrite as.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MODEL_FP_CORE_H
#define POSEIDON_MODEL_FP_CORE_H

#include "FPNode.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Value.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace llvm {
class Instruction;
class Type;
} // namespace llvm

namespace poseidon {

std::shared_ptr<FPNode> parseHerbieExpr(
    const std::string &expr,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
    llvm::SmallVectorImpl<std::pair<std::string, std::string>> *approxSpecs =
        nullptr,
    const llvm::SetVector<llvm::Value *> *subgraphInputs = nullptr);

std::string dtypeOfType(llvm::Type *T);

struct CRendering {
  const llvm::SetVector<llvm::Value *> *inputs = nullptr;
  std::function<std::string(llvm::Value *)> leafName;
  const llvm::DenseMap<const FPNode *, std::string> *dtypes = nullptr;
};

std::string renderC(const FPNode &node, const CRendering &how,
                    llvm::StringRef resultDtype = "");

bool isRenderableFPCore(
    llvm::StringRef expr,
    const std::unordered_map<std::string, llvm::Value *> &symbolToValueMap);

llvm::SmallVector<std::string, 1> fpcoreToC(
    const std::string &expr,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
    const CRendering &how, llvm::StringRef resultDtype = "",
    llvm::SmallVectorImpl<std::pair<std::string, std::string>> *approxSpecs =
        nullptr);

std::string getHerbieOperator(const llvm::Instruction &I);

void getUniqueArgs(const std::string &expr,
                   llvm::SmallSet<std::string, 8> &args);

} // namespace poseidon
#endif // POSEIDON_MODEL_FP_CORE_H
