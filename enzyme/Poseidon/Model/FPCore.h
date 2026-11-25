//===- FPCore.h - FP expressions as FPCore and as C -----------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
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
#include "llvm/Support/Error.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace llvm {
class Function;
class Instruction;
class Type;
} // namespace llvm

namespace poseidon {

// The expression graph of the FPCore expression `expr`, over the values its
// symbols name. An error, saying what and where, for an expression that is not
// one the evaluators and the lowering model: an unknown symbol or operator, an
// operator of an unmodelled format or with the wrong operand count, a truth
// value where a number belongs or the reverse, or a malformed literal, approx
// form or array.
llvm::Expected<std::shared_ptr<FPNode>> parseHerbieExpr(
    llvm::StringRef expr,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
    llvm::SmallVectorImpl<std::pair<std::string, std::string>> *approxSpecs =
        nullptr,
    const llvm::SetVector<llvm::Value *> *subgraphInputs = nullptr);

std::string dtypeOfType(llvm::Type *T);
// The C type of an FP dtype; "double" for any other.
llvm::StringRef cTypeName(llvm::StringRef dtype);

struct CRendering {
  const llvm::SetVector<llvm::Value *> *inputs = nullptr;
  std::function<std::string(llvm::Value *)> leafName;
  const llvm::DenseMap<const FPNode *, std::string> *dtypes = nullptr;
};

std::string renderC(const FPNode &node, const CRendering &how,
                    llvm::StringRef resultDtype = "");

llvm::SmallVector<std::string, 1> fpcoreToC(
    llvm::StringRef expr,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
    const CRendering &how, llvm::StringRef resultDtype = "",
    llvm::SmallVectorImpl<std::pair<std::string, std::string>> *approxSpecs =
        nullptr);

std::string getHerbieOperator(const llvm::Instruction &I);

// The name of intrinsic `Callee` without its "llvm." prefix and type suffixes:
// "sin" for llvm.sin.f64.
llvm::StringRef intrinsicBaseName(const llvm::Function &Callee);

void getUniqueArgs(const std::string &expr,
                   llvm::SmallSet<std::string, 8> &args);

} // namespace poseidon
#endif // POSEIDON_MODEL_FP_CORE_H
