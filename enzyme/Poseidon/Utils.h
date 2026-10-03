//=- Utils.h - Utility functions for Poseidon optimization pass ------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utility functions for the Poseidon floating-point
// optimization pass.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_UTILS_H
#define POSEIDON_UTILS_H

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/InstructionCost.h"

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace poseidon {

#if LLVM_VERSION_MAJOR >= 23
template <typename DerivedT>
using PassParent = llvm::RequiredPassInfoMixin<DerivedT>;
#else
template <typename DerivedT> using PassParent = llvm::PassInfoMixin<DerivedT>;
#endif

// LLVM 24 split BranchInst into CondBrInst and UncondBrInst.
inline bool isConditionalBranch(const llvm::Value *V) {
#if LLVM_VERSION_MAJOR >= 24
  return llvm::isa<llvm::CondBrInst>(V);
#else
  auto *BI = llvm::dyn_cast<llvm::BranchInst>(V);
  return BI && BI->isConditional();
#endif
}

inline llvm::Value *branchCondition(llvm::Value *V) {
#if LLVM_VERSION_MAJOR >= 24
  return llvm::cast<llvm::CondBrInst>(V)->getCondition();
#else
  return llvm::cast<llvm::BranchInst>(V)->getCondition();
#endif
}

bool isGPUMode(const llvm::Function &F);

struct Subgraph;
class FPNode;

std::string getLibmFunctionForPrecision(llvm::StringRef funcName,
                                        llvm::Type *newType);
// The libm calls Poseidon prices and can retype.
const std::unordered_set<std::string> &libmFuncs();
// The libm name behind a CUDA libdevice entry point Poseidon handles, for both
// the f64 and the f32 form ("__nv_sqrt" and "__nv_sqrtf" both give "sqrt"), or
// the empty string. Poseidon recognizes these itself instead of reading the
// attribute an AD pass would have put on them.
llvm::StringRef deviceMathName(llvm::StringRef fnName);
double stringToDouble(const std::string &str);
void topoSort(const llvm::SetVector<llvm::Instruction *> &insts,
              llvm::SmallVectorImpl<llvm::Instruction *> &instsSorted);
void reverseTopoSort(const llvm::SetVector<llvm::Instruction *> &insts,
                     llvm::SmallVectorImpl<llvm::Instruction *> &instsSorted);

void getUniqueArgs(const std::string &expr,
                   llvm::SmallSet<std::string, 8> &args);

void collectExprInsts(llvm::Value *V,
                      const llvm::SetVector<llvm::Value *> &inputs,
                      llvm::SmallPtrSetImpl<llvm::Instruction *> &exprInsts,
                      llvm::SmallPtrSetImpl<llvm::Value *> &visited);

void splitSubgraphs(llvm::SmallVectorImpl<Subgraph> &subgraphs);

void simplifyFunction(llvm::Function &F, llvm::OptimizationLevel Level);

} // namespace poseidon
#endif // POSEIDON_UTILS_H
