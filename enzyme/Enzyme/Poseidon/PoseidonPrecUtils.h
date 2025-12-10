//=- PoseidonPrecUtils.h - Precision change utilities for Poseidon --------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for handling precision changes in the Poseidon
// optimization pass.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_POSEIDON_PREC_UTILS_H
#define ENZYME_POSEIDON_PREC_UTILS_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <limits>
#include <string>
#include <unordered_map>

using namespace llvm;

extern "C" {
extern llvm::cl::opt<bool> FPOptShowPTDetails;
extern llvm::cl::opt<unsigned> FPOptMaxMPFRPrec;
}

class FPNode;
class FPLLValue;
struct Subgraph;
class CandidateSubgraph;

enum class PrecisionChangeType { BF16, FP16, FP32, FP64, FP80, FP128 };
unsigned getMPFRPrec(PrecisionChangeType type);
Type *getLLVMFPType(PrecisionChangeType type, LLVMContext &context);
PrecisionChangeType getPrecisionChangeType(Type *type);
StringRef getPrecisionChangeTypeString(PrecisionChangeType type);

struct PrecisionChange {
  SetVector<FPLLValue *> nodes;
  PrecisionChangeType oldType;
  PrecisionChangeType newType;

  explicit PrecisionChange(SetVector<FPLLValue *> &nodes,
                           PrecisionChangeType oldType,
                           PrecisionChangeType newType)
      : nodes(nodes), oldType(oldType), newType(newType) {}
};

struct PTCandidate {
  SmallVector<PrecisionChange, 1> changes;
  double accuracyCost = std::numeric_limits<double>::quiet_NaN();
  InstructionCost CompCost = std::numeric_limits<InstructionCost>::max();
  std::string desc;
  std::unordered_map<FPNode *, double> perOutputAccCost;
  std::unordered_map<FPNode *, SmallVector<double, 4>> errors;

  explicit PTCandidate(SmallVector<PrecisionChange> changes,
                       const std::string &desc)
      : changes(std::move(changes)), desc(desc) {}

  void apply(Subgraph &subgraph, ValueToValueMapTy *VMap = nullptr);
};

void changePrecision(Instruction *I, PrecisionChange &change,
                     MapVector<Value *, Value *> &oldToNew);

InstructionCost getCompCost(Subgraph &subgraph, const TargetTransformInfo &TTI,
                            PTCandidate &pt);

void setUnifiedAccuracyCost(
    CandidateSubgraph &CS,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap);

#endif // ENZYME_POSEIDON_PREC_UTILS_H