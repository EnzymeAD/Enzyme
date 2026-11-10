//=- CostModel.h - the measured cost model and cost walks -----------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Every price Poseidon uses comes from one CSV measured on the target device,
// loaded once per compilation. The queries here are the only way to ask what
// an operation costs, and the cost walks below turn them into the per-candidate
// numbers the DP compares.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_COST_MODEL_H
#define POSEIDON_COST_MODEL_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace poseidon {

// The cost model of one compile: the CSV resolved from the request (the
// -poseidon-cost-model it is constructed with, POSEIDON_COST_MODEL) or from
// the first priced function's target-cpu, loaded once, and what is derived
// from it.
struct CostModelState {
  explicit CostModelState(std::string requested)
      : Requested(std::move(requested)) {}

  std::string Requested;
  std::string Resolved;
  bool IsResolved = false;

  struct {
    std::unordered_set<std::string> scalar;
    std::unordered_set<std::string> matrix;
  } SupportedTypes;
  std::string NativeArch;
  std::map<std::pair<std::string, std::string>, double> Model;
  bool Loaded = false;

  uint64_t Fingerprint = 0;
  bool FingerprintComputed = false;

  std::optional<std::unordered_set<std::string>> PTFuncs;
};

// Poseidon prices every candidate from the measured CSV; a site cannot be
// optimized without one, and a CSV measured on another device is not a price
// for this one. Aborts unless the model is present and its native_arch header
// names this function's target-cpu.
void requireCostModel(CostModelState &CM, const llvm::Function &F);

const std::map<std::pair<std::string, std::string>, double> &
getCostModel(CostModelState &CM);

// The device the loaded CSV was measured on ("# native_arch=").
const std::string &getCostModelNativeArch(CostModelState &CM);

// The CSV in effect, once resolved; empty before the first requireCostModel
// unless the flag or POSEIDON_COST_MODEL named one.
const std::string &costModelPath(CostModelState &CM);

// Content fingerprint of the loaded cost model (0 when none is configured).
// Used to key the DP table cache so a changed model cannot silently replay the
// previous model's picks.
uint64_t getCostModelFingerprint(CostModelState &CM);

// The scalar formats the target has hardware for, from the CSV header.
const std::unordered_set<std::string> &getScalarTypes(CostModelState &CM);

double queryCostModel(CostModelState &CM, const std::string &OpcodeName,
                      const std::string &TypeName);
// Non-aborting variant: returns `fallback` when the (opcode,type) row is absent
// (optional calibration rows such as "ozaki_dispatch_rel").
double queryCostModelOr(CostModelState &CM, const std::string &OpcodeName,
                        const std::string &TypeName, double fallback);

double getInstructionCompCost(CostModelState &CM, const llvm::Instruction *I);

// The libm calls that are cheaper computed at FP32 and converted back than at
// FP64 on this device.
const std::unordered_set<std::string> &getPTFuncs(CostModelState &CM);

double computeMaxCost(CostModelState &CM, llvm::BasicBlock *BB,
                      std::unordered_map<llvm::BasicBlock *, double> &MaxCost,
                      std::unordered_set<llvm::BasicBlock *> &Visited);

double getCompCost(CostModelState &CM, llvm::Function *F);

// With `opExec`, each priced instruction is weighted by its measured
// execution count over `normalizer` (Subgraph::opExec); an instruction with no
// count, or no map, weighs 1.
double getCompCost(
    CostModelState &CM, const llvm::SmallVector<llvm::Value *> &outputs,
    const llvm::SetVector<llvm::Value *> &inputs,
    const llvm::DenseMap<const llvm::Instruction *, uint64_t> *opExec = nullptr,
    uint64_t normalizer = 0);

} // namespace poseidon
#endif // POSEIDON_COST_MODEL_H
