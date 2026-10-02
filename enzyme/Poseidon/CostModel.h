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
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace poseidon {

// Poseidon prices every candidate from the measured CSV; a site cannot be
// optimized without one, and a CSV measured on another device is not a price
// for this one. Aborts unless the model is present and its native_arch header
// names this function's target-cpu.
void requireCostModel(const llvm::Function &F);

const std::map<std::pair<std::string, std::string>, double> &getCostModel();

// The device the loaded CSV was measured on ("# native_arch=").
const std::string &getCostModelNativeArch();

// The CSV in effect, once resolved; empty before the first requireCostModel
// unless the flag or POSEIDON_COST_MODEL named one.
const std::string &costModelPath();

// Content fingerprint of the loaded cost model (0 when none is configured).
// Used to key the DP table cache so a changed model cannot silently replay the
// previous model's picks.
uint64_t getCostModelFingerprint();

// The scalar formats the target has hardware for, from the CSV header.
const std::unordered_set<std::string> &getScalarTypes();

double queryCostModel(const std::string &OpcodeName,
                      const std::string &TypeName);
// Non-aborting variant: returns `fallback` when the (opcode,type) row is absent
// (optional calibration rows such as "ozaki_dispatch_rel").
double queryCostModelOr(const std::string &OpcodeName,
                        const std::string &TypeName, double fallback);

double getInstructionCompCost(const llvm::Instruction *I);

// The libm calls that are cheaper computed at FP32 and converted back than at
// FP64 on this device.
const std::unordered_set<std::string> &getPTFuncs();

double computeMaxCost(llvm::BasicBlock *BB,
                      std::unordered_map<llvm::BasicBlock *, double> &MaxCost,
                      std::unordered_set<llvm::BasicBlock *> &Visited);

double getCompCost(llvm::Function *F);

// `opExec` / `normalizer`: the caller's MEASURED per-instruction execution
// counts and the count the result is multiplied by (Subgraph::opExec,
// Subgraph::execNormalizer). Supplied, each priced instruction is weighted by
// exec(I)/normalizer, so a subgraph that mixes a reduction-loop body with
// once-per-thread code is not billed entirely at the body's frequency. Omitted,
// or for an instruction with no measured count, the weight is 1 and the result
// is bit-identical to the unweighted walk.
double getCompCost(
    const llvm::SmallVector<llvm::Value *> &outputs,
    const llvm::SetVector<llvm::Value *> &inputs,
    const llvm::DenseMap<const llvm::Instruction *, uint64_t> *opExec = nullptr,
    uint64_t normalizer = 0);

} // namespace poseidon
#endif // POSEIDON_COST_MODEL_H
