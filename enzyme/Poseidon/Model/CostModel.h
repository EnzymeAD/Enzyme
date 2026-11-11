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

#include "Core/CostTable.h"
#include "Profile/ProfileRead.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
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

// Poseidon prices every candidate from the measured CSV; a site cannot be
// optimized without one, and a CSV measured on another device is not a price
// for this one. Aborts unless the model is present and its native_arch header
// names this function's target-cpu.
void requireCostModel(CostModelState &CM, const llvm::Function &F);

// The row that prices `I`; aborts when no row family prices its kind.
double getInstructionCompCost(CostModelState &CM, const llvm::Instruction *I);

// What the profiled run spent in `F`: every operation the profile records, its
// execution count times its price. `slotOf` gives an instruction's profile
// slot, or false for an instruction without one.
double profiledCost(
    CostModelState &CM, const llvm::Function &F, const ProfileMap &profile,
    llvm::function_ref<bool(const llvm::Instruction &, size_t &)> slotOf);

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
