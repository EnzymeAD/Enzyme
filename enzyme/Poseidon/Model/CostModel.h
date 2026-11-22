//===- CostModel.h - the measured cost model and cost walks ---------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prices from the cost-model CSV measured on the target device, and the cost
// walks over IR built on them.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_COST_MODEL_H
#define POSEIDON_COST_MODEL_H

#include "Core/CostTable.h"
#include "Profile/ProfileRead.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>

namespace poseidon {

// Stops the compile with an error at `F` unless the cost model is present and
// its native_arch header names this function's target-cpu.
void requireCostModel(CostModelState &CM, const llvm::Function &F);

// The row that prices `I`; a model without that row is an error at `At`, by
// default `I`, and an instruction no row family prices aborts.
double getInstructionCompCost(CostModelState &CM, const llvm::Instruction *I,
                              const llvm::Instruction *At = nullptr);

// The scalar formats the device of `F` has hardware for, from the CSV header;
// a model without the "# scalar_types=" header is an error at `F`.
const std::unordered_set<std::string> &getScalarTypes(CostModelState &CM,
                                                      const llvm::Function &F);

// What the profiled run spent in `F`: every operation the profile records, its
// execution count times its price. `slotOf` gives an instruction's profile
// slot, or false for an instruction without one.
double profiledCost(
    CostModelState &CM, const llvm::Function &F, const ProfileMap &profile,
    llvm::function_ref<bool(const llvm::Instruction &, size_t &)> slotOf);

// The libm calls that are cheaper computed at FP32 and converted back than at
// FP64 on this device. A model without a row this needs is an error at `F`.
const std::unordered_set<std::string> &getPTFuncs(CostModelState &CM,
                                                  const llvm::Function &F);

// `Weight`, when given, is how often an instruction executes per execution of
// the function it is priced in.
using InstWeight = llvm::function_ref<double(const llvm::Instruction &)>;

// `At`, when given, is where a missing row is reported: the instruction a
// scratch function `F` is priced for.
double getCompCost(CostModelState &CM, llvm::Function *F,
                   InstWeight Weight = nullptr,
                   const llvm::Instruction *At = nullptr);

// With `opExec`, each priced instruction is weighted by its measured
// execution count over `normalizer`; an instruction with no count, or no map,
// weighs 1.
double getCompCost(
    CostModelState &CM, llvm::ArrayRef<llvm::Value *> outputs,
    const llvm::SetVector<llvm::Value *> &inputs,
    const llvm::DenseMap<const llvm::Instruction *, uint64_t> *opExec = nullptr,
    uint64_t normalizer = 0);

} // namespace poseidon
#endif // POSEIDON_COST_MODEL_H
