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
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>

namespace poseidon {

// An error at `F` unless the cost model is present and its native_arch header
// names this function's target-cpu.
llvm::Error requireCostModel(CostModelState &CM, const llvm::Function &F);

// The row that prices `I`; a model without that row is an error at `At`, by
// default `I`, as is an instruction no row family prices.
llvm::Expected<double> instructionCost(CostModelState &CM,
                                       const llvm::Instruction *I,
                                       const llvm::Instruction *At = nullptr);

// The scalar formats the device of `F` has hardware for, from the CSV header;
// a model without the "# scalar_types=" header is an error at `F`.
llvm::Expected<const std::unordered_set<std::string> *>
scalarTypes(CostModelState &CM, const llvm::Function &F);

// What the profiled run spent in `F`: every operation the profile records, its
// execution count times its price. `slotOf` gives an instruction's profile
// slot, or false for an instruction without one.
llvm::Expected<double> profiledCost(
    CostModelState &CM, const llvm::Function &F, const ProfileMap &profile,
    llvm::function_ref<bool(const llvm::Instruction &, size_t &)> slotOf);

// The libm calls that are cheaper computed at FP32 and converted back than at
// FP64 on this device. A model without a row this needs is an error at `F`.
llvm::Expected<const std::unordered_set<std::string> *>
ptFuncs(CostModelState &CM, const llvm::Function &F);

// `Weight`, when given, is how often an instruction executes per execution of
// the function it is priced in.
using InstWeight = llvm::function_ref<double(const llvm::Instruction &)>;

// The most expensive path through `F`, a scratch function priced for the
// instruction `At`, where a missing row is reported.
llvm::Expected<double> compCost(CostModelState &CM, llvm::Function *F,
                                InstWeight Weight, const llvm::Instruction *At);

// With `opExec`, each priced instruction is weighted by its measured
// execution count over `normalizer`; an instruction with no count, or no map,
// weighs 1.
llvm::Expected<double> compCost(
    CostModelState &CM, llvm::ArrayRef<llvm::Value *> outputs,
    const llvm::SetVector<llvm::Value *> &inputs,
    const llvm::DenseMap<const llvm::Instruction *, uint64_t> *opExec = nullptr,
    uint64_t normalizer = 0);

} // namespace poseidon
#endif // POSEIDON_COST_MODEL_H
