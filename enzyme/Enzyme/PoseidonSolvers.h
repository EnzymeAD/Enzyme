//=- PoseidonSolvers.h - Solver utilities for Poseidon --------------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares solver-related utilities for the Poseidon optimization
// pass.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_POSEIDON_SOLVERS_H
#define ENZYME_POSEIDON_SOLVERS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"

#include <memory>
#include <unordered_map>

#include "PoseidonTypes.h"

bool accuracyGreedySolver(
    SmallVector<ApplicableOutput, 4> &AOs, SmallVector<ApplicableFPCC, 4> &ACCs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap);

bool accuracyDPSolver(
    SmallVector<ApplicableOutput, 4> &AOs, SmallVector<ApplicableFPCC, 4> &ACCs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap);

#endif // ENZYME_POSEIDON_SOLVERS_H