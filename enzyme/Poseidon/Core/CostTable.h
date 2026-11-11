//=- CostTable.h - the measured cost table of one compile -----------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The CSV measured on the target device that every price comes from: rows
// "<opcode>,<type>,<cost>" and the headers "# native_arch=" and
// "# scalar_types=", resolved and loaded once per compilation. A part of the
// model that is missing or unreadable stops the compile, naming the model.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CORE_COST_TABLE_H
#define POSEIDON_CORE_COST_TABLE_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
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

  std::unordered_set<std::string> ScalarTypes;
  std::string NativeArch;
  std::map<std::pair<std::string, std::string>, double> Model;
  bool Loaded = false;

  uint64_t Fingerprint = 0;
  bool FingerprintComputed = false;

  std::optional<std::unordered_set<std::string>> PTFuncs;
};

// Resolves and loads the model for the function named `function`, compiled for
// `targetCpu`, and aborts unless its native_arch header names `targetCpu`.
void requireCostModel(CostModelState &CM, llvm::StringRef targetCpu,
                      llvm::StringRef function);

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

// The scalar formats the target has hardware for, from the CSV header; aborts
// when the model has no "# scalar_types=" header.
const std::unordered_set<std::string> &getScalarTypes(CostModelState &CM);

// The (opcode,type) row; aborts naming the model when it is absent.
double queryCostModel(CostModelState &CM, const std::string &OpcodeName,
                      const std::string &TypeName);
// The (opcode,type) row, or `fallback` when the model has none: for the rows a
// candidate is proposed only with (such as "ozaki_dispatch_rel").
double queryCostModelOr(CostModelState &CM, const std::string &OpcodeName,
                        const std::string &TypeName, double fallback);

} // namespace poseidon
#endif // POSEIDON_CORE_COST_TABLE_H
