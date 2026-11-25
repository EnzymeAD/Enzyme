//===- CostTable.h - the measured cost table of one compile ---------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The CSV measured on the target device that every price comes from: rows
// "<opcode>,<type>,<cost>" and the headers "# native_arch=" and
// "# scalar_types=", resolved and loaded once per compilation. A part of the
// model that is missing or unreadable is an error naming the model.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MODEL_COST_TABLE_H
#define POSEIDON_MODEL_COST_TABLE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

namespace poseidon {

/// The cost model of one compile: the CSV resolved from the request (the
/// -poseidon-cost-model it is constructed with, POSEIDON_COST_MODEL) or from
/// the first priced function's target-cpu, loaded once, and what is derived
/// from it.
struct CostModelState {
  /// A state for the model `requested` names; empty for one resolved by
  /// target-cpu.
  explicit CostModelState(std::string requested)
      : Requested(std::move(requested)) {}

  /// The CSV the compile names; empty when none does.
  std::string Requested;
  /// The CSV in effect, once resolved.
  std::string Resolved;
  /// Whether `Resolved` has been decided.
  bool IsResolved = false;

  /// The scalar formats the device has hardware for, from the "# scalar_types="
  /// header; empty when the model has none.
  std::unordered_set<std::string> ScalarTypes;
  /// The processor the model was measured on, from the "# native_arch=" header.
  std::string NativeArch;
  /// The model's prices by (operation, type) row, in the units the CSV gives.
  std::map<std::pair<std::string, std::string>, double> Model;
  /// Whether the model has been parsed.
  bool Loaded = false;

  /// The libm functions whose single-precision call, with its two casts, is
  /// priced below the double-precision call; computed once, on first use.
  std::optional<std::unordered_set<std::string>> PTFuncs;
};

/// The directory a compile looks a model up in by target-cpu:
/// $XDG_CACHE_HOME/poseidon, else $HOME/.cache/poseidon; empty when neither
/// variable is set.
std::string userCacheDir();

/// Parses the model the request names, and fails unless every line is a header
/// or a row of a family poseidon-calibrate writes.
llvm::Error loadCostModel(CostModelState &CM);

/// Resolves and loads the model for the function named `function`, compiled for
/// `targetCpu`, and fails unless its native_arch header names `targetCpu`.
/// `OnDevice` says whether `targetCpu` is a GPU's, which decides the
/// poseidon-calibrate command a missing model is named with.
llvm::Error requireCostModel(CostModelState &CM, llvm::StringRef targetCpu,
                             llvm::StringRef function, bool OnDevice);

/// The loaded model; a site has required it.
const std::map<std::pair<std::string, std::string>, double> &
getCostModel(CostModelState &CM);

/// The device the loaded CSV was measured on ("# native_arch=").
const std::string &getCostModelNativeArch(CostModelState &CM);

/// The CSV in effect, once resolved; empty before the first requireCostModel
/// unless the flag or POSEIDON_COST_MODEL named one.
const std::string &costModelPath(CostModelState &CM);

/// The loaded model as text, its rows in order with exact costs and its
/// native_arch header, without comments or blank lines: what a result computed
/// from its prices depends on. Empty when no model is configured.
std::string costModelRows(CostModelState &CM);

/// The (opcode,type) row, if the model has it.
std::optional<double> lookupCost(CostModelState &CM, llvm::StringRef OpcodeName,
                                 llvm::StringRef TypeName);

} // namespace poseidon
#endif // POSEIDON_MODEL_COST_TABLE_H
