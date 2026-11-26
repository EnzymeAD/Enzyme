//===- SICMRuleSetInterface.h - Rewrite rules of the sicm pass --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_SICM_RULE_SET_INTERFACE_H
#define ENZYME_MLIR_INTERFACES_SICM_RULE_SET_INTERFACE_H

#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringMap.h"

#include <string>

namespace mlir {
namespace impulse {

/// The `rule-options` of a `sicm` run, as key and value.
using SICMRuleOptions = llvm::StringMap<std::string>;

/// The rewrite rules a dialect contributes to the `sicm` pass, and the ops
/// the pass may move out of a branch of a density.
class SICMRuleSetInterface
    : public DialectInterface::Base<SICMRuleSetInterface> {
public:
  SICMRuleSetInterface(Dialect *dialect) : Base(dialect) {}

  /// Adds the rules of the dialect.
  virtual void populateRules(RewritePatternSet &patterns) const = 0;

  /// True if `key` is a `rule-options` key of this rule set.
  virtual bool hasOption(StringRef key) const { return false; }

  /// Applies `options` to `patterns`, the rules of one rewrite step. Fails
  /// after emitting on `root` when a value is invalid.
  virtual LogicalResult configureRules(RewritePatternSet &patterns,
                                       const SICMRuleOptions &options,
                                       Operation *root) const {
    return success();
  }

  /// Called once per run, after every region.
  virtual void finalize(Operation *root, const SICMRuleOptions &options) const {
  }

  /// True if `op`, which has results but no regions or memory effects, may
  /// move out of a branch of a density and run whether or not the branch is
  /// taken. That requires `op` to be a constant or a linear function of its
  /// operands, such as an op that moves or converts elements: its derivative
  /// does not depend on the operand values, so the branch not taken adds
  /// exactly zero to the gradient of the density.
  virtual bool isHoistableFromBranch(Operation *op) const { return false; }
};

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_SICM_RULE_SET_INTERFACE_H
