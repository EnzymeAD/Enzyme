//===- SICMRuleSetInterface.h - Rewrite rules of the sicm pass --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_SICM_RULE_SET_INTERFACE_H
#define ENZYME_MLIR_INTERFACES_SICM_RULE_SET_INTERFACE_H

#include "Dialect/Impulse/Impulse.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringMap.h"

#include <string>

namespace mlir {
namespace impulse {

/// The `rule-options` of a `sicm` run, as key and value.
using SICMRuleOptions = llvm::StringMap<std::string>;

/// The rewrite rules a dialect contributes to the `sicm` pass.
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
};

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_SICM_RULE_SET_INTERFACE_H
