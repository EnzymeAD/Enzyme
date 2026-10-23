//===- EnzymeOps.h - Enzyme dialect ops -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEOPS_H
#define ENZYMEOPS_H

#include <type_traits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "Dialect/EnzymeAttributeInterfaces.h.inc"
#include "Dialect/EnzymeEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/EnzymeAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/EnzymeOpsTypes.h.inc"

// forward declare Enzyme op definitions
#include "Dialect/EnzymeOps.h.inc"

namespace mlir {
namespace enzyme {
namespace detail {

// For any differentiation op, we either return input primal values or selective
// derivative values. When `filterGrad` is true, `includeShadows` controls
// whether input shadow arguments (activity `enzyme_dup` / `enzyme_dupnoneed`)
// are collected, while `includeDifferentialReturns` controls whether
// reverse-mode output shadows (`enzyme_active` / `enzyme_activenoneed`) are
// collected.
template <typename SourceOp, bool filterGrad, bool includeShadows = true,
          bool includeDifferentialReturns = true>
llvm::SmallVector<mlir::Value, 2> filterGradInputs(SourceOp uop) {
  llvm::SmallVector<mlir::Value, 2> outs;
  size_t in_idx = 0;

  for (auto act : uop.getActivity()) {
    auto iattr = cast<ActivityAttr>(act);
    auto act_val = iattr.getValue();

    if constexpr (!filterGrad) {
      outs.push_back(uop.getInputs()[in_idx]);
    }

    ++in_idx;

    if (act_val == Activity::enzyme_dup ||
        act_val == Activity::enzyme_dupnoneed) {

      if constexpr (filterGrad && includeShadows) {
        outs.push_back(uop.getInputs()[in_idx]);
      }

      ++in_idx;
    }
  }

  // For reverse mode AD, add derivative values corresponding to active outputs
  // clang-format off
  if constexpr ((std::is_same_v<SourceOp, AutoDiffOp> ||
                 std::is_same_v<SourceOp, AutoDiffRegionOp>) &&
                filterGrad && includeDifferentialReturns) {
    // clang-format on
    if (in_idx != uop.getInputs().size()) {
      for (auto act : uop.getRetActivity()) {
        auto iattr = cast<ActivityAttr>(act);
        auto act_val = iattr.getValue();

        if (act_val == Activity::enzyme_active ||
            act_val == Activity::enzyme_activenoneed) {
          outs.push_back(uop.getInputs()[in_idx]);
          in_idx++;
        }
      }
    }
  }

  return outs;
}

} // namespace detail

} // namespace enzyme
} // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/EnzymeOps.h.inc"

// Declared after the op classes: `SmallVector<CustomReverseRuleOp>` needs the
// complete type.
namespace mlir {
namespace enzyme {

// The custom reverse rules the attribute `attrName` of `op` names: either one
// symbol (`@rule`) or a rule set (`[@rule_a, @rule_b]`), one rule per activity
// pattern. Fails if a symbol does not name an `enzyme.custom_reverse_rule`.
llvm::FailureOr<llvm::SmallVector<CustomReverseRuleOp>>
lookupCustomReverseRules(Operation *op, llvm::StringRef attrName);

// Adds `rule` to the rule set in attribute `attrName` of `op`.
void appendCustomReverseRule(Operation *op, llvm::StringRef attrName,
                             FlatSymbolRefAttr rule);

// Removes every reference to `rule` from the rule set in attribute `attrName`
// of `op`, and the attribute itself once it names no rule.
void removeCustomReverseRule(Operation *op, llvm::StringRef attrName,
                             llvm::StringRef rule);

// The caches of a custom reverse rule: its top-level `enzyme.init` ops of
// cache type, in block order. Their element types are the values
// `enzyme.call_augmented_primal` returns after the primal results, in the same
// order.
llvm::SmallVector<InitOp>
getCustomReverseRuleCacheInits(CustomReverseRuleOp rule);
llvm::SmallVector<Type>
getCustomReverseRuleCacheTypes(CustomReverseRuleOp rule);

} // namespace enzyme
} // namespace mlir

// #include "Dialect/EnzymeTypes.h.inc"

#endif // ENZYMEOPS_H
