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

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

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
// derivative values. When `filterGrad` is true, `includeInShadows` controls
// whether input shadow arguments (activity `enzyme_dup` / `enzyme_dupnoneed`)
// are collected, while `includeOutShadows` controls whether reverse-mode output
// shadows (`enzyme_active` / `enzyme_activenoneed`) are collected.
template <typename SourceOp, bool filterGrad, bool includeInShadows = true,
          bool includeOutShadows = true>
llvm::SmallVector<mlir::Value> filterGradInputs(SourceOp uop) {
  llvm::SmallVector<mlir::Value, 2> outs;
  auto in_idx = 0;

  for (auto act : uop.getActivity()) {
    auto iattr = cast<ActivityAttr>(act);
    auto act_val = iattr.getValue();

    if constexpr (!filterGrad) {
      outs.push_back(uop.getInputs()[in_idx]);
    }

    ++in_idx;

    if (act_val == Activity::enzyme_dup ||
        act_val == Activity::enzyme_dupnoneed) {

      if constexpr (filterGrad && includeInShadows) {
        outs.push_back(uop.getInputs()[in_idx]);
      }

      ++in_idx;
    }
  }

  // For reverse mode AD, add derivative values corresponding to active outputs
  if constexpr ((std::is_same_v<SourceOp, AutoDiffOp> ||
                 std::is_same_v<SourceOp, AutoDiffRegionOp>) &&
                filterGrad && includeOutShadows) {
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

// #include "Dialect/EnzymeTypes.h.inc"

#endif // ENZYMEOPS_H
