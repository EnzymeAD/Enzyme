//===- VIUtils.h - Variational inference lowering ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_VI_UTILS_H
#define ENZYME_MLIR_INTERFACES_VI_UTILS_H

#include "Dialect/Impulse/Impulse.h"
#include "HMCUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace impulse {

/// Replaces the VI inference `op` with an SVI loop over `model`, configured by
/// the op's `vi_config`, that starts from the op's `state_in` when it has one.
/// Fails with a diagnostic on guide kinds and modes the lowering does not
/// support.
template <typename OpT>
LogicalResult lowerVI(PatternRewriter &rewriter, OpT op, ModelContext model);

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_VI_UTILS_H
