//===- LogpdfRegion.h - Build the logpdf region of infer_region -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_LOGPDF_REGION_H
#define ENZYME_MLIR_INTERFACES_LOGPDF_REGION_H

#include "Dialect/Impulse/Impulse.h"

namespace mlir {
namespace impulse {

/// Builds the `logpdf` region of `regionOp` from its `sampling` region.
/// Returns false, leaving the op unchanged, when `logpdf` is not empty or a
/// site of `sampling` is nested in an op other than `impulse.if`, has no
/// address or log density, or does not match the address arrays of the op.
/// An op in a branch that holds sample sites moves above the branch when its
/// operands come from outside the branch, it has results but no regions or
/// memory effects, and it is an `arith.constant`, `impulse.slice` or
/// `impulse.reshape` or `isHoistableFromBranch` holds.
bool constructLogpdfRegion(
    InferRegionOp regionOp,
    function_ref<bool(Operation *)> isHoistableFromBranch);

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_LOGPDF_REGION_H
