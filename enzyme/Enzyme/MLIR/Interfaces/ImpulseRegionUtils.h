//===- ImpulseRegionUtils.h - Impulse region-form utilities -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_IMPULSE_REGION_UTILS_H
#define ENZYME_MLIR_INTERFACES_IMPULSE_REGION_UTILS_H

#include "Dialect/Impulse/Impulse.h"

namespace mlir {
namespace impulse {

/// Builds the logpdf region from the sampler; false leaves the op unchanged.
bool constructLogpdfRegion(InferRegionOp regionOp);

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_IMPULSE_REGION_UTILS_H
