//===- Passes.h - Poseidon MLIR passes ------------------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MLIR_PASSES_H
#define POSEIDON_MLIR_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::poseidon {

#define GEN_PASS_DECL
#include "Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

} // namespace mlir::poseidon

#endif // POSEIDON_MLIR_PASSES_H
