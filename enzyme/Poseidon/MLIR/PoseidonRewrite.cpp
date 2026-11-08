//===- PoseidonRewrite.cpp - Rewrite elementwise FP chains ----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The poseidon-rewrite pass: chains of elementwise arith/math ops on
// floating-point tensors are written as FPCore, rewritten by Herbie, and
// replaced by the rewrite.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

namespace mlir::poseidon {
#define GEN_PASS_DEF_POSEIDONREWRITEPASS
#include "Passes.h.inc"
} // namespace mlir::poseidon

using namespace mlir;

namespace {

struct PoseidonRewrite
    : public poseidon::impl::PoseidonRewritePassBase<PoseidonRewrite> {
  void runOnOperation() override {}
};

} // namespace
