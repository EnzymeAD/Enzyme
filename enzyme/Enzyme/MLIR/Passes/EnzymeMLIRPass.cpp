//===- EnzymeMLIRPass.cpp - //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower gpu kernels in NVVM/gpu dialects into
// a generic parallel for representation
//===----------------------------------------------------------------------===//
//#include "PassDetails.h"

#include "Dialect/Ops.h"
#include "Passes/Passes.h"
#include "PassDetails.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace enzyme;

namespace {
struct DifferentiatePass : public DifferentiatePassBase<DifferentiatePass> {
  void runOnOperation() override;
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createDifferentiatePass() {
    new DifferentiatePass();
  return std::make_unique<DifferentiatePass>();
}
} // namespace enzyme
} // namespace mlir

void DifferentiatePass::runOnOperation() {
  getOperation()->walk([&](enzyme::ForwardDiffOp dop) { });
}
