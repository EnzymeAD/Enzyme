//===- TestTypeTreeAttr.cpp - Test the conversion of MTypeTree -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Dialect.h"
#include "Dialect/Ops.h"
#include "Interfaces/EnzymeLogic.h"
#include "Passes/PassDetails.h"
#include "Passes/Passes.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_TESTTYPETREEATTRPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {
struct TestTypeTreeAttrPass
    : public enzyme::impl::TestTypeTreeAttrPassBase<TestTypeTreeAttrPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    getOperation()->walk([&](Operation *op) {
      auto str = op->getAttrOfType<StringAttr>("test.type_tree");
      if (!str)
        return;
      op->removeAttr("test.type_tree");
      MTypeTree tree = MTypeTree::parse(str.getValue(), *context);
      if (TypeTreeAttr attr = getTypeTreeAttr(context, tree))
        op->setAttr(EnzymeDialect::getTypeTreeAttrName(), attr);
    });
  }
};
} // namespace
