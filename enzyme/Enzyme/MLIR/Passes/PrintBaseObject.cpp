//===- PrintBaseObject.cpp - Print base objects for tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interfaces/Utils.h"
#include "Passes/PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_PRINTBASEOBJECTPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {
struct PrintBaseObjectPass
    : public enzyme::impl::PrintBaseObjectPassBase<PrintBaseObjectPass> {
  using PrintBaseObjectPassBase::PrintBaseObjectPassBase;

  void runOnOperation() override {
    getOperation()->walk([&](FunctionOpInterface function) {
      AsmState state(function);
      function.walk([&](Operation *op) {
        if (!op->hasTrait<OpTrait::ReturnLike>() ||
            op->getParentOp() != function.getOperation())
          return;
        for (auto [index, value] : llvm::enumerate(op->getOperands())) {
          auto &os = llvm::outs();
          os << '@' << SymbolTable::getSymbolName(function).getValue()
             << " return " << index << ": ";
          value.printAsOperand(os, state);
          os << " -> ";
          oputils::getBaseObject(value, offsetAllowed)
              .printAsOperand(os, state);
          os << '\n';
        }
      });
    });
  }
};
} // namespace
