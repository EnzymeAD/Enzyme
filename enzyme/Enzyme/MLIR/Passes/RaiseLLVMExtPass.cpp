//===- RaiseLLVMExtPass.cpp - Raise LLVM Ext operations  ------------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to raise LLVM ops to the LLVM Ext
// dialect.
//
//===----------------------------------------------------------------------===//

<<<<<<< HEAD
#include "Dialect/LLVMExt/Dialect.h"
#include "Dialect/LLVMExt/Ops.h"
=======
#include "Dialect/LLVMExt/LLVMExt.h"
>>>>>>> upstream/main
#include "Passes/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace enzyme {
using namespace mlir::enzyme;
#define GEN_PASS_DEF_RAISELLVMEXTPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {
using namespace mlir;
using namespace enzyme;

struct RaiseLLVMExtPass
    : public enzyme::impl::RaiseLLVMExtPassBase<RaiseLLVMExtPass> {
  using RaiseLLVMExtPassBase::RaiseLLVMExtPassBase;

  void runOnOperation() override {
    bool failed = false;

    SymbolTable::walkSymbolTables(
        getOperation(),
        /*allUsesVisible*/ true, [&](Operation *st, bool allUsesVisible) {
          SymbolTable symtable(st);

          auto name = StringAttr::get(&getContext(), "__enzyme_ptr_size_hint");
          auto uses = SymbolTable::getSymbolUses(name, st);

          if (!uses)
            return;

          auto fn = cast<FunctionOpInterface>(symtable.lookup(name));
          if (!fn.isExternal()) {
            failed = true;
            fn.emitError() << "__enzyme_ptr_size_hint is not declared external";
            return;
          }

          for (auto use : *uses) {
            auto call = dyn_cast<LLVM::CallOp>(use.getUser());
            if (!call) {
              failed = true;
              use.getUser()->emitError()
                  << "user of __enzyme_ptr_size_hint is not a llvm.call";
              return;
            }

            OpBuilder builder(call);
            builder.create<llvm_ext::PtrSizeHintOp>(
                call.getLoc(), call.getOperand(0), call.getOperand(1));

            call.erase();
          }

          symtable.erase(fn);
        });

    if (failed)
      signalPassFailure();
  }
};

} // end anonymous namespace
