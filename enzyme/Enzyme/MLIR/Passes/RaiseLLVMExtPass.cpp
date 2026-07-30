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

#include "Dialect/LLVMExt/LLVMExt.h"
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

          // getSymbolUses returns an empty (not nullopt) range when the
          // symbol simply isn't referenced anywhere in this module, which is
          // the common case (most translation units never call
          // __enzyme_ptr_size_hint); only bail out on the lookup+cast below
          // if there's actually a use to process, since symtable.lookup(name)
          // returns null when the symbol isn't declared in this module at
          // all, and casting that to FunctionOpInterface crashes.
          if (!uses || uses->empty())
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

            auto args = call.getArgOperands();
            if (args.size() < 2 || args.size() > 3) {
              failed = true;
              call.emitError() << "__enzyme_ptr_size_hint expects (ptr, size) "
                                  "or (ptr, size, addrspace), got "
                               << args.size() << " arguments";
              return;
            }

            // The address space selects which allocator/copy the clone of this
            // pointer has to use, so it has to be known at compile time rather
            // than being an arbitrary runtime value.
            int64_t memorySpace = 0;
            if (args.size() == 3) {
              APInt space;
              if (!matchPattern(args[2], m_ConstantInt(&space))) {
                failed = true;
                call.emitError() << "address space argument of "
                                    "__enzyme_ptr_size_hint is not a constant";
                return;
              }
              memorySpace = space.getSExtValue();
            }

            OpBuilder builder(call);
            llvm_ext::PtrSizeHintOp::create(builder, call.getLoc(), args[0],
                                            args[1], memorySpace);

            call.erase();
          }

          symtable.erase(fn);
        });

    if (failed)
      signalPassFailure();
  }
};

} // end anonymous namespace
