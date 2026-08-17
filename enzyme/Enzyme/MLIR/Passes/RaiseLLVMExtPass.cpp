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

            auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(args[0].getType());
            if (!ptrTy) {
              failed = true;
              call.emitError()
                  << "first argument of __enzyme_ptr_size_hint is not a "
                     "pointer";
              return;
            }

            OpBuilder builder(call);
            Value ptr = args[0];

            // The optional third argument names the memory space the pointer
            // really addresses. The callers that need it are the ones whose
            // frontend cannot put that space in the type -- a cudaMalloc'd
            // buffer is a plain `float *`, address space 0 like every other
            // host pointer in CUDA C -- so rather than rejecting an annotation
            // that disagrees with the type, hint an llvm.addrspacecast to the
            // annotated space. Everything downstream keeps taking the memory
            // space from a pointer's own type: the clone of this allocation is
            // made from the cast, so it is allocated and copied with the device
            // runtime instead of malloc/memcpy.
            if (args.size() == 3) {
              APInt space;
              if (!matchPattern(args[2], m_ConstantInt(&space))) {
                failed = true;
                call.emitError() << "address space argument of "
                                    "__enzyme_ptr_size_hint is not a constant";
                return;
              }
              if (space.isNegative()) {
                failed = true;
                call.emitError() << "address space argument of "
                                    "__enzyme_ptr_size_hint is negative: "
                                 << space.getSExtValue();
                return;
              }
              if (space.getZExtValue() != ptrTy.getAddressSpace()) {
                auto castTy = LLVM::LLVMPointerType::get(
                    &getContext(), (unsigned)space.getZExtValue());
                ptr = LLVM::AddrSpaceCastOp::create(builder, call.getLoc(),
                                                    castTy, ptr);
              }
            }

            llvm_ext::PtrSizeHintOp::create(builder, call.getLoc(), ptr,
                                            args[1]);

            call.erase();
          }

          symtable.erase(fn);
        });

    if (failed)
      signalPassFailure();
  }
};

} // end anonymous namespace
