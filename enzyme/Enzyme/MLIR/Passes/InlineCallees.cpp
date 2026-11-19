//===- InlineCallees.cpp - Inline the callees of what is differentiated ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Inlines the calls in the functions enzyme.autodiff and enzyme.fwddiff
// differentiate, and in what is inlined into them, so that each is
// differentiated as one function, as LLVM Enzyme differentiates after -O2:
// without a derivative of every helper and a tape between them. The rest of
// the module is left alone. Only direct calls of functions with a body that
// are not recursive are inlined, through the dialects' inliner interfaces.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Dialect.h"
#include "Dialect/Ops.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_INLINECALLEESPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

// The function a call calls, if it names one.
FunctionOpInterface directCallee(CallOpInterface call,
                                 SymbolTableCollection &symbols) {
  auto sym = dyn_cast_if_present<SymbolRefAttr>(call.getCallableForCallee());
  if (!sym)
    return nullptr;
  return dyn_cast_if_present<FunctionOpInterface>(
      symbols.lookupNearestSymbolFrom(call, sym));
}

// The functions that can call themselves, directly or not: the members of
// the call graph's cycles, by Tarjan's algorithm.
DenseSet<Operation *> recursiveFunctions(ModuleOp module,
                                         SymbolTableCollection &symbols) {
  DenseMap<Operation *, SmallVector<Operation *>> callees;
  module.walk([&](FunctionOpInterface fn) {
    auto &out = callees[fn];
    fn.walk([&](CallOpInterface call) {
      if (FunctionOpInterface callee = directCallee(call, symbols))
        out.push_back(callee);
    });
  });

  DenseSet<Operation *> recursive;
  DenseMap<Operation *, unsigned> index, low;
  SmallVector<Operation *> stack;
  DenseSet<Operation *> onStack;
  unsigned next = 0;
  std::function<void(Operation *)> visit = [&](Operation *fn) {
    index[fn] = low[fn] = next++;
    stack.push_back(fn);
    onStack.insert(fn);
    for (Operation *callee : callees[fn]) {
      if (callee == fn)
        recursive.insert(fn);
      if (!index.count(callee)) {
        visit(callee);
        low[fn] = std::min(low[fn], low[callee]);
      } else if (onStack.contains(callee)) {
        low[fn] = std::min(low[fn], index[callee]);
      }
    }
    if (low[fn] != index[fn])
      return;
    SmallVector<Operation *> scc;
    Operation *member;
    do {
      member = stack.pop_back_val();
      onStack.erase(member);
      scc.push_back(member);
    } while (member != fn);
    if (scc.size() > 1)
      recursive.insert(scc.begin(), scc.end());
  };
  for (auto &[fn, _] : callees)
    if (!index.count(fn))
      visit(fn);
  return recursive;
}

struct InlineCalleesPass
    : public enzyme::impl::InlineCalleesPassBase<InlineCalleesPass> {
  using InlineCalleesPassBase::InlineCalleesPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTableCollection symbols;

    SetVector<Operation *> targets;
    module.walk([&](Operation *op) {
      if (!isa<enzyme::AutoDiffOp, enzyme::ForwardDiffOp>(op))
        return;
      auto fn = op->getAttrOfType<FlatSymbolRefAttr>("fn");
      if (Operation *target = symbols.lookupNearestSymbolFrom(op, fn))
        targets.insert(target);
    });
    if (targets.empty())
      return;

    DenseSet<Operation *> recursive = recursiveFunctions(module, symbols);
    InlinerInterface interface(&getContext());
    auto cloneCallback = [](OpBuilder &builder, Region *src, Block *inlineBlock,
                            Block *postInsertBlock, IRMapping &mapper,
                            bool shouldCloneInlinedRegion) {
      Region *insertRegion = inlineBlock->getParent();
      src->cloneInto(insertRegion, postInsertBlock->getIterator(), mapper);
    };

    for (Operation *target : targets) {
      auto fn = cast<FunctionOpInterface>(target);
      // Inlined callees bring their own calls; those are inlined in turn,
      // until there are none left that can be, or the function is as large
      // as it is allowed to grow.
      unsigned size = 0;
      fn.walk([&](Operation *) { ++size; });
      bool changed = true;
      while (changed) {
        changed = false;
        SmallVector<std::pair<CallOpInterface, FunctionOpInterface>> calls;
        fn.walk([&](CallOpInterface call) {
          FunctionOpInterface callee = directCallee(call, symbols);
          if (callee && callee != fn && !callee.getFunctionBody().empty() &&
              !recursive.contains(callee))
            calls.push_back({call, callee});
        });
        for (auto [call, callee] : calls) {
          if (size > maxSize)
            break;
          auto callable = cast<CallableOpInterface>(callee.getOperation());
          Region *body = callable.getCallableRegion();
          unsigned added = 0;
          body->walk([&](Operation *) { ++added; });
          if (failed(
                  inlineCall(interface, cloneCallback, call, callable, body)))
            continue;
          call.erase();
          size += added - 1;
          changed = true;
        }
      }
    }
  }
};

} // namespace
