//===- CIRInlinerInterface.cpp - Inlining ClangIR calls -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ClangIR does not say how its calls are inlined, so the inline pass cannot
// inline them. Differentiating a function its helpers are inlined into is
// what LLVM Enzyme gets from running after -O2: no derivative of each helper,
// with a tape between them, and what the helpers keep on their stack is the
// caller's, for mem2reg to promote.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Transforms/InliningUtils.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;

namespace {
struct CIRInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // A direct call of a function with a body.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    auto fn = dyn_cast<cir::FuncOp>(callable);
    auto callOp = dyn_cast<cir::CallOp>(call);
    return fn && callOp && callOp.getCallee() && !fn.isDeclaration() &&
           !fn.getFunctionType().isVarArg();
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  // The variadic arguments are the callee's frame's.
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return !isa<cir::VAStartOp, cir::VAEndOp, cir::VACopyOp, cir::VAArgOp>(op);
  }

  // A return goes on to what follows the call.
  void handleTerminator(Operation *op, Block *newDest) const final {
    auto ret = dyn_cast<cir::ReturnOp>(op);
    if (!ret)
      return;
    OpBuilder builder(op);
    cir::BrOp::create(builder, op->getLoc(), newDest, ret->getOperands());
    op->erase();
  }

  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    for (auto [result, returned] :
         llvm::zip(valuesToRepl, cast<cir::ReturnOp>(op)->getOperands()))
      result.replaceAllUsesWith(returned);
  }

  // Hoist alloca slot to the top of inlined function.
  void processInlinedCallBlocks(
      Operation *call,
      iterator_range<Region::iterator> inlinedBlocks) const final {
    auto fn = call->getParentOfType<cir::FuncOp>();
    if (!fn)
      return;
    Block &entry = fn.getBody().front();
    Operation *insertPoint = &entry.front();
    for (Block &block : inlinedBlocks)
      for (auto alloca :
           llvm::make_early_inc_range(block.getOps<cir::AllocaOp>())) {
        if (alloca.getDynAllocSize())
          continue;
        // Kept const, a hoisted slot can get LLVM invariant-group metadata it
        // should not, as cir-hoist-allocas notes.
        alloca.setConstant(false);
        alloca->moveBefore(insertPoint);
      }
  }
};
} // namespace

void mlir::enzyme::registerCIRInlinerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *, cir::CIRDialect *dialect) {
    dialect->addInterfaces<CIRInlinerInterface>();
  });
}
