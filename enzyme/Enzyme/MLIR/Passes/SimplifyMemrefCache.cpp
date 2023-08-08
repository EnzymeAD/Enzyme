#include "Dialect/Dialect.h"
#include "Dialect/Ops.h"
#include "PassDetails.h"
#include "Passes/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Rewrite/PatternApplicator.h"

#include "mlir/IR/Dominance.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace enzyme;
using llvm::errs;
namespace {

struct SimplifyMemrefCachePass
    : public enzyme::SimplifyMemrefCachePassBase<SimplifyMemrefCachePass> {

  void handlePushOp(enzyme::PushOp pushOp, Type newType, enzyme::CacheType c2) {
    auto v = pushOp.getValue();
    auto definingOp = v.getDefiningOp();
    auto allocOp = dyn_cast<memref::AllocOp>(definingOp);
    if (!allocOp) {
      llvm_unreachable("Unknown user of memref<CacheType>");
    }
    OpBuilder allocBuilder(allocOp);
    auto newAllocOp = allocBuilder.create<memref::AllocOp>(
        allocOp.getLoc(), dyn_cast<MemRefType>(newType),
        allocOp.getDynamicSizes(), allocOp.getSymbolOperands(),
        allocOp.getAlignmentAttr());

    for (auto user : allocOp->getUsers()) {
      auto linalgOp = dyn_cast<linalg::GenericOp>(user);
      if (isa<enzyme::PushOp>(user)) {
        continue;
      } else if (!linalgOp) {
        llvm_unreachable("Unknown user of memref<CacheType>");
      }

      for (auto &&output : llvm::enumerate(linalgOp.getOutputs())) {
        if (output.value() != allocOp) {
          continue;
        }
        unsigned outputIndex =
            linalgOp.getNumDpsInputs() + (unsigned)output.index();

        // We should never actually use the value of the output!
        assert(linalgOp.getRegion().getArgument(outputIndex).use_empty());

        linalgOp.getRegion().eraseArgument(outputIndex);
        linalgOp.getRegion().insertArgument(outputIndex, c2.getType(),
                                            output.value().getLoc());

        Value cache = linalgOp.getRegion().front().getTerminator()->getOperand(
            (unsigned)output.index());
        for (auto user : cache.getUsers()) {
          if (isa<enzyme::PopOp>(user)) {
            llvm_unreachable("PopOp should not be used in forward pass");
          }
          auto pushOp = dyn_cast<enzyme::PushOp>(user);
          if (!pushOp) {
            continue;
          }
          linalgOp.getRegion().front().getTerminator()->setOperand(
              (unsigned)output.index(), pushOp.getValue());
          pushOp.erase();
        }
        cache.getDefiningOp()->erase();
      }
    }

    allocOp.replaceAllUsesWith((Value)newAllocOp);
    allocOp.erase();
  }

  void handlePopOp(enzyme::PopOp popOp, Type newType, enzyme::CacheType c2) {
    OpBuilder popBuilder(popOp);
    auto newPopOp = popBuilder.create<enzyme::PopOp>(popOp.getLoc(), newType,
                                                     popOp.getCache());

    // TODO: handle all the stuff inside linalg.generic
    for (auto user : popOp->getUsers()) {
      auto subviewOp = dyn_cast<memref::SubViewOp>(user);
      if (!subviewOp) {
        continue;
      }
      for (auto user : subviewOp->getUsers()) {
        auto linalgOp = dyn_cast<enzyme::GenericAdjointOp>(user);
        if (!linalgOp) {
          continue;
        }
        for (auto &&input : llvm::enumerate(linalgOp.getInputs())) {
          if (input.value() != subviewOp) {
            continue;
          }
          unsigned inputIndex = (unsigned)input.index();
          Value inputCacheSSA = linalgOp.getRegion().insertArgument(
              inputIndex, c2.getType(), input.value().getLoc());
          Value oldArg = linalgOp.getRegion().getArgument(inputIndex + 1);
          for (auto user : oldArg.getUsers()) {
            auto popOp = dyn_cast<enzyme::PopOp>(user);
            if (!popOp) {
              llvm_unreachable("Unknown user");
            }
            popOp.replaceAllUsesWith(inputCacheSSA);
            popOp.erase();
          }

          // +1 because we inserted an argument above
          linalgOp.getRegion().eraseArgument(inputIndex + 1);
        }
      }
      // Replace Subview Op
      OpBuilder subviewBuilder(subviewOp);
      auto newSubviewOp = subviewBuilder.create<memref::SubViewOp>(
          subviewOp.getLoc(), newPopOp, subviewOp.getOffsets(),
          subviewOp.getSizes(), subviewOp.getStrides());
      subviewOp.replaceAllUsesWith((Value)newSubviewOp);
      subviewOp.erase();
    }
    popOp.replaceAllUsesWith((Value)newPopOp);
    popOp.erase();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    getOperation()->walk([&](Operation *op) {
      auto initOp = dyn_cast<enzyme::InitOp>(op);
      if (!initOp) {
        return;
      }
      auto c1 = dyn_cast<enzyme::CacheType>(initOp.getType());
      if (!c1) {
        return;
      }
      auto memref = dyn_cast<MemRefType>(c1.getType());
      if (!memref) {
        return;
      }
      auto c2 = dyn_cast<enzyme::CacheType>(memref.getElementType());
      if (!c2) {
        return;
      }
      mlir::MemRefType::Builder memrefTypeBuilder(memref);
      memrefTypeBuilder.setElementType(c2.getType());
      Type newType = memrefTypeBuilder;
      Type newCacheType = enzyme::CacheType::get(context, newType);
      for (auto user : op->getUsers()) {
        if (auto pushOp = dyn_cast<enzyme::PushOp>(user)) {
          handlePushOp(pushOp, newType, c2);
        } else if (auto popOp = dyn_cast<enzyme::PopOp>(user)) {
          handlePopOp(popOp, newType, c2);
        } else {
          llvm_unreachable("Unknown user of InitOp");
        }
      }

      OpBuilder builder(op);
      auto newInit = builder.create<enzyme::InitOp>(op->getLoc(), newCacheType);
      op->replaceAllUsesWith(newInit);

      op->erase();
    });
  };
};
} // namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createSimplifyMemrefCachePass() {
  return std::make_unique<SimplifyMemrefCachePass>();
}
} // namespace enzyme
} // namespace mlir
