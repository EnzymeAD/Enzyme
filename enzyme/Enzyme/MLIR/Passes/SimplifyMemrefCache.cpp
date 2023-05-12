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
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    getOperation()->walk([&](Operation *op) {
      if (auto initOp = dyn_cast<enzyme::InitOp>(op)) {

        if (auto c1 = dyn_cast<enzyme::CacheType>(initOp.getType())) {
          if (auto memref = dyn_cast<MemRefType>(c1.getType())) {
            if (auto c2 =
                    dyn_cast<enzyme::CacheType>(memref.getElementType())) {
              mlir::MemRefType::Builder memrefTypeBuilder(memref);
              memrefTypeBuilder.setElementType(c2.getType());
              Type newType = memrefTypeBuilder;
              Type newCacheType = enzyme::CacheType::get(context, newType);

              for (auto user : op->getUsers()) {
                if (auto pushOp = dyn_cast<enzyme::PushOp>(user)) {
                  auto v = pushOp.getValue();
                  auto definingOp = v.getDefiningOp();
                  if (auto allocOp = dyn_cast<memref::AllocOp>(definingOp)) {
                    OpBuilder allocBuilder(allocOp);
                    auto newAllocOp = allocBuilder.create<memref::AllocOp>(
                        allocOp.getLoc(), dyn_cast<MemRefType>(newType),
                        allocOp.getDynamicSizes(), allocOp.getSymbolOperands(),
                        allocOp.getAlignmentAttr());

                    // TODO: handle all the stuff inside linalg.generic
                    for (auto user : allocOp->getUsers()) {
                      if (auto linalgOp = dyn_cast<linalg::GenericOp>(user)) {
                        for (auto output :
                             llvm::enumerate(linalgOp.getOutputs())) {
                          if (output.value() == allocOp) {
                            unsigned outputIndex = linalgOp.getNumInputs() +
                                                   (unsigned)output.index();
                            assert(linalgOp.getRegion()
                                       .getArgument(outputIndex)
                                       .use_empty()); // We should never
                                                      // actually use the value
                                                      // of the output!
                            linalgOp.getRegion().eraseArgument(outputIndex);
                            linalgOp.getRegion().insertArgument(
                                outputIndex, c2.getType(),
                                output.value().getLoc());

                            Value cache =
                                linalgOp.getRegion()
                                    .front()
                                    .getTerminator()
                                    ->getOperand((unsigned)output.index());
                            for (auto user : cache.getUsers()) {
                              if (auto pushOp =
                                      dyn_cast<enzyme::PushOp>(user)) {
                                linalgOp.getRegion()
                                    .front()
                                    .getTerminator()
                                    ->setOperand((unsigned)output.index(),
                                                 pushOp.getValue());
                                pushOp.erase();
                              } else if (auto popOp =
                                             dyn_cast<enzyme::PopOp>(user)) {
                                llvm_unreachable(
                                    "PopOp should not be used in forward pass");
                              }
                            }
                            cache.getDefiningOp()->erase();
                          }
                        }
                      } else if (auto pushOp = dyn_cast<enzyme::PushOp>(user)) {
                        // Do nothing
                      } else {
                        llvm_unreachable("Unknown user of memref<CacheType>");
                      }
                    }

                    allocOp.replaceAllUsesWith((Value)newAllocOp);
                    allocOp.erase();
                  } else {
                    llvm_unreachable("Unknown memref initialization");
                  }
                } else if (auto popOp = dyn_cast<enzyme::PopOp>(user)) {
                  OpBuilder popBuilder(popOp);
                  auto newPopOp = popBuilder.create<enzyme::PopOp>(
                      popOp.getLoc(), newType, popOp.getCache());

                  // TODO: handle all the stuff inside linalg.generic
                  for (auto user : popOp->getUsers()) {
                    if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
                      for (auto user : subviewOp->getUsers()) {
                        if (auto linalgOp = dyn_cast<linalg::GenericOp>(user)) {
                          for (auto input :
                               llvm::enumerate(linalgOp.getInputs())) {
                            if (input.value() == subviewOp) {
                              unsigned inputIndex = (unsigned)input.index();
                              Value inputCacheSSA =
                                  linalgOp.getRegion().insertArgument(
                                      inputIndex, c2.getType(),
                                      input.value().getLoc());
                              Value oldArg = linalgOp.getRegion().getArgument(
                                  inputIndex + 1);
                              for (auto user : oldArg.getUsers()) {
                                if (auto popOp =
                                        dyn_cast<enzyme::PopOp>(user)) {
                                  popOp.replaceAllUsesWith(inputCacheSSA);
                                  popOp.erase();
                                } else {
                                  llvm_unreachable("Unknown user");
                                }
                              }

                              linalgOp.getRegion().eraseArgument(
                                  inputIndex + 1); // +1 because we inserted an
                                                   // argument above
                            }
                          }
                        }
                      }

                      // Replace Subview Op
                      OpBuilder subviewBuilder(subviewOp);
                      auto newSubviewOp =
                          subviewBuilder.create<memref::SubViewOp>(
                              subviewOp.getLoc(), newPopOp,
                              subviewOp.getOffsets(), subviewOp.getSizes(),
                              subviewOp.getStrides());
                      subviewOp.replaceAllUsesWith((Value)newSubviewOp);
                      subviewOp.erase();
                    }
                  }
                  popOp.replaceAllUsesWith((Value)newPopOp);
                  popOp.erase();
                } else {
                  llvm_unreachable("Unknown user of InitOp");
                }
              }

              OpBuilder builder(op);
              auto newInit =
                  builder.create<enzyme::InitOp>(op->getLoc(), newCacheType);
              op->replaceAllUsesWith(newInit);

              op->erase();
            }
          }
        }
      }
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
