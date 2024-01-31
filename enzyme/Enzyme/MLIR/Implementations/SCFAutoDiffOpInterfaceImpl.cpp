//===- SCFAutoDiffOpInterfaceImpl.cpp - Interface external model ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR SCF dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/EnzymeLogic.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/SCFDerivatives.inc"

struct ForOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<ForOpInterfaceReverse,
                                                       scf::ForOp> {
  void createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                MGradientUtilsReverse *gutils,
                                SmallVector<Value> caches) const {
    auto forOp = cast<scf::ForOp>(op);

    SmallVector<Value> nArgs;
    for (Value v : forOp.getResults()) {
      if (auto iface = dyn_cast<AutoDiffTypeInterface>(v.getType())) {
        if (gutils->hasInvertPointer(v)) {
          nArgs.push_back(gutils->invertPointerM(v, builder));
        } else {
          nArgs.push_back(iface.createNullValue(builder, v.getLoc()));
        }
      }
    }

    auto repFor = builder.create<scf::ForOp>(
        forOp.getLoc(), gutils->popCache(caches[0], builder),
        gutils->popCache(caches[1], builder),
        gutils->popCache(caches[2], builder), nArgs); // TODO
    repFor.getRegion().begin()->erase();

    auto buildFuncReturnOp = [](OpBuilder &builder, Location loc,
                                SmallVector<Value> retargs) {
      builder.create<scf::YieldOp>(loc, retargs);
      return;
    };

    gutils->Logic.differentiate(gutils, forOp.getRegion(), repFor.getRegion(),
                                /*parentRegion=*/false, buildFuncReturnOp,
                                nullptr);

    // Insert the index which is carried by the scf for op.
    Type indexType = IndexType::get(builder.getContext());
    repFor.getRegion().insertArgument((unsigned)0, indexType, forOp.getLoc());

    for (const auto &[iterOperand, adjResult] :
         llvm::zip(forOp.getInitArgs(), repFor.getResults())) {
      if (gutils->hasInvertPointer(iterOperand)) {
        auto autoDiffType = cast<AutoDiffTypeInterface>(iterOperand.getType());
        Value before = gutils->invertPointerM(iterOperand, builder);
        Value after = autoDiffType.createAddOp(builder, forOp.getLoc(), before,
                                               adjResult);
        gutils->mapInvertPointer(iterOperand, after, builder);
      }
    }
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto forOp = cast<scf::ForOp>(op);

    Operation *newOp = gutils->getNewFromOriginal(op);
    OpBuilder cacheBuilder(newOp);
    SmallVector<Value> caches;

    Value cacheLB = gutils->initAndPushCache(
        gutils->getNewFromOriginal(forOp.getLowerBound()), cacheBuilder);
    caches.push_back(cacheLB);

    Value cacheUB = gutils->initAndPushCache(
        gutils->getNewFromOriginal(forOp.getUpperBound()), cacheBuilder);
    caches.push_back(cacheUB);

    Value cacheStep = gutils->initAndPushCache(
        gutils->getNewFromOriginal(forOp.getStep()), cacheBuilder);
    caches.push_back(cacheStep);

    return caches;
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    // auto forOp = cast<scf::ForOp>(op);
  }
};

} // namespace

void mlir::enzyme::registerSCFDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, scf::SCFDialect *) {
    registerInterfaces(context);
    scf::ForOp::attachInterface<ForOpInterfaceReverse>(*context);
  });
}
