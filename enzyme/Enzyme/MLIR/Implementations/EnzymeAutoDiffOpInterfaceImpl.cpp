//===- ArithAutoDiffOpInterfaceImpl.cpp - Interface external model --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR arithmetic dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "Dialect/Dialect.h"
#include "Dialect/Ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/TypeSupport.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/EnzymeDerivatives.inc"

// The primal op atomically performs old = m[i]; m[i] = old + v and yields
// old. Backward, the location's adjoint serves both reads: the added value
// takes it (d_v += dm[i]), and it keeps flowing to whatever wrote m[i]
// before -- the atomic changed the value, not which value the location
// stands for. A used result read the pre-add value, so its adjoint joins
// the location's, atomically for the same reason the primal add was atomic.
// The load of the location's adjoint is plain: the reverse sweep mirrors
// the forward structure, so the barriers that ordered the primal's atomics
// against this thread's reads order the shadow's atomics against this load.
struct AffineAtomicRMWOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          AffineAtomicRMWOpInterfaceReverse, enzyme::AffineAtomicRMWOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto rmwOp = cast<enzyme::AffineAtomicRMWOp>(op);
    Value memref = rmwOp.getMemref();
    Value val = rmwOp.getValue();

    if (gutils->isConstantValue(memref)) {
      // Inactive memory: the value folds into it and the result read from it
      // carries nothing out.
      if (!gutils->isConstantValue(rmwOp.getResult()))
        gutils->zeroDiffe(rmwOp.getResult(), builder);
      return success();
    }

    if (rmwOp.getKind() != arith::AtomicRMWKind::addf)
      return op->emitError()
             << "could not compute the adjoint of a non-add atomic rmw " << *op;

    auto iface = dyn_cast<AutoDiffTypeInterface>(val.getType());
    if (!iface)
      return op->emitError()
             << "could not compute the adjoint of an atomic rmw on a type "
                "without autodiff semantics "
             << *op;

    Value memrefGradient = gutils->popCache(caches.front(), builder);
    SmallVector<Value> retrievedArguments;
    for (Value cache : ValueRange(caches).drop_front(1))
      retrievedArguments.push_back(gutils->popCache(cache, builder));

    auto alignAttr = rmwOp.getAlignmentAttr();
    if (!gutils->isConstantValue(val)) {
      auto loadedGradient = affine::AffineLoadOp::create(
          builder, rmwOp.getLoc(), memrefGradient, rmwOp.getMap(),
          ArrayRef<Value>(retrievedArguments));
      if (alignAttr)
        loadedGradient->setAttr("alignment", alignAttr);
      gutils->addToDiffe(val, loadedGradient, builder);
    }

    if (!gutils->isConstantValue(rmwOp.getResult())) {
      Value gradient = gutils->diffe(rmwOp.getResult(), builder);
      gutils->zeroDiffe(rmwOp.getResult(), builder);
      setDerivativeFastMath(enzyme::AffineAtomicRMWOp::create(
          builder, rmwOp.getLoc(), gradient.getType(),
          arith::AtomicRMWKind::addf, gradient, memrefGradient,
          retrievedArguments, rmwOp.getMap(), alignAttr));
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto rmwOp = cast<enzyme::AffineAtomicRMWOp>(op);
    if (gutils->isConstantValue(rmwOp.getMemref()))
      return {};
    OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
    SmallVector<Value> caches;
    caches.push_back(gutils->initAndPushCache(
        gutils->invertPointerM(rmwOp.getMemref(), cacheBuilder), cacheBuilder));
    for (Value v : rmwOp.getIndices())
      caches.push_back(gutils->initAndPushCache(gutils->getNewFromOriginal(v),
                                                cacheBuilder));
    return caches;
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

// Forward mode mirrors the primal on the shadows: the tangent of the value
// joins the location's tangent through the same atomic add, and what the
// primal read out (the pre-add value) has as its tangent what the shadow
// location held before the shadow's add -- exactly what the mirrored atomic
// returns.
struct AffineAtomicRMWOpForwardInterface
    : public AutoDiffOpInterface::ExternalModel<
          AffineAtomicRMWOpForwardInterface, enzyme::AffineAtomicRMWOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto rmwOp = cast<enzyme::AffineAtomicRMWOp>(op);
    auto iface = dyn_cast<AutoDiffTypeInterface>(rmwOp.getValue().getType());
    if (!iface)
      return op->emitError()
             << "could not compute the tangent of an atomic rmw on a type "
                "without autodiff semantics "
             << *op;

    if (gutils->isConstantValue(rmwOp.getMemref())) {
      // Inactive memory: the value's tangent folds into it and the result
      // read from it carries a zero tangent out.
      if (!gutils->isConstantValue(rmwOp.getResult()))
        gutils->setDiffe(rmwOp.getResult(),
                         iface.createNullValue(builder, op->getLoc()), builder);
      return success();
    }

    if (rmwOp.getKind() != arith::AtomicRMWKind::addf)
      return op->emitError()
             << "could not compute the tangent of a non-add atomic rmw " << *op;

    Value shadowMemref = gutils->invertPointerM(rmwOp.getMemref(), builder);
    Value shadowVal = gutils->isConstantValue(rmwOp.getValue())
                          ? iface.createNullValue(builder, op->getLoc())
                          : gutils->invertPointerM(rmwOp.getValue(), builder);
    auto newOp =
        cast<enzyme::AffineAtomicRMWOp>(gutils->getNewFromOriginal(op));
    SmallVector<Value> shadowIndices;
    for (Value v : newOp.getIndices())
      shadowIndices.push_back(v);
    auto shadowOp = enzyme::AffineAtomicRMWOp::create(
        builder, rmwOp.getLoc(), shadowVal.getType(),
        arith::AtomicRMWKind::addf, shadowVal, shadowMemref, shadowIndices,
        rmwOp.getMap(), rmwOp.getAlignmentAttr());
    shadowOp->setAttr("fastmath", newOp.getFastmathAttr());
    if (!gutils->isConstantValue(rmwOp.getResult()))
      gutils->setDiffe(rmwOp.getResult(), shadowOp.getResult(), builder);
    return success();
  }
};

} // namespace

void mlir::enzyme::registerEnzymeDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, enzyme::EnzymeDialect *) {
    registerInterfaces(context);
    // The rules build affine loads and atomic adds over affine maps.
    context->getOrLoadDialect<affine::AffineDialect>();
    enzyme::AffineAtomicRMWOp::attachInterface<
        AffineAtomicRMWOpInterfaceReverse>(*context);
    enzyme::AffineAtomicRMWOp::attachInterface<
        AffineAtomicRMWOpForwardInterface>(*context);
  });
}
